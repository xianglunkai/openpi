from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import pickle
import random
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

"""
Offline actor/critic training from a replay journal.

Inputs:

- data is selected by `--replay-path`
- algorithm/runtime config comes from the same task YAML used online
  (`configs/tasks/<task>/online_rl.yaml`, or `--config`)
- replay subsets are selected by `--phase` and `--source`

The script:

1. reads the replay journal
2. filters samples by `phase/source`
3. initializes actor and critic from scratch
4. runs offline training
5. periodically evaluates train/validation actor fit
6. exports snapshots, checkpoints, and a manifest (no forked config copy)
"""

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from _common import PHASE_CHOICES
from _common import SOURCE_CHOICES
from _common import default_filter_suffix
from _common import filter_replay_records
from _common import infer_task_dir_from_replay_path
from _common import write_replay_journal
from _common import JOINT_LABELS

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.action_representation import jax_denormalize_to_abs_chunk
from rlt_online_rl.config import OnlineRLSystemConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.config import relativize_rl_config_paths
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import PyTree
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.critic_loss import compute_critic_loss
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.trainer import RLTTrainState
from rlt_online_rl.trainer import init_train_state
from rlt_online_rl.trainer import soft_update_targets


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-train actor/critic from a replay journal with custom BC/Q weights."
    )
    parser.add_argument(
        "--replay-path",
        type=Path,
        required=True,
        help="Replay journal to train from, usually runs/<task>/replay/replay_journal.pkl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Task YAML shared with online RL (default: configs/tasks/<task>/online_rl.yaml).",
    )
    # None → take value from the shared task YAML (or phase-specific fields).
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bc-weight", type=float, default=None)
    parser.add_argument("--q-weight", type=float, default=None)
    parser.add_argument("--delta-weight", type=float, default=None)
    parser.add_argument("--fixed-std", type=float, default=None)
    parser.add_argument("--actor-hidden-dim", type=int, default=None)
    parser.add_argument("--actor-num-layers", type=int, default=None)
    parser.add_argument("--critic-hidden-dim", type=int, default=None)
    parser.add_argument("--critic-num-layers", type=int, default=None)
    parser.add_argument("--reference-dropout-prob", type=float, default=None)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument(
        "--disable-ref-input",
        action="store_true",
        help="Train/evaluate the actor with ref_chunk input replaced by zeros.",
    )
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--source", choices=tuple(SOURCE_CHOICES), default="all")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional offline output directory.")
    return parser.parse_args()


def _default_task_config_path(task_dir: Path) -> Path:
    return ROOT / "configs" / "tasks" / task_dir.name / "online_rl.yaml"


def _resolve_task_config_path(task_dir: Path, config_arg: Path | None) -> Path:
    if config_arg is not None:
        path = config_arg.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config not found: {path}")
        return path
    path = _default_task_config_path(task_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"Shared task config not found: {path}. Pass --config or create configs/tasks/{task_dir.name}/online_rl.yaml"
        )
    return path


def _load_system_config(config_path: Path) -> OnlineRLSystemConfig:
    return load_system_config_yaml(str(config_path))


def _apply_cli_rl_overrides(rl_config: RLTOnlineRLConfig, args: argparse.Namespace) -> RLTOnlineRLConfig:
    updates: dict[str, Any] = {}
    for field_name, attr in (
        ("fixed_std", "fixed_std"),
        ("actor_hidden_dim", "actor_hidden_dim"),
        ("actor_num_layers", "actor_num_layers"),
        ("critic_hidden_dim", "critic_hidden_dim"),
        ("critic_num_layers", "critic_num_layers"),
        ("reference_dropout_prob", "reference_dropout_prob"),
        ("delta_weight", "delta_weight"),
    ):
        value = getattr(args, attr)
        if value is not None:
            updates[field_name] = value
    return dataclasses.replace(rl_config, **updates) if updates else rl_config


def _resolve_train_weights(
    rl_config: RLTOnlineRLConfig,
    *,
    phase: str,
    bc_weight: float | None,
    q_weight: float | None,
) -> tuple[float, float]:
    if phase == "online":
        default_bc, default_q = float(rl_config.online_bc_weight), float(rl_config.online_q_weight)
    else:
        # warmup / all / unknown → warmup weights (offline demo BCQ).
        default_bc, default_q = float(rl_config.warmup_bc_weight), float(rl_config.warmup_q_weight)
    return (
        float(default_bc if bc_weight is None else bc_weight),
        float(default_q if q_weight is None else q_weight),
    )


def _load_replay_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("rb") as f:
        try:
            while True:
                item = pickle.load(f)
                if isinstance(item, dict):
                    records.append(item)
        except EOFError:
            pass
    if not records:
        raise RuntimeError(f"No replay records found in {path}")
    return records


def _stack_records(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    keys = (
        "z_rl",
        "proprio",
        "ref_chunk",
        "action_chunk",
        "rewards",
        "done",
        "next_z_rl",
        "next_proprio",
        "next_ref_chunk",
        "source",
        "success",
        "intervention_flag",
        "episode_id",
        "step_id",
    )
    batch: dict[str, np.ndarray] = {}
    for key in keys:
        batch[key] = np.stack([np.asarray(record[key]) for record in records], axis=0)
    batch["source_chunk"] = np.stack(
        [
            np.asarray(
                record.get(
                    "source_chunk",
                    np.full((np.asarray(record["ref_chunk"]).shape[0],), int(record["source"]), dtype=np.uint8),
                ),
                dtype=np.uint8,
            )
            for record in records
        ],
        axis=0,
    )
    batch["done"] = batch["done"].astype(np.float32, copy=False)
    batch["source"] = batch["source"].astype(np.uint8, copy=False)
    batch["source_chunk"] = batch["source_chunk"].astype(np.uint8, copy=False)
    batch["success"] = batch["success"].astype(np.int8, copy=False)
    batch["intervention_flag"] = batch["intervention_flag"].astype(np.bool_, copy=False)
    batch["episode_id"] = batch["episode_id"].astype(np.int32, copy=False)
    batch["step_id"] = batch["step_id"].astype(np.int32, copy=False)
    if "actual_steps" in records[0]:
        batch["actual_steps"] = np.stack(
            [np.asarray(record["actual_steps"], dtype=np.int32) for record in records],
            axis=0,
        )
    else:
        batch["actual_steps"] = np.full(
            (len(records),),
            np.asarray(records[0]["ref_chunk"]).shape[0],
            dtype=np.int32,
        )
    return batch


def _split_dataset(
    dataset: dict[str, np.ndarray], *, val_ratio: float, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    size = dataset["z_rl"].shape[0]
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    val_size = max(1, int(size * val_ratio))
    val_idx = np.asarray(indices[:val_size], dtype=np.int32)
    train_idx = np.asarray(indices[val_size:], dtype=np.int32)
    if train_idx.size == 0:
        raise RuntimeError("Validation split consumed the entire dataset.")
    train = {key: value[train_idx] for key, value in dataset.items()}
    val = {key: value[val_idx] for key, value in dataset.items()}
    return train, val


def _sample_batch(
    dataset: dict[str, np.ndarray], *, batch_size: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    size = dataset["z_rl"].shape[0]
    indices = rng.integers(0, size, size=batch_size, endpoint=False)
    return {key: value[indices] for key, value in dataset.items()}


def _maybe_zero_model_ref_input(ref_chunk: np.ndarray, *, disable_ref_input: bool) -> np.ndarray:
    ref_chunk = np.asarray(ref_chunk, dtype=np.float32)
    if disable_ref_input:
        return np.zeros_like(ref_chunk, dtype=np.float32)
    return ref_chunk


def _predict_abs_chunk_batch(
    actor: ChunkActor,
    actor_params: PyTree,
    adapter: ActionRepresentationAdapter | None,
    batch: dict[str, np.ndarray],
    *,
    disable_ref_input: bool = False,
) -> np.ndarray:
    z_rl = np.asarray(batch["z_rl"], dtype=np.float32)
    proprio = np.asarray(batch["proprio"], dtype=np.float32)
    ref_chunk = np.asarray(batch["ref_chunk"], dtype=np.float32)
    model_ref_chunk = adapter.normalize_ref_chunk(ref_chunk, proprio) if adapter is not None else ref_chunk
    model_ref_chunk = _maybe_zero_model_ref_input(model_ref_chunk, disable_ref_input=disable_ref_input)
    pred = actor.actor_mean(actor_params, jnp.asarray(z_rl), jnp.asarray(proprio), jnp.asarray(model_ref_chunk))
    pred_np = np.asarray(jax.device_get(pred), dtype=np.float32)
    if adapter is not None:
        pred_np = adapter.denormalize_to_abs_chunk(pred_np, proprio)
    return pred_np


def _evaluate_fit(
    actor: ChunkActor,
    actor_params: PyTree,
    adapter: ActionRepresentationAdapter | None,
    dataset: dict[str, np.ndarray],
    *,
    batch_size: int = 256,
    disable_ref_input: bool = False,
) -> dict[str, float]:
    total_abs = []
    size = dataset["z_rl"].shape[0]
    for start in range(0, size, batch_size):
        end = min(start + batch_size, size)
        batch = {key: value[start:end] for key, value in dataset.items()}
        pred_abs = _predict_abs_chunk_batch(actor, actor_params, adapter, batch, disable_ref_input=disable_ref_input)
        ref_abs = np.asarray(batch["ref_chunk"], dtype=np.float32)
        total_abs.append(np.abs(pred_abs - ref_abs))
    abs_delta = np.concatenate(total_abs, axis=0)
    metrics = {
        "mean_abs_delta": float(abs_delta.mean()),
        "median_abs_delta": float(np.median(abs_delta)),
        "p95_abs_delta": float(np.percentile(abs_delta, 95)),
        "max_abs_delta": float(abs_delta.max()),
    }
    for joint_idx in range(abs_delta.shape[-1]):
        joint_vals = abs_delta[:, :, joint_idx]
        # JOINT_LABELS: joint1..joint6, gripper (dim 6) — must match visualize_offline_training.
        label = JOINT_LABELS[joint_idx] if joint_idx < len(JOINT_LABELS) else f"joint{joint_idx + 1}"
        metrics[f"{label}_mean_abs_delta"] = float(joint_vals.mean())
        metrics[f"{label}_max_abs_delta"] = float(joint_vals.max())
    return metrics


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


def _relativize_path(path_value: str | Path | None, anchor_path: Path) -> str | None:
    if path_value is None:
        return None
    candidate = Path(path_value).expanduser()
    if not candidate.is_absolute():
        return str(candidate)
    try:
        candidate.relative_to(ROOT)
    except ValueError:
        return os.path.relpath(str(candidate), start=str(anchor_path.resolve().parent))
    return os.path.relpath(str(candidate), start=str(ROOT))


def _portable_rl_config_dict(rl_config: RLTOnlineRLConfig, anchor_path: Path) -> dict[str, Any]:
    portable = relativize_rl_config_paths(rl_config, str(anchor_path))
    return dataclasses.asdict(portable)


def _save_actor_snapshot(path: Path, version: int, rl_config: RLTOnlineRLConfig, actor_params: PyTree) -> None:
    payload = {
        "version": int(version),
        "rl_config": _portable_rl_config_dict(rl_config, path),
        "actor_params": jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), actor_params),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_critic_snapshot(path: Path, version: int, rl_config: RLTOnlineRLConfig, critic_params: PyTree) -> None:
    payload = {
        "version": int(version),
        "rl_config": _portable_rl_config_dict(rl_config, path),
        "critic_params": jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), critic_params),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _tree_to_numpy(tree: Any) -> Any:
    return jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), tree)


def _warmup_required_updates(rl_config: RLTOnlineRLConfig, warmup_ready_adds_total: int) -> int:
    if rl_config.warmup_post_collect_updates is not None:
        return int(rl_config.warmup_post_collect_updates)
    return int(warmup_ready_adds_total) * int(rl_config.grad_updates_per_cycle)


def _save_online_checkpoint_bundle(
    *,
    output_dir: Path,
    rl_config: RLTOnlineRLConfig,
    state: RLTTrainState,
    replay_size: int,
) -> tuple[Path, Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    actor_snapshot_dir = output_dir / "actor_snapshot"
    metrics_dir = output_dir / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    actor_snapshot_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    global_step = int(jax.device_get(state.global_step))
    actor_version = int(jax.device_get(state.actor_version))
    warmup_ready_adds_total = int(replay_size)
    warmup_required_updates = _warmup_required_updates(rl_config, warmup_ready_adds_total)
    learner_ready_for_online = global_step >= warmup_required_updates
    latest_path = checkpoint_dir / "latest.pkl"
    step_path = checkpoint_dir / f"step_{global_step}.pkl"

    checkpoint_payload = {
        "rl_config": _portable_rl_config_dict(rl_config, latest_path),
        "state": {
            "actor_params": _tree_to_numpy(state.actor_params),
            "target_actor_params": _tree_to_numpy(state.target_actor_params),
            "critic_params": _tree_to_numpy(state.critic_params),
            "target_critic_params": _tree_to_numpy(state.target_critic_params),
            "actor_opt_state": _tree_to_numpy(state.actor_opt_state),
            "critic_opt_state": _tree_to_numpy(state.critic_opt_state),
            "rng": _tree_to_numpy(state.rng),
            "global_step": global_step,
            "actor_version": actor_version,
        },
        "progress": {
            "warmup_ready_adds_total": warmup_ready_adds_total,
        },
    }
    with latest_path.open("wb") as f:
        pickle.dump(checkpoint_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with step_path.open("wb") as f:
        pickle.dump(checkpoint_payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    actor_snapshot_path = actor_snapshot_dir / "actor_snapshot.pkl"
    _save_actor_snapshot(actor_snapshot_path, actor_version, rl_config, state.actor_params)

    learner_status = {
        "global_step": global_step,
        "actor_version": actor_version,
        "replay_size": replay_size,
        "adds_total": replay_size,
        "pending_update_budget": 0,
        "warmup_ready_adds_total": warmup_ready_adds_total,
        "warmup_required_updates": warmup_required_updates,
        "warmup_post_collect_updates": rl_config.warmup_post_collect_updates,
        "ready_for_online": learner_ready_for_online,
        "training_frozen": False,
    }
    _atomic_write_json(metrics_dir / "learner_status.json", learner_status)
    return latest_path, actor_snapshot_path, metrics_dir / "learner_status.json"


def _save_bundle_config(
    *,
    output_dir: Path,
    run_dir: Path,
    rl_config: RLTOnlineRLConfig,
    canonical_config_path: Path,
) -> Path | None:
    """Record the shared task YAML path; do not fork a second online_rl_config.yaml.

    Online and warmup both use ``canonical_config_path`` (typically
    ``configs/tasks/<task>/online_rl.yaml``). Paths in that file already point at
    ``runs/<task>/...``; deploy copies weights there without rewriting config.
    """
    del output_dir, run_dir, rl_config  # kept for call-site compatibility
    if not canonical_config_path.is_file():
        return None
    return canonical_config_path.resolve()


def _write_filtered_replay(output_dir: Path, records: list[dict[str, Any]]) -> Path:
    replay_dir = output_dir / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    target_path = replay_dir / "replay_journal.pkl"
    write_replay_journal(target_path, records)
    return target_path


def _write_bundle_manifest(
    *,
    output_dir: Path,
    run_dir: Path,
    source_replay_path: Path,
    bundle_replay_path: Path,
    rl_config: RLTOnlineRLConfig,
    checkpoint_path: Path,
    actor_snapshot_path: Path,
    learner_status_path: Path,
    bundle_config_path: Path | None,
    state: RLTTrainState,
    published_model_tag: str,
) -> None:
    payload = {
        "format_version": 1,
        "source_run_dir": _relativize_path(run_dir, output_dir / "manifest.json"),
        "source_replay_path": _relativize_path(source_replay_path, output_dir / "manifest.json"),
        "bundle_replay_path": _relativize_path(bundle_replay_path, output_dir / "manifest.json"),
        "published_model_tag": published_model_tag,
        "rl_config": _portable_rl_config_dict(rl_config, output_dir / "manifest.json"),
        "global_step": int(jax.device_get(state.global_step)),
        "actor_version": int(jax.device_get(state.actor_version)),
        "checkpoint_path": _relativize_path(checkpoint_path, output_dir / "manifest.json"),
        "actor_snapshot_path": _relativize_path(actor_snapshot_path, output_dir / "manifest.json"),
        "learner_status_path": _relativize_path(learner_status_path, output_dir / "manifest.json"),
        # Shared with online: configs/tasks/<task>/online_rl.yaml (not a forked copy).
        "bundle_config_path": None
        if bundle_config_path is None
        else _relativize_path(bundle_config_path, output_dir / "manifest.json"),
        "canonical_config_path": None
        if bundle_config_path is None
        else _relativize_path(bundle_config_path, output_dir / "manifest.json"),
    }
    _atomic_write_json(output_dir / "manifest.json", payload)


def _custom_actor_loss(
    actor: ChunkActor,
    actor_params: PyTree,
    critic: TwinCritic,
    critic_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    ref_chunk: jax.Array,
    behavior_chunk: jax.Array,
    source_chunk: jax.Array,
    *,
    bc_weight: float,
    q_weight: float,
    delta_weight: float,
    reference_dropout_prob: float,
    disable_ref_input: bool,
    rng: jax.Array,
    use_action_adapter: bool,
    action_q01: jax.Array | None,
    action_q99: jax.Array | None,
    action_representation: str,
    bc_reduction: str,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Actor loss: L = w_bc * BC - w_q * mean(Q) + w_delta * Delta.

    Shape legend (screw_sorting defaults as example):
      B = batch size, e.g. 256
      C = chunk_len, e.g. 10
      A = action_dim, e.g. 7  (xyz+rot+gripper; delta uses first 6)
      z_rl:          (B, 2048)
      proprio:       (B, 7)
      ref_chunk:     (B, C, A)  VLA reference actions
      behavior_chunk:(B, C, A)  actually executed / demo actions
      source_chunk:  (B, C)     per-step TransitionSource enum
    """
    # --- 1) Reference conditioning for the actor ---
    # Binary ref dropout per batch element (same mask for all C*A of that sample).
    # keep_mask: (B, 1, 1) broadcasts over (B, C, A).
    # Example: B=2, p_drop=0.5 → keep_mask maybe [[1],[0]] → sample0 keeps ref, sample1 zeros ref.
    dropout_rng, _ = jax.random.split(rng)
    if reference_dropout_prob > 0.0:
        keep_mask = jax.random.bernoulli(
            dropout_rng, 1.0 - reference_dropout_prob, (ref_chunk.shape[0], 1, 1)
        )
        dropped_ref = ref_chunk * keep_mask.astype(ref_chunk.dtype)  # (B, C, A)
    else:
        dropped_ref = ref_chunk
    # Optional ablation: feed zeros so actor cannot rely on VLA ref at all.
    model_ref_input = jnp.zeros_like(dropped_ref) if disable_ref_input else dropped_ref  # (B, C, A)

    # π_θ(z, proprio, ref) → action mean; Q_φ twin-min of that action.
    action_chunk = actor.actor_mean(actor_params, z_rl, proprio, model_ref_input)  # (B, C, A)
    q = critic.min_q(critic_params, z_rl, proprio, action_chunk)  # (B,) or (B, 1)

    # --- 2) Per-step human vs policy mask for BC target selection ---
    # human_mask: (B, C) True where that step came from HUMAN or MIXED teleop.
    # Example source_chunk[0] = [POLICY, HUMAN, POLICY, ...] → human_mask[0,1]=True.
    human_mask = jnp.logical_or(
        source_chunk == int(TransitionSource.HUMAN),
        source_chunk == int(TransitionSource.MIXED),
    )  # (B, C)

    # Chunk-level flags for metrics only: any human step → whole sample is "human".
    # human_mask_chunk / policy_mask_chunk: (B,)
    # Example: human_mask[i] has any True → human_mask_chunk[i]=1, policy_mask_chunk[i]=0.
    human_mask_chunk = jnp.any(human_mask, axis=-1).astype(jnp.float32)  # (B,)
    policy_mask_chunk = 1.0 - human_mask_chunk  # (B,)

    # BC target: human/mixed steps → behavior (demo/exec); else → VLA ref.
    # human_mask[..., None]: (B, C, 1) broadcasts over A.
    # Example: step t human → bc_target[i,t]=behavior; else bc_target[i,t]=ref.
    bc_target = jnp.where(human_mask[..., None], behavior_chunk, ref_chunk)  # (B, C, A)

    # --- 3) BC term that enters the loss ---
    # Squared error then sum over (C, A) → per-sample scalar, then mean over B.
    # bc_error: (B, C, A); per_sample_bc: (B,)
    # Example C=10,A=7: sum over 70 dims; "sum" reduction keeps that scale (Evo-RLT/paper β).
    # bc_error = jnp.square(action_chunk - bc_target)  # (B, C, A)
    bc_error = jnp.square(action_chunk - ref_chunk)  # (B, C, A)
    per_sample_bc = jnp.sum(bc_error, axis=(-2, -1))  # (B,)
    if bc_reduction == "sum":
        bc_penalty = jnp.mean(per_sample_bc)  # ()  E[||μ - a_bc||_F^2]
    else:
        chunk_len = bc_error.shape[-2]  # C
        action_dim = bc_error.shape[-1]  # A
        # "mean": divide by C*A so BC magnitude ~ per-dim MSE (e.g. /70).
        bc_penalty = jnp.mean(per_sample_bc) / jnp.maximum(chunk_len * action_dim, 1.0)  # ()

    # --- 4) Split BC metrics (logging only; do not affect gradients via this path) ---
    # Compare μ to pure ref / pure behavior, then average over policy-only or human-only samples.
    ref_error_squared = jnp.square(action_chunk - ref_chunk)  # (B, C, A)
    human_error_squared = jnp.square(action_chunk - behavior_chunk)  # (B, C, A)
    per_sample_ref = jnp.sum(ref_error_squared, axis=(-2, -1))  # (B,)
    per_sample_human = jnp.sum(human_error_squared, axis=(-2, -1))  # (B,)
    if bc_reduction == "mean":
        chunk_len = bc_error.shape[-2]
        action_dim = bc_error.shape[-1]
        scale = jnp.maximum(chunk_len * action_dim, 1.0)
        per_sample_ref = per_sample_ref / scale
        per_sample_human = per_sample_human / scale
    # Masked means: if no policy samples in batch, denom→1 avoids NaN.
    # Example: B=4 with masks [0,1,1,0] → bc_ref averages samples 0,3 only.
    bc_ref_penalty = jnp.sum(per_sample_ref * policy_mask_chunk) / jnp.maximum(
        jnp.sum(policy_mask_chunk), 1.0
    )  # ()
    bc_human_penalty = jnp.sum(per_sample_human * human_mask_chunk) / jnp.maximum(
        jnp.sum(human_mask_chunk), 1.0
    )  # ()
    human_mask_ratio = jnp.mean(human_mask_chunk)  # () fraction of human-tagged samples

    # --- 5) Delta (step-to-step) smoothness penalty ---
    # Optionally map normalized actions to absolute space before differencing.
    # pred/target_abs_chunk: (B, C, A) in either normalized or absolute coords.
    if not use_action_adapter:
        pred_abs_chunk = action_chunk
        target_abs_chunk = bc_target
    else:
        pred_abs_chunk = jax_denormalize_to_abs_chunk(
            action_chunk,
            proprio,
            action_q01,
            action_q99,
            action_representation=action_representation,
        )
        target_abs_chunk = jax_denormalize_to_abs_chunk(
            bc_target,
            proprio,
            action_q01,
            action_q99,
            action_representation=action_representation,
        )
    # Adjacent-step deltas on pose dims only (exclude gripper dim 6).
    # [:, 1:, :6] - [:, :-1, :6] → (B, C-1, 6); e.g. C=10 → (B, 9, 6).
    pred_step_delta = pred_abs_chunk[:, 1:, :6] - pred_abs_chunk[:, :-1, :6]
    target_step_delta = target_abs_chunk[:, 1:, :6] - target_abs_chunk[:, :-1, :6]
    delta_penalty = jnp.mean(jnp.square(pred_step_delta - target_step_delta))  # ()

    # --- 6) Weighted objective ---
    # L = w_bc * BC - w_q * Q̄ + w_delta * Delta
    # Example: w_bc=1, w_q=1, w_delta=0 → classic Evo-style -Q + BC.
    actor_q = jnp.mean(q)  # ()
    weighted_bc = jnp.asarray(bc_weight, dtype=jnp.float32) * bc_penalty
    weighted_q = jnp.asarray(q_weight, dtype=jnp.float32) * actor_q
    weighted_delta = jnp.asarray(delta_weight, dtype=jnp.float32) * delta_penalty
    actor_loss = weighted_bc - weighted_q + weighted_delta  # ()
    metrics = {
        "actor_loss": actor_loss,
        "actor_q": actor_q,
        "bc_penalty": bc_penalty,
        "bc_ref_penalty": bc_ref_penalty,
        "bc_human_penalty": bc_human_penalty,
        "human_mask_ratio": human_mask_ratio,
        "policy_mask_ratio": 1.0 - human_mask_ratio,
        "delta_penalty": delta_penalty,
        "weighted_bc": weighted_bc,
        "weighted_delta": weighted_delta,
        "weighted_q": weighted_q,
    }
    return actor_loss, metrics


def _update_critic(
    state: RLTTrainState,
    batch: dict[str, jax.Array],
    actor: ChunkActor,
    critic: TwinCritic,
    rl_config: RLTOnlineRLConfig,
    *,
    disable_ref_input: bool,
) -> tuple[RLTTrainState, dict[str, jax.Array]]:
    next_ref_chunk = jnp.zeros_like(batch["next_ref_chunk"]) if disable_ref_input else batch["next_ref_chunk"]

    def loss_fn(critic_params: PyTree) -> tuple[jax.Array, dict[str, jax.Array]]:
        return compute_critic_loss(
            critic,
            critic_params,
            actor,
            state.actor_params,
            state.target_critic_params,
            batch["z_rl"],
            batch["proprio"],
            batch["action_chunk"],
            batch["rewards"],
            batch["done"],
            batch["next_z_rl"],
            batch["next_proprio"],
            next_ref_chunk,
            batch["actual_steps"],
            rl_config,
        )

    (critic_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.critic_params)
    updates, critic_opt_state = state.critic_tx.update(grads, state.critic_opt_state, state.critic_params)
    critic_params = optax.apply_updates(state.critic_params, updates)
    new_state = state.replace(critic_params=critic_params, critic_opt_state=critic_opt_state)
    metrics = {**metrics, "critic_loss": critic_loss}
    return new_state, metrics


@jax.jit
def _zero_metrics() -> dict[str, jax.Array]:
    nan = jnp.array(jnp.nan, dtype=jnp.float32)
    return {
        "actor_loss": nan,
        "actor_q": nan,
        "bc_penalty": nan,
        "bc_ref_penalty": nan,
        "bc_human_penalty": nan,
        "human_mask_ratio": nan,
        "policy_mask_ratio": nan,
        "delta_penalty": nan,
        "weighted_bc": nan,
        "weighted_delta": nan,
        "weighted_q": nan,
    }


def _make_train_step(
    actor: ChunkActor,
    critic: TwinCritic,
    rl_config: RLTOnlineRLConfig,
    *,
    bc_weight: float,
    q_weight: float,
    delta_weight: float,
    disable_ref_input: bool,
    use_action_adapter: bool,
    action_q01: jax.Array | None,
    action_q99: jax.Array | None,
):
    @jax.jit
    def train_step(state: RLTTrainState, batch: dict[str, jax.Array]) -> tuple[RLTTrainState, dict[str, jax.Array]]:
        state_after_critic, critic_metrics = _update_critic(
            state,
            batch,
            actor,
            critic,
            rl_config,
            disable_ref_input=disable_ref_input,
        )
        state_after_critic = state_after_critic.replace(
            target_critic_params=soft_update_targets(
                state_after_critic.target_critic_params,
                state_after_critic.critic_params,
                rl_config.target_tau,
            )
        )
        should_update_actor = ((state_after_critic.global_step + 1) % rl_config.actor_update_period) == 0

        def do_actor_update(train_state: RLTTrainState) -> tuple[RLTTrainState, dict[str, jax.Array]]:
            actor_rng, next_rng = jax.random.split(train_state.rng)

            def loss_fn(actor_params: PyTree) -> tuple[jax.Array, dict[str, jax.Array]]:
                return _custom_actor_loss(
                    actor,
                    actor_params,
                    critic,
                    train_state.critic_params,
                    batch["z_rl"],
                    batch["proprio"],
                    batch["ref_chunk"],
                    batch["action_chunk"],
                    batch["source_chunk"],
                    bc_weight=bc_weight,
                    q_weight=q_weight,
                    delta_weight=delta_weight,
                    reference_dropout_prob=rl_config.reference_dropout_prob,
                    disable_ref_input=disable_ref_input,
                    rng=actor_rng,
                    use_action_adapter=use_action_adapter,
                    action_q01=action_q01,
                    action_q99=action_q99,
                    action_representation=rl_config.action_representation,
                    bc_reduction=rl_config.bc_reduction,
                )

            (actor_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.actor_params)
            updates, actor_opt_state = train_state.actor_tx.update(
                grads, train_state.actor_opt_state, train_state.actor_params
            )
            actor_params = optax.apply_updates(train_state.actor_params, updates)
            updated_state = train_state.replace(
                actor_params=actor_params,
                actor_opt_state=actor_opt_state,
                rng=next_rng,
            )
            updated_state = updated_state.replace(
                target_actor_params=soft_update_targets(
                    updated_state.target_actor_params, updated_state.actor_params, rl_config.target_tau
                ),
                actor_version=updated_state.actor_version + 1,
            )
            metrics = {**metrics, "actor_loss": actor_loss}
            return updated_state, metrics

        state_after_actor, actor_metrics = jax.lax.cond(
            should_update_actor,
            do_actor_update,
            lambda train_state: (train_state, _zero_metrics()),
            state_after_critic,
        )
        state_after_actor = state_after_actor.replace(global_step=state_after_actor.global_step + 1)
        metrics = {
            **critic_metrics,
            **actor_metrics,
            "did_actor_update": should_update_actor.astype(jnp.float32),
            "global_step": state_after_actor.global_step.astype(jnp.float32),
            "actor_version": state_after_actor.actor_version.astype(jnp.float32),
        }
        return state_after_actor, metrics

    return train_step


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.resolve()
    task_dir = infer_task_dir_from_replay_path(replay_path)
    config_path = _resolve_task_config_path(task_dir, args.config)
    system = _load_system_config(config_path)
    rl_config = _apply_cli_rl_overrides(system.rl, args)
    bc_weight, q_weight = _resolve_train_weights(
        rl_config,
        phase=args.phase,
        bc_weight=args.bc_weight,
        q_weight=args.q_weight,
    )
    delta_weight = float(rl_config.delta_weight)
    steps = int(
        args.steps
        if args.steps is not None
        else (rl_config.warmup_post_collect_updates or 10000)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else system.learner_service.sample_batch_size
    )
    seed = int(args.seed if args.seed is not None else system.replay.seed)

    default_output_name = "offline_train_bcq"
    if args.disable_ref_input:
        default_output_name += "_noref"
    default_output_name += default_filter_suffix(phase=args.phase, source=args.source)
    output_dir = (
        args.output_dir.resolve() if args.output_dir is not None else (task_dir / default_output_name).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = filter_replay_records(
        _load_replay_records(replay_path),
        phase=args.phase,
        source=args.source,
    )
    if not records:
        raise RuntimeError(f"No replay samples left after filtering: replay={replay_path}")
    dataset = _stack_records(records)
    train_ds, val_ds = _split_dataset(dataset, val_ratio=args.val_ratio, seed=seed)
    collection_phase_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for record in records:
        phase = str(record.get("collection_phase", "unknown"))
        collection_phase_counts[phase] = collection_phase_counts.get(phase, 0) + 1
        source_key = str(int(record["source"]))
        source_counts[source_key] = source_counts.get(source_key, 0) + 1

    state, actor, critic = init_train_state(rl_config, rng=jax.random.PRNGKey(seed))
    adapter = ActionRepresentationAdapter.from_config(rl_config)
    if adapter is None:
        action_q01 = None
        action_q99 = None
    else:
        action_q01 = jnp.asarray(adapter.stats.q01, dtype=jnp.float32)
        action_q99 = jnp.asarray(adapter.stats.q99, dtype=jnp.float32)
    train_step = _make_train_step(
        actor,
        critic,
        rl_config,
        bc_weight=bc_weight,
        q_weight=q_weight,
        delta_weight=delta_weight,
        disable_ref_input=args.disable_ref_input,
        use_action_adapter=adapter is not None,
        action_q01=action_q01,
        action_q99=action_q99,
    )
    rng = np.random.default_rng(seed)

    metrics_path = output_dir / "metrics.jsonl"
    status_path = output_dir / "status.json"
    metrics_path.write_text("", encoding="utf-8")
    best_actor_snapshot_path = output_dir / "best_actor_snapshot.pkl"
    best_critic_snapshot_path = output_dir / "best_critic_snapshot.pkl"
    final_actor_snapshot_path = output_dir / "final_actor_snapshot.pkl"
    final_critic_snapshot_path = output_dir / "final_critic_snapshot.pkl"

    experiment_meta = {
        "task_dir": _relativize_path(task_dir, output_dir / "experiment.json"),
        "replay_path": _relativize_path(replay_path, output_dir / "experiment.json"),
        "config_path": _relativize_path(config_path, output_dir / "experiment.json"),
        "phase": args.phase,
        "source": args.source,
        "collection_phase_counts": collection_phase_counts,
        "source_counts": source_counts,
        "num_records": int(len(records)),
        "train_size": int(train_ds["z_rl"].shape[0]),
        "val_size": int(val_ds["z_rl"].shape[0]),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "seed": int(seed),
        "bc_weight": float(bc_weight),
        "q_weight": float(q_weight),
        "delta_weight": float(delta_weight),
        "fixed_std": float(rl_config.fixed_std),
        "disable_ref_input": bool(args.disable_ref_input),
        "actor_hidden_dim": int(rl_config.actor_hidden_dim),
        "actor_num_layers": int(rl_config.actor_num_layers),
        "critic_hidden_dim": int(rl_config.critic_hidden_dim),
        "critic_num_layers": int(rl_config.critic_num_layers),
        "published_model_tag": "final",
        "rl_config": _portable_rl_config_dict(rl_config, output_dir / "experiment.json"),
    }
    _atomic_write_json(output_dir / "experiment.json", experiment_meta)
    print(
        "offline_train_from_replay "
        f"config={config_path} "
        f"replay={replay_path.name} phase={args.phase} source={args.source} "
        f"train={train_ds['z_rl'].shape[0]} val={val_ds['z_rl'].shape[0]} "
        f"steps={steps} batch={batch_size} bc_weight={bc_weight:.3f} q_weight={q_weight:.3f} "
        f"fixed_std={rl_config.fixed_std:.4f} disable_ref_input={args.disable_ref_input}"
    )

    best_val = float("inf")
    best_step = 0
    for step in range(1, steps + 1):
        batch_np = _sample_batch(train_ds, batch_size=batch_size, rng=rng)
        if adapter is not None:
            batch_np = adapter.prepare_training_batch(batch_np)
        batch = {key: jnp.asarray(value) for key, value in batch_np.items()}
        state, raw_metrics = train_step(state, batch)
        metrics = {key: float(value) for key, value in jax.device_get(raw_metrics).items()}

        if step == 1 or step % args.eval_every == 0 or step == steps:
            actor_params_np = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), state.actor_params)
            train_fit = _evaluate_fit(
                actor, actor_params_np, adapter, train_ds, disable_ref_input=args.disable_ref_input
            )
            val_fit = _evaluate_fit(actor, actor_params_np, adapter, val_ds, disable_ref_input=args.disable_ref_input)
            metrics.update({f"train_{k}": v for k, v in train_fit.items()})
            metrics.update({f"val_{k}": v for k, v in val_fit.items()})
            if val_fit["mean_abs_delta"] < best_val:
                best_val = val_fit["mean_abs_delta"]
                best_step = step
                _save_actor_snapshot(
                    best_actor_snapshot_path, int(metrics["actor_version"]), rl_config, state.actor_params
                )
                _save_critic_snapshot(
                    best_critic_snapshot_path, int(metrics["actor_version"]), rl_config, state.critic_params
                )
                best_tag = " best"
            else:
                best_tag = ""
            _atomic_write_json(
                status_path,
                {
                    "step": step,
                    "best_val_mean_abs_delta": best_val,
                    "best_step": best_step,
                    "latest_metrics": metrics,
                },
            )
            print(
                f"[step {step:>6d}/{steps}] "
                f"critic={metrics['critic_loss']:.4f} "
                f"actor={metrics['actor_loss']:.4f} "
                f"q={metrics['actor_q']:.4f} "
                f"bc={metrics['bc_penalty']:.4f} "
                f"delta={metrics['delta_penalty']:.4f} "
                f"train_fit={metrics['train_mean_abs_delta']:.4f} "
                f"val_fit={metrics['val_mean_abs_delta']:.4f} "
                f"val_max={metrics['val_max_abs_delta']:.4f} "
                f"actor_ver={int(metrics['actor_version'])}{best_tag}"
            )

        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    _save_actor_snapshot(
        final_actor_snapshot_path, int(jax.device_get(state.actor_version)), rl_config, state.actor_params
    )
    _save_critic_snapshot(
        final_critic_snapshot_path, int(jax.device_get(state.actor_version)), rl_config, state.critic_params
    )
    bundle_replay_path = _write_filtered_replay(output_dir, records)
    checkpoint_path, actor_snapshot_path, learner_status_path = _save_online_checkpoint_bundle(
        output_dir=output_dir,
        rl_config=rl_config,
        state=state,
        replay_size=len(records),
    )
    bundle_config_path = _save_bundle_config(
        output_dir=output_dir,
        run_dir=task_dir,
        rl_config=rl_config,
        canonical_config_path=config_path,
    )
    _write_bundle_manifest(
        output_dir=output_dir,
        run_dir=task_dir,
        source_replay_path=replay_path,
        bundle_replay_path=bundle_replay_path,
        rl_config=rl_config,
        checkpoint_path=checkpoint_path,
        actor_snapshot_path=actor_snapshot_path,
        learner_status_path=learner_status_path,
        bundle_config_path=bundle_config_path,
        state=state,
        published_model_tag="final",
    )
    experiment_meta["best_step"] = int(best_step)
    experiment_meta["best_val_mean_abs_delta"] = float(best_val)
    _atomic_write_json(output_dir / "experiment.json", experiment_meta)
    print(f"wrote offline training artifacts to: {output_dir} (shared config={config_path})")


if __name__ == "__main__":
    main()
