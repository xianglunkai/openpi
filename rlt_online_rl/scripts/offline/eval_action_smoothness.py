from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import jax
import matplotlib

matplotlib.use("Agg")
from _common import JOINT_COLORS
from _common import JOINT_LABELS
from _common import PHASE_CHOICES
from _common import SOURCE_CHOICES
from _common import ActionRepresentationAdapter
from _common import RLTPolicyInferenceWrapper
from _common import collection_phase
from _common import default_filter_suffix
from _common import filter_replay_records
from _common import infer_task_dir_from_replay_path
from _common import load_replay_journal
from _common import load_snapshot
from _common import predict_refined_chunk
from _common import resolve_default_actor_snapshot_path
from rlt_online_rl.config import load_system_config_yaml
import matplotlib.pyplot as plt
import numpy as np

"""
Evaluate action-chunk smoothness for an offline BCQ / RL actor on replay data.

Reports (in absolute action space):

1. Intra-chunk smoothness
   - step velocity  ||a_{t+1} - a_t||
   - step acceleration ||Δa_{t+1} - Δa_t||
2. Inter-chunk smoothness (same episode, consecutive journal records)
   - boundary jump ||chunk_i[-1] - chunk_{i+1}[0]||
   - overlap consistency when step_id gap d < chunk_len:
     align chunk_i[d:] vs chunk_{i+1}[:C-d]
3. Comparison vs ground-truth ``action_chunk`` (and optionally ``ref_chunk``)

Plots are written under ``--output-dir``.
"""


POSE_DIM = 6  # exclude gripper for delta-style metrics (matches training delta_penalty)


def _lpf_alpha(dt: float, cutoff_freq: float) -> float:
    """Same alpha as ``examples/mobile_aloha_AgileX/td_filter.LowPassFilter``."""
    if dt < 0.0 or cutoff_freq < 0.0:
        raise ValueError("dt and cutoff_freq must be non-negative")
    if cutoff_freq == 0.0:
        return 1.0
    if dt == 0.0:
        return 0.0
    rc = 1.0 / (2.0 * math.pi * cutoff_freq)
    return dt / (dt + rc)


def _apply_lpf_sequence(
    actions: np.ndarray,
    *,
    cutoff_freq: float,
    dt: float,
    state: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """First-order LPF along time with fixed dt (offline stand-in for wall-clock deploy filter).

    Args:
        actions: (T, A)
        state: previous filtered sample (A,), or None to initialize from actions[0]

    Returns:
        filtered (T, A), last_state (A,)
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] == 0:
        raise ValueError(f"actions must be (T, A) with T>0, got {actions.shape}")
    alpha = float(_lpf_alpha(dt, cutoff_freq))
    out = np.empty_like(actions)
    if state is None:
        y = actions[0].copy()
        out[0] = y
        start = 1
    else:
        y = np.asarray(state, dtype=np.float32).reshape(-1)
        if y.shape[0] != actions.shape[1]:
            raise ValueError(f"state dim {y.shape[0]} != action dim {actions.shape[1]}")
        start = 0
    for t in range(start, actions.shape[0]):
        y = alpha * actions[t] + (1.0 - alpha) * y
        out[t] = y
    return out, y.copy()


def _filter_pred_chunks_episode_continuous(
    replay_records: list[dict[str, Any]],
    pred_chunks: list[np.ndarray],
    *,
    cutoff_freq: float,
    dt: float,
) -> list[np.ndarray]:
    """Apply LPF across each episode in step_id order (state carries between chunks)."""
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records:
        by_episode[int(record["episode_id"])].append(record)
    filtered: list[np.ndarray | None] = [None] * len(pred_chunks)
    for ep_records in by_episode.values():
        ordered = sorted(ep_records, key=lambda r: int(r["step_id"]))
        state: np.ndarray | None = None
        for rec in ordered:
            idx = int(rec["_eval_index"])
            filt, state = _apply_lpf_sequence(
                pred_chunks[idx],
                cutoff_freq=cutoff_freq,
                dt=dt,
                state=state,
            )
            filtered[idx] = filt
    return [np.asarray(x, dtype=np.float32) for x in filtered]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate actor action-chunk smoothness on replay.")
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Task YAML (online_rl.yaml). Supplies default LPF params from runtime.env_driver.",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--disable-ref-input", action="store_true")
    parser.add_argument("--actor-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--actor-seed", type=int, default=0)
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--source", choices=tuple(SOURCE_CHOICES), default="all")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--num-episode-plots", type=int, default=3, help="Stitched episode overlays to draw.")
    parser.add_argument("--topk-jerky", type=int, default=5, help="Most jerky chunks to overlay.")
    parser.add_argument(
        "--simulate-lpf",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Simulate deploy LPF on pred. Default: follow env_driver.use_actions_filter from --config.",
    )
    parser.add_argument(
        "--lpf-cutoff-freq",
        type=float,
        default=None,
        help="LPF cutoff Hz (default: env_driver.action_lpf_cutoff_freq from --config, else 3.0).",
    )
    parser.add_argument(
        "--lpf-dt",
        type=float,
        default=None,
        help="Fixed control dt for offline LPF sim (default: env_driver.action_lpf_dt or 1/control_hz).",
    )
    return parser.parse_args()


def _step_velocity(chunk: np.ndarray, *, pose_only: bool = True) -> np.ndarray:
    """(C-1, A') absolute step deltas."""
    x = chunk[:, :POSE_DIM] if pose_only else chunk
    return np.diff(x, axis=0)


def _step_accel(chunk: np.ndarray, *, pose_only: bool = True) -> np.ndarray:
    """(C-2, A') second differences."""
    return np.diff(_step_velocity(chunk, pose_only=pose_only), axis=0)


def _l2_rows(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x.reshape(x.shape[0], -1), axis=-1)


def _chunk_intra_stats(chunk: np.ndarray) -> dict[str, float]:
    vel = _step_velocity(chunk, pose_only=True)
    acc = _step_accel(chunk, pose_only=True)
    vel_l2 = _l2_rows(vel)
    acc_l2 = _l2_rows(acc)
    vel_abs = np.abs(vel)
    return {
        "intra_vel_mean": float(vel_l2.mean()),
        "intra_vel_p95": float(np.percentile(vel_l2, 95)),
        "intra_vel_max": float(vel_l2.max()),
        "intra_acc_mean": float(acc_l2.mean()) if acc_l2.size else 0.0,
        "intra_acc_p95": float(np.percentile(acc_l2, 95)) if acc_l2.size else 0.0,
        "intra_acc_max": float(acc_l2.max()) if acc_l2.size else 0.0,
        "intra_vel_abs_mean": float(vel_abs.mean()),
        "intra_vel_abs_p95": float(np.percentile(vel_abs, 95)),
    }


def _series_summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _boxplot_compare(
    path: Path,
    series_by_name: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
) -> None:
    names = list(series_by_name.keys())
    data = [np.asarray(series_by_name[n], dtype=np.float64).reshape(-1) for n in names]
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, showfliers=False)
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _hist_compare(
    path: Path,
    series_by_name: dict[str, np.ndarray],
    *,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"pred": "#4c78a8", "gt": "#f58518", "ref": "#54a24b"}
    for name, values in series_by_name.items():
        v = np.asarray(values, dtype=np.float64).reshape(-1)
        ax.hist(v, bins=50, alpha=0.45, label=name, color=colors.get(name, None), density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _joint_vel_boxplot(
    path: Path,
    pred_chunks: list[np.ndarray],
    gt_chunks: list[np.ndarray],
    *,
    title: str,
) -> None:
    pred = np.stack([np.abs(_step_velocity(c, pose_only=False)) for c in pred_chunks], axis=0)
    gt = np.stack([np.abs(_step_velocity(c, pose_only=False)) for c in gt_chunks], axis=0)
    # (N, C-1, A) -> per-joint flatten
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, arr, name in ((axes[0], pred, "pred"), (axes[1], gt, "gt")):
        series = [arr[:, :, j].reshape(-1) for j in range(arr.shape[-1])]
        bp = ax.boxplot(series, tick_labels=JOINT_LABELS, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], JOINT_COLORS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax.set_title(f"{name} |step Δ|")
        ax.set_xlabel("joint")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("|a_{t+1}-a_t|")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _scatter_smooth_vs_fit(
    path: Path,
    pred_intra: np.ndarray,
    gt_intra: np.ndarray,
    fit_mae: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(gt_intra, pred_intra, s=8, alpha=0.35, c="#4c78a8")
    lim = max(float(gt_intra.max()), float(pred_intra.max()), 1e-6)
    axes[0].plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6)
    axes[0].set_xlabel("gt intra vel mean")
    axes[0].set_ylabel("pred intra vel mean")
    axes[0].set_title("Smoothness: pred vs gt")
    axes[0].grid(True, alpha=0.25)

    axes[1].scatter(fit_mae, pred_intra, s=8, alpha=0.35, c="#f58518")
    axes[1].set_xlabel("mean |pred - gt|")
    axes[1].set_ylabel("pred intra vel mean")
    axes[1].set_title("Fit vs smoothness")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_jerky_overlays(
    path: Path,
    rows: list[dict[str, Any]],
    pred_chunks: list[np.ndarray],
    gt_chunks: list[np.ndarray],
    ref_chunks: list[np.ndarray],
    *,
    top_k: int,
    lpf_chunks: list[np.ndarray] | None = None,
) -> None:
    ranked = sorted(rows, key=lambda r: r["pred_intra_acc_max"], reverse=True)[: min(top_k, len(rows))]
    if not ranked:
        return
    fig, axes = plt.subplots(len(ranked), 1, figsize=(12, 3.2 * len(ranked)), squeeze=False)
    for plot_idx, row in enumerate(ranked):
        idx = int(row["index"])
        ax = axes[plot_idx, 0]
        pred = pred_chunks[idx]
        gt = gt_chunks[idx]
        ref = ref_chunks[idx]
        lpf = lpf_chunks[idx] if lpf_chunks is not None else None
        for j, label in enumerate(JOINT_LABELS[:POSE_DIM]):
            ax.plot(gt[:, j], "--", color=JOINT_COLORS[j], alpha=0.4, lw=1.5, label=f"{label} gt" if plot_idx == 0 else None)
            ax.plot(ref[:, j], ":", color=JOINT_COLORS[j], alpha=0.35, lw=1.3, label=f"{label} ref" if plot_idx == 0 else None)
            ax.plot(pred[:, j], "-", color=JOINT_COLORS[j], alpha=0.55, lw=1.6, label=f"{label} pred" if plot_idx == 0 else None)
            if lpf is not None:
                ax.plot(
                    lpf[:, j],
                    "-",
                    color=JOINT_COLORS[j],
                    alpha=0.95,
                    lw=2.2,
                    label=f"{label} pred+lpf" if plot_idx == 0 else None,
                )
        ax.set_title(
            f"idx={idx} ep={row['episode_id']} step={row['step_id']} "
            f"pred_acc_max={row['pred_intra_acc_max']:.4f} fit_mae={row['fit_mae']:.4f}"
        )
        ax.set_xlabel("chunk step")
        ax.set_ylabel("abs action")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_episode_stitch(
    path: Path,
    episode_id: int,
    records: list[dict[str, Any]],
    pred_by_index: dict[int, np.ndarray],
    lpf_by_index: dict[int, np.ndarray] | None = None,
) -> None:
    """Stitch consecutive chunks using first ``gap`` steps of each chunk (stride-aware)."""
    ordered = sorted(
        ((int(r["_eval_index"]), r) for r in records),
        key=lambda item: int(item[1]["step_id"]),
    )
    if len(ordered) < 2:
        return
    times: list[float] = []
    pred_seq: list[np.ndarray] = []
    gt_seq: list[np.ndarray] = []
    lpf_seq: list[np.ndarray] = []
    for i, (idx, rec) in enumerate(ordered):
        pred = pred_by_index[idx]
        gt = np.asarray(rec["action_chunk"], dtype=np.float32)
        lpf = lpf_by_index[idx] if lpf_by_index is not None else None
        step0 = int(rec["step_id"])
        if i + 1 < len(ordered):
            gap = max(int(ordered[i + 1][1]["step_id"]) - step0, 1)
            gap = min(gap, pred.shape[0])
        else:
            gap = pred.shape[0]
        for t in range(gap):
            times.append(float(step0 + t))
            pred_seq.append(pred[t])
            gt_seq.append(gt[t])
            if lpf is not None:
                lpf_seq.append(lpf[t])
    pred_arr = np.stack(pred_seq, axis=0)
    gt_arr = np.stack(gt_seq, axis=0)
    times_arr = np.asarray(times, dtype=np.float32)
    lpf_arr = np.stack(lpf_seq, axis=0) if lpf_seq else None

    fig, axes = plt.subplots(POSE_DIM, 1, figsize=(14, 2.1 * POSE_DIM), sharex=True)
    for j in range(POSE_DIM):
        ax = axes[j]
        ax.plot(times_arr, gt_arr[:, j], "--", color=JOINT_COLORS[j], alpha=0.55, lw=1.6, label="gt")
        ax.plot(times_arr, pred_arr[:, j], "-", color=JOINT_COLORS[j], alpha=0.45, lw=1.5, label="pred")
        if lpf_arr is not None:
            ax.plot(times_arr, lpf_arr[:, j], "-", color=JOINT_COLORS[j], alpha=0.95, lw=2.2, label="pred+lpf")
        for _, rec in ordered:
            ax.axvline(float(rec["step_id"]), color="k", alpha=0.12, lw=0.8)
        ax.set_ylabel(JOINT_LABELS[j])
        ax.grid(True, alpha=0.2)
        if j == 0:
            ax.legend(loc="upper right")
            title = f"Episode {episode_id}: stitched chunks"
            if lpf_arr is not None:
                title += " (pred vs pred+LPF vs gt)"
            ax.set_title(title)
    axes[-1].set_xlabel("step_id")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_boundary_hist(
    path: Path,
    pred_boundary: np.ndarray,
    gt_boundary: np.ndarray,
    pred_overlap: np.ndarray,
    gt_overlap: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(pred_boundary, bins=40, alpha=0.5, label="pred", color="#4c78a8", density=True)
    axes[0].hist(gt_boundary, bins=40, alpha=0.5, label="gt", color="#f58518", density=True)
    axes[0].set_title("Inter-chunk boundary jump ||c_i[-1]-c_{i+1}[0]||")
    axes[0].set_xlabel("L2")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    axes[1].hist(pred_overlap, bins=40, alpha=0.5, label="pred", color="#4c78a8", density=True)
    axes[1].hist(gt_overlap, bins=40, alpha=0.5, label="gt", color="#f58518", density=True)
    axes[1].set_title("Overlap consistency MAE (aligned by step_id gap)")
    axes[1].set_xlabel("MAE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.resolve()
    task_dir = infer_task_dir_from_replay_path(replay_path)
    model_dir = (args.model_dir or (task_dir / "offline_train_bcq")).resolve()
    # LPF defaults from task YAML (same knobs as deploy env_driver).
    lpf_cutoff = 3.0
    lpf_dt = 1.0 / 30.0
    simulate_lpf = False
    config_path = args.config
    if config_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        candidate = repo_root / "configs" / "tasks" / task_dir.name / "online_rl.yaml"
        if candidate.is_file():
            config_path = candidate
    if config_path is not None and Path(config_path).is_file():
        env_cfg = load_system_config_yaml(str(config_path)).env_driver
        simulate_lpf = bool(env_cfg.use_actions_filter)
        lpf_cutoff = float(env_cfg.action_lpf_cutoff_freq)
        if env_cfg.action_lpf_dt is not None:
            lpf_dt = float(env_cfg.action_lpf_dt)
        else:
            lpf_dt = 1.0 / max(float(env_cfg.control_frequency_hz), 1e-6)
    if args.simulate_lpf is not None:
        simulate_lpf = bool(args.simulate_lpf)
    if args.lpf_cutoff_freq is not None:
        lpf_cutoff = float(args.lpf_cutoff_freq)
    if args.lpf_dt is not None:
        lpf_dt = float(args.lpf_dt)
    args.simulate_lpf = simulate_lpf
    args.lpf_cutoff_freq = lpf_cutoff
    args.lpf_dt = lpf_dt

    out_name = "eval_action_smoothness" + default_filter_suffix(phase=args.phase, source=args.source)
    if args.disable_ref_input:
        out_name += "_noref"
    if args.actor_mode == "sample":
        out_name += "_sample"
    output_dir = (args.output_dir or (model_dir / out_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_records = filter_replay_records(
        load_replay_journal(replay_path),
        phase=args.phase,
        source=args.source,
    )
    if args.max_samples is not None:
        replay_records = replay_records[: max(int(args.max_samples), 0)]
    if not replay_records:
        raise RuntimeError(f"No replay samples left after filtering: replay={replay_path}")

    actor_path = (
        args.actor_path.resolve() if args.actor_path is not None else resolve_default_actor_snapshot_path(model_dir)
    )
    rl_config, actor_params = load_snapshot(actor_path, task_dir)
    adapter = ActionRepresentationAdapter.from_config(rl_config)
    wrapper = RLTPolicyInferenceWrapper(rl_config)

    sample_base_rng = jax.random.PRNGKey(args.actor_seed)
    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    ref_chunks: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(replay_records):
        record["_eval_index"] = index
        deterministic = args.actor_mode == "mean"
        rng = None if deterministic else jax.random.fold_in(sample_base_rng, index)
        pred = predict_refined_chunk(
            wrapper,
            adapter,
            actor_params,
            record,
            disable_ref_input=args.disable_ref_input,
            deterministic=deterministic,
            rng=rng,
        )
        gt = np.asarray(record["action_chunk"], dtype=np.float32)
        ref = np.asarray(record["ref_chunk"], dtype=np.float32)
        pred_chunks.append(pred)
        gt_chunks.append(gt)
        ref_chunks.append(ref)

        pred_stats = _chunk_intra_stats(pred)
        gt_stats = _chunk_intra_stats(gt)
        ref_stats = _chunk_intra_stats(ref)
        fit_abs = np.abs(pred - gt)
        row = {
            "index": index,
            "episode_id": int(record["episode_id"]),
            "step_id": int(record["step_id"]),
            "collection_phase": collection_phase(record),
            "source": int(record["source"]),
            "done": bool(record["done"]),
            "fit_mae": float(fit_abs.mean()),
            "fit_max": float(fit_abs.max()),
            "fit_pose_mae": float(fit_abs[:, :POSE_DIM].mean()),
            **{f"pred_{k}": v for k, v in pred_stats.items()},
            **{f"gt_{k}": v for k, v in gt_stats.items()},
            **{f"ref_{k}": v for k, v in ref_stats.items()},
        }
        rows.append(row)

    # --- Inter-chunk metrics (same episode, consecutive step_id) ---
    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records:
        by_episode[int(record["episode_id"])].append(record)

    boundary_rows: list[dict[str, Any]] = []
    pred_boundary_l2: list[float] = []
    gt_boundary_l2: list[float] = []
    pred_overlap_mae: list[float] = []
    gt_overlap_mae: list[float] = []

    for episode_id, ep_records in by_episode.items():
        ordered = sorted(ep_records, key=lambda r: int(r["step_id"]))
        for i in range(len(ordered) - 1):
            a = ordered[i]
            b = ordered[i + 1]
            ia = int(a["_eval_index"])
            ib = int(b["_eval_index"])
            pa, pb = pred_chunks[ia], pred_chunks[ib]
            ga, gb = gt_chunks[ia], gt_chunks[ib]
            gap = int(b["step_id"]) - int(a["step_id"])
            pred_jump = float(np.linalg.norm(pa[-1, :POSE_DIM] - pb[0, :POSE_DIM]))
            gt_jump = float(np.linalg.norm(ga[-1, :POSE_DIM] - gb[0, :POSE_DIM]))
            pred_boundary_l2.append(pred_jump)
            gt_boundary_l2.append(gt_jump)
            overlap_mae_pred = float("nan")
            overlap_mae_gt = float("nan")
            if 0 < gap < pa.shape[0]:
                overlap_mae_pred = float(np.mean(np.abs(pa[gap:, :POSE_DIM] - pb[: pa.shape[0] - gap, :POSE_DIM])))
                overlap_mae_gt = float(np.mean(np.abs(ga[gap:, :POSE_DIM] - gb[: ga.shape[0] - gap, :POSE_DIM])))
                pred_overlap_mae.append(overlap_mae_pred)
                gt_overlap_mae.append(overlap_mae_gt)
            boundary_rows.append(
                {
                    "episode_id": episode_id,
                    "step_id_a": int(a["step_id"]),
                    "step_id_b": int(b["step_id"]),
                    "gap": gap,
                    "pred_boundary_l2": pred_jump,
                    "gt_boundary_l2": gt_jump,
                    "pred_overlap_mae": overlap_mae_pred,
                    "gt_overlap_mae": overlap_mae_gt,
                }
            )

    pred_intra = np.asarray([r["pred_intra_vel_mean"] for r in rows], dtype=np.float32)
    gt_intra = np.asarray([r["gt_intra_vel_mean"] for r in rows], dtype=np.float32)
    ref_intra = np.asarray([r["ref_intra_vel_mean"] for r in rows], dtype=np.float32)
    pred_acc = np.asarray([r["pred_intra_acc_mean"] for r in rows], dtype=np.float32)
    gt_acc = np.asarray([r["gt_intra_acc_mean"] for r in rows], dtype=np.float32)
    fit_mae = np.asarray([r["fit_mae"] for r in rows], dtype=np.float32)
    pred_boundary = np.asarray(pred_boundary_l2, dtype=np.float32)
    gt_boundary = np.asarray(gt_boundary_l2, dtype=np.float32)
    pred_overlap = np.asarray(pred_overlap_mae, dtype=np.float32)
    gt_overlap = np.asarray(gt_overlap_mae, dtype=np.float32)

    # --- Optional deploy LPF simulation (episode-continuous, fixed dt) ---
    lpf_chunks: list[np.ndarray] | None = None
    lpf_intra = np.zeros(0, dtype=np.float32)
    lpf_acc = np.zeros(0, dtype=np.float32)
    lpf_fit_mae = np.zeros(0, dtype=np.float32)
    lpf_boundary = np.zeros(0, dtype=np.float32)
    if args.simulate_lpf:
        lpf_chunks = _filter_pred_chunks_episode_continuous(
            replay_records,
            pred_chunks,
            cutoff_freq=float(args.lpf_cutoff_freq),
            dt=float(args.lpf_dt),
        )
        for i, row in enumerate(rows):
            lpf_stats = _chunk_intra_stats(lpf_chunks[i])
            fit_lpf = np.abs(lpf_chunks[i] - gt_chunks[i])
            row["lpf_intra_vel_mean"] = lpf_stats["intra_vel_mean"]
            row["lpf_intra_acc_mean"] = lpf_stats["intra_acc_mean"]
            row["lpf_intra_acc_max"] = lpf_stats["intra_acc_max"]
            row["lpf_fit_mae"] = float(fit_lpf.mean())
        lpf_intra = np.asarray([r["lpf_intra_vel_mean"] for r in rows], dtype=np.float32)
        lpf_acc = np.asarray([r["lpf_intra_acc_mean"] for r in rows], dtype=np.float32)
        lpf_fit_mae = np.asarray([r["lpf_fit_mae"] for r in rows], dtype=np.float32)
        lpf_boundary_l2: list[float] = []
        for episode_id, ep_records in by_episode.items():
            ordered = sorted(ep_records, key=lambda r: int(r["step_id"]))
            for i in range(len(ordered) - 1):
                ia = int(ordered[i]["_eval_index"])
                ib = int(ordered[i + 1]["_eval_index"])
                pa, pb = lpf_chunks[ia], lpf_chunks[ib]
                lpf_boundary_l2.append(float(np.linalg.norm(pa[-1, :POSE_DIM] - pb[0, :POSE_DIM])))
        lpf_boundary = np.asarray(lpf_boundary_l2, dtype=np.float32)

    summary = {
        "replay_path": str(replay_path),
        "actor_path": str(actor_path),
        "model_dir": str(model_dir),
        "actor_mode": args.actor_mode,
        "num_chunks": len(rows),
        "num_inter_chunk_pairs": len(boundary_rows),
        "num_overlap_pairs": int(pred_overlap.size),
        "fit_vs_gt": _series_summary(fit_mae),
        "intra_vel_mean": {
            "pred": _series_summary(pred_intra),
            "gt": _series_summary(gt_intra),
            "ref": _series_summary(ref_intra),
        },
        "intra_acc_mean": {
            "pred": _series_summary(pred_acc),
            "gt": _series_summary(gt_acc),
        },
        "inter_boundary_l2": {
            "pred": _series_summary(pred_boundary),
            "gt": _series_summary(gt_boundary),
        },
        "inter_overlap_mae": {
            "pred": _series_summary(pred_overlap),
            "gt": _series_summary(gt_overlap),
        },
        "ratio_pred_gt_intra_vel": float(pred_intra.mean() / max(gt_intra.mean(), 1e-8)),
        "ratio_pred_gt_boundary": float(pred_boundary.mean() / max(gt_boundary.mean(), 1e-8))
        if pred_boundary.size and gt_boundary.size
        else None,
    }
    if lpf_chunks is not None:
        summary["lpf_sim"] = {
            "cutoff_freq_hz": float(args.lpf_cutoff_freq),
            "dt": float(args.lpf_dt),
            "mode": "episode_continuous_fixed_dt",
            "fit_vs_gt": _series_summary(lpf_fit_mae),
            "intra_vel_mean": _series_summary(lpf_intra),
            "intra_acc_mean": _series_summary(lpf_acc),
            "inter_boundary_l2": _series_summary(lpf_boundary) if lpf_boundary.size else None,
            "ratio_lpf_gt_intra_vel": float(lpf_intra.mean() / max(gt_intra.mean(), 1e-8)),
            "ratio_lpf_pred_intra_vel": float(lpf_intra.mean() / max(pred_intra.mean(), 1e-8)),
            "ratio_lpf_gt_boundary": float(lpf_boundary.mean() / max(gt_boundary.mean(), 1e-8))
            if lpf_boundary.size and gt_boundary.size
            else None,
        }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output_dir / "per_chunk.csv", rows)
    _write_csv(output_dir / "inter_chunk.csv", boundary_rows)

    _boxplot_compare(
        output_dir / "intra_vel_boxplot.png",
        {"pred": pred_intra, "gt": gt_intra, "ref": ref_intra},
        title="Intra-chunk step velocity (pose L2 mean)",
        ylabel="mean ||Δa|| over chunk",
    )
    _boxplot_compare(
        output_dir / "intra_acc_boxplot.png",
        {"pred": pred_acc, "gt": gt_acc},
        title="Intra-chunk acceleration (pose L2 mean)",
        ylabel="mean ||Δ²a|| over chunk",
    )
    _hist_compare(
        output_dir / "intra_vel_hist.png",
        {"pred": pred_intra, "gt": gt_intra, "ref": ref_intra},
        title="Intra-chunk velocity distribution",
        xlabel="mean ||Δa||",
    )
    _joint_vel_boxplot(
        output_dir / "intra_vel_per_joint_boxplot.png",
        pred_chunks,
        gt_chunks,
        title="Per-joint |step Δ| inside chunk",
    )
    _scatter_smooth_vs_fit(
        output_dir / "smoothness_vs_fit_scatter.png",
        pred_intra,
        gt_intra,
        fit_mae,
    )
    if pred_boundary.size:
        _plot_boundary_hist(
            output_dir / "inter_chunk_hist.png",
            pred_boundary,
            gt_boundary,
            pred_overlap if pred_overlap.size else np.zeros(1),
            gt_overlap if gt_overlap.size else np.zeros(1),
        )
        _boxplot_compare(
            output_dir / "inter_boundary_boxplot.png",
            {"pred": pred_boundary, "gt": gt_boundary},
            title="Inter-chunk boundary jump",
            ylabel="||c_i[-1] - c_{i+1}[0]|| (pose)",
        )
        if pred_overlap.size:
            _boxplot_compare(
                output_dir / "inter_overlap_boxplot.png",
                {"pred": pred_overlap, "gt": gt_overlap},
                title="Inter-chunk overlap consistency MAE",
                ylabel="MAE on aligned overlap",
            )

    if lpf_chunks is not None:
        _boxplot_compare(
            output_dir / "lpf_intra_vel_boxplot.png",
            {"pred": pred_intra, "pred+lpf": lpf_intra, "gt": gt_intra, "ref": ref_intra},
            title=f"Intra-chunk velocity: LPF sim (fc={args.lpf_cutoff_freq}Hz, dt={args.lpf_dt:.4f})",
            ylabel="mean ||Δa|| over chunk",
        )
        _boxplot_compare(
            output_dir / "lpf_intra_acc_boxplot.png",
            {"pred": pred_acc, "pred+lpf": lpf_acc, "gt": gt_acc},
            title=f"Intra-chunk acceleration: LPF sim (fc={args.lpf_cutoff_freq}Hz)",
            ylabel="mean ||Δ²a|| over chunk",
        )
        _hist_compare(
            output_dir / "lpf_intra_vel_hist.png",
            {"pred": pred_intra, "pred+lpf": lpf_intra, "gt": gt_intra},
            title="Intra-chunk velocity: pred vs pred+LPF vs gt",
            xlabel="mean ||Δa||",
        )
        if lpf_boundary.size and pred_boundary.size:
            _boxplot_compare(
                output_dir / "lpf_inter_boundary_boxplot.png",
                {"pred": pred_boundary, "pred+lpf": lpf_boundary, "gt": gt_boundary},
                title="Inter-chunk boundary jump: LPF sim",
                ylabel="||c_i[-1] - c_{i+1}[0]|| (pose)",
            )
        _scatter_smooth_vs_fit(
            output_dir / "lpf_smoothness_vs_fit_scatter.png",
            lpf_intra,
            gt_intra,
            lpf_fit_mae,
        )
        # Rename axes conceptually: scatter uses pred_intra arg as y — file name clarifies LPF.
        _joint_vel_boxplot(
            output_dir / "lpf_intra_vel_per_joint_boxplot.png",
            lpf_chunks,
            gt_chunks,
            title="Per-joint |step Δ| after LPF",
        )

    _plot_jerky_overlays(
        output_dir / f"top{min(args.topk_jerky, len(rows))}_jerky_chunks.png",
        rows,
        pred_chunks,
        gt_chunks,
        ref_chunks,
        top_k=args.topk_jerky,
        lpf_chunks=lpf_chunks,
    )

    pred_by_index = {i: pred_chunks[i] for i in range(len(pred_chunks))}
    lpf_by_index = {i: lpf_chunks[i] for i in range(len(lpf_chunks))} if lpf_chunks is not None else None
    ranked_eps = sorted(by_episode.items(), key=lambda kv: len(kv[1]), reverse=True)
    for ep_id, ep_recs in ranked_eps[: max(int(args.num_episode_plots), 0)]:
        _plot_episode_stitch(
            output_dir / f"episode_{ep_id:06d}_stitched.png",
            ep_id,
            ep_recs,
            pred_by_index,
            lpf_by_index=lpf_by_index,
        )

    print(f"wrote action-smoothness analysis to: {output_dir}")
    print(
        "intra_vel mean  pred={:.4f} gt={:.4f} ref={:.4f}  (pred/gt={:.2f}x)".format(
            summary["intra_vel_mean"]["pred"]["mean"],
            summary["intra_vel_mean"]["gt"]["mean"],
            summary["intra_vel_mean"]["ref"]["mean"],
            summary["ratio_pred_gt_intra_vel"],
        )
    )
    if summary["ratio_pred_gt_boundary"] is not None:
        print(
            "boundary mean pred={:.4f} gt={:.4f}  (pred/gt={:.2f}x)".format(
                summary["inter_boundary_l2"]["pred"]["mean"],
                summary["inter_boundary_l2"]["gt"]["mean"],
                summary["ratio_pred_gt_boundary"],
            )
        )
    print(
        "fit vs gt  mae={:.4f}  p95={:.4f}".format(
            summary["fit_vs_gt"]["mean"],
            summary["fit_vs_gt"]["p95"],
        )
    )
    if lpf_chunks is not None:
        lpf_info = summary["lpf_sim"]
        print(
            "LPF sim fc={:.2f}Hz dt={:.4f}: intra_vel pred+lpf={:.4f} (lpf/gt={:.2f}x, lpf/pred={:.2f}x) "
            "fit_mae={:.4f}".format(
                lpf_info["cutoff_freq_hz"],
                lpf_info["dt"],
                lpf_info["intra_vel_mean"]["mean"],
                lpf_info["ratio_lpf_gt_intra_vel"],
                lpf_info["ratio_lpf_pred_intra_vel"],
                lpf_info["fit_vs_gt"]["mean"],
            )
        )


if __name__ == "__main__":
    main()
