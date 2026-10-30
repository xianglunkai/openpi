"""Open-loop VLA vs RL comparison on a replay journal.

The RL actor is rolled out on each recorded ``(z_rl, proprio, ref_chunk)``.
Outputs stay in absolute joint space (same units as ``ref_chunk``).

Within each chunk, at control period ``dt = 1 / control_frequency_hz``:

- action:     q
- velocity:   (q_{t+1} - q_t) / dt
- acceleration: (v_{t+1} - v_t) / dt

VLA is the journal ``ref_chunk``. This is not a closed-loop robot rollout.
"""

from __future__ import annotations

import argparse
import csv
import json
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
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np

VLA_COLOR = "#54a24b"
RL_COLOR = "#4c78a8"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot VLA ref vs RL actor action, velocity, and acceleration.")
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--actor-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--actor-mode", choices=("mean", "sample"), default="mean")
    parser.add_argument("--actor-seed", type=int, default=0)
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--source", choices=tuple(SOURCE_CHOICES), default="all")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--episode-ids", type=int, nargs="*", default=None)
    parser.add_argument("--num-episode-plots", type=int, default=3)
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Override dt=1/control_hz. Default: rl_config.control_frequency_hz from the actor snapshot.",
    )
    return parser.parse_args()


def _series_summary(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "p95": float(np.percentile(flat, 95)),
        "max": float(flat.max()),
    }


def _per_joint_summary(values: np.ndarray, labels: list[str]) -> dict[str, dict[str, float]]:
    """``values`` is (N, T, A) of absolute quantities."""
    out: dict[str, dict[str, float]] = {}
    for joint_idx, label in enumerate(labels):
        out[label] = _series_summary(values[:, :, joint_idx])
    return out


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-8))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _paired_boxplot(
    path: Path,
    vla: np.ndarray,
    rl: np.ndarray,
    labels: list[str],
    *,
    title: str,
    ylabel: str,
) -> None:
    data: list[np.ndarray] = []
    positions: list[float] = []
    for joint_idx in range(len(labels)):
        data.append(np.abs(vla[:, :, joint_idx]).reshape(-1))
        data.append(np.abs(rl[:, :, joint_idx]).reshape(-1))
        positions.extend((joint_idx * 3.0, joint_idx * 3.0 + 1.0))
    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, positions=positions, widths=0.8, showfliers=False, patch_artist=True, manage_ticks=False)
    for idx, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(VLA_COLOR if idx % 2 == 0 else RL_COLOR)
        patch.set_alpha(0.55)
    ax.set_xticks([joint_idx * 3.0 + 0.5 for joint_idx in range(len(labels))])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(
        [Patch(facecolor=VLA_COLOR, alpha=0.55), Patch(facecolor=RL_COLOR, alpha=0.55)],
        ["VLA", "RL"],
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _hist_compare(path: Path, vla: np.ndarray, rl: np.ndarray, *, title: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(np.abs(vla).reshape(-1), bins=60, density=True, alpha=0.5, color=VLA_COLOR, label="VLA")
    ax.hist(np.abs(rl).reshape(-1), bins=60, density=True, alpha=0.5, color=RL_COLOR, label="RL")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _dataset_bars(
    path: Path,
    labels: list[str],
    action_mae: np.ndarray,
    vla_vel: np.ndarray,
    rl_vel: np.ndarray,
    vla_acc: np.ndarray,
    rl_acc: np.ndarray,
) -> None:
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    axes[0].bar(x, action_mae, color="#f58518", alpha=0.85)
    axes[0].set_title("Action |RL − VLA|")
    axes[0].set_ylabel("mean abs error (rad or m)")

    axes[1].bar(x - width / 2, vla_vel, width=width, color=VLA_COLOR, alpha=0.85, label="VLA")
    axes[1].bar(x + width / 2, rl_vel, width=width, color=RL_COLOR, alpha=0.85, label="RL")
    axes[1].set_title("Mean |velocity|")
    axes[1].set_ylabel("rad/s or m/s")
    axes[1].legend()

    axes[2].bar(x - width / 2, vla_acc, width=width, color=VLA_COLOR, alpha=0.85, label="VLA")
    axes[2].bar(x + width / 2, rl_acc, width=width, color=RL_COLOR, alpha=0.85, label="RL")
    axes[2].set_title("Mean |acceleration|")
    axes[2].set_ylabel("rad/s² or m/s²")
    axes[2].legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle("Dataset open-loop comparison")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _horizon_profile(
    path: Path,
    labels: list[str],
    action_error: np.ndarray,
    vla_speed: np.ndarray,
    rl_speed: np.ndarray,
) -> None:
    """Mean signed action error and mean |velocity| along the chunk horizon."""
    n_joints = len(labels)
    fig, axes = plt.subplots(n_joints, 2, figsize=(12, 1.8 * n_joints), sharex="col", squeeze=False)
    horizon = np.arange(action_error.shape[0])
    speed_horizon = np.arange(vla_speed.shape[0])
    for joint_idx, label in enumerate(labels):
        ax_action = axes[joint_idx, 0]
        ax_speed = axes[joint_idx, 1]
        ax_action.plot(horizon, action_error[:, joint_idx], color=JOINT_COLORS[joint_idx], lw=1.8)
        ax_action.axhline(0.0, color="k", lw=0.8, alpha=0.5)
        ax_action.set_ylabel(label)
        ax_action.grid(True, alpha=0.2)
        ax_speed.plot(speed_horizon, vla_speed[:, joint_idx], color=VLA_COLOR, lw=1.6, label="VLA")
        ax_speed.plot(speed_horizon, rl_speed[:, joint_idx], color=RL_COLOR, lw=1.6, label="RL")
        ax_speed.grid(True, alpha=0.2)
        if joint_idx == 0:
            ax_action.set_title("Mean RL − VLA")
            ax_speed.set_title("Mean |velocity|")
            ax_speed.legend(loc="upper right")
    axes[-1, 0].set_xlabel("chunk step")
    axes[-1, 1].set_xlabel("chunk step")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _stitch_episode(
    records: list[dict[str, Any]],
    rl_chunks: np.ndarray,
    vla_chunks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep the first ``step_id`` gap of each chunk so overlapping horizons are not double-counted."""
    ordered = sorted(records, key=lambda record: int(record["step_id"]))
    times: list[float] = []
    vla_seq: list[np.ndarray] = []
    rl_seq: list[np.ndarray] = []
    for idx, record in enumerate(ordered):
        sample_index = int(record["_eval_index"])
        rl = rl_chunks[sample_index]
        vla = vla_chunks[sample_index]
        step0 = int(record["step_id"])
        if idx + 1 < len(ordered):
            gap = int(ordered[idx + 1]["step_id"]) - step0
            gap = min(max(gap, 1), rl.shape[0])
        else:
            gap = rl.shape[0]
        for t in range(gap):
            times.append(float(step0 + t))
            vla_seq.append(vla[t])
            rl_seq.append(rl[t])
    return (
        np.asarray(times, dtype=np.float32),
        np.stack(vla_seq, axis=0),
        np.stack(rl_seq, axis=0),
    )


def _plot_episode(
    path: Path,
    episode_id: int,
    times: np.ndarray,
    vla: np.ndarray,
    rl: np.ndarray,
    *,
    dt: float,
    labels: list[str],
) -> None:
    vla_vel = np.diff(vla, axis=0) / dt
    rl_vel = np.diff(rl, axis=0) / dt
    vla_acc = np.diff(vla_vel, axis=0) / dt if vla_vel.shape[0] >= 2 else np.zeros((0, vla.shape[1]))
    rl_acc = np.diff(rl_vel, axis=0) / dt if rl_vel.shape[0] >= 2 else np.zeros((0, rl.shape[1]))
    vel_times = times[:-1] if times.size else times
    acc_times = times[:-2] if times.size >= 2 else times[:0]

    fig, axes = plt.subplots(len(labels), 3, figsize=(16, 1.7 * len(labels)), sharex="col", squeeze=False)
    for joint_idx, label in enumerate(labels):
        ax_q, ax_v, ax_a = axes[joint_idx]
        ax_q.plot(times, vla[:, joint_idx], color=VLA_COLOR, lw=1.4, label="VLA")
        ax_q.plot(times, rl[:, joint_idx], color=RL_COLOR, lw=1.4, label="RL")
        ax_q.set_ylabel(label)
        ax_q.grid(True, alpha=0.2)
        if vel_times.size:
            ax_v.plot(vel_times, vla_vel[:, joint_idx], color=VLA_COLOR, lw=1.3)
            ax_v.plot(vel_times, rl_vel[:, joint_idx], color=RL_COLOR, lw=1.3)
        ax_v.grid(True, alpha=0.2)
        if acc_times.size:
            ax_a.plot(acc_times, vla_acc[:, joint_idx], color=VLA_COLOR, lw=1.3)
            ax_a.plot(acc_times, rl_acc[:, joint_idx], color=RL_COLOR, lw=1.3)
        ax_a.grid(True, alpha=0.2)
        if joint_idx == 0:
            ax_q.set_title("Action")
            ax_v.set_title("Velocity")
            ax_a.set_title("Acceleration")
            ax_q.legend(loc="upper right")
    axes[-1, 0].set_xlabel("step_id")
    axes[-1, 1].set_xlabel("step_id")
    axes[-1, 2].set_xlabel("step_id")
    fig.suptitle(f"Episode {episode_id}: VLA vs RL")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.resolve()
    task_dir = infer_task_dir_from_replay_path(replay_path)
    model_dir = (args.model_dir or (task_dir / "offline_train_bcq")).resolve()
    out_name = "eval_vla_vs_rl" + default_filter_suffix(phase=args.phase, source=args.source)
    if args.actor_mode == "sample":
        out_name += "_sample"
    output_dir = (args.output_dir or (model_dir / out_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_records = filter_replay_records(load_replay_journal(replay_path), phase=args.phase, source=args.source)
    if args.max_samples is not None:
        replay_records = replay_records[: max(int(args.max_samples), 0)]
    if not replay_records:
        raise RuntimeError(f"No replay samples left after filtering: replay={replay_path}")

    actor_path = args.actor_path.resolve() if args.actor_path is not None else resolve_default_actor_snapshot_path(model_dir)
    rl_config, actor_params = load_snapshot(actor_path, task_dir)
    adapter = ActionRepresentationAdapter.from_config(rl_config)
    wrapper = RLTPolicyInferenceWrapper(rl_config)
    control_hz = float(args.control_hz if args.control_hz is not None else rl_config.control_frequency_hz)
    if control_hz <= 0.0:
        raise ValueError(f"control_hz must be positive, got {control_hz}")
    dt = 1.0 / control_hz

    sample_base_rng = jax.random.PRNGKey(args.actor_seed)
    rl_chunks: list[np.ndarray] = []
    vla_chunks: list[np.ndarray] = []
    for index, record in enumerate(replay_records):
        record["_eval_index"] = index
        deterministic = args.actor_mode == "mean"
        rng = None if deterministic else jax.random.fold_in(sample_base_rng, index)
        rl_chunks.append(
            predict_refined_chunk(
                wrapper,
                adapter,
                actor_params,
                record,
                deterministic=deterministic,
                rng=rng,
            )
        )
        vla_chunks.append(np.asarray(record["ref_chunk"], dtype=np.float32))
        if index == 0 or (index + 1) % 500 == 0 or index + 1 == len(replay_records):
            print(f"inferred {index + 1}/{len(replay_records)}", flush=True)

    rl = np.stack(rl_chunks, axis=0).astype(np.float32, copy=False)
    vla = np.stack(vla_chunks, axis=0).astype(np.float32, copy=False)
    n_joints = int(rl.shape[-1])
    labels = [JOINT_LABELS[i] if i < len(JOINT_LABELS) else f"dim{i}" for i in range(n_joints)]
    action_error = rl - vla
    rl_vel = np.diff(rl, axis=1) / dt
    vla_vel = np.diff(vla, axis=1) / dt
    rl_acc = np.diff(rl_vel, axis=1) / dt
    vla_acc = np.diff(vla_vel, axis=1) / dt

    action_mae_joint = np.mean(np.abs(action_error), axis=(0, 1))
    rl_vel_mean_joint = np.mean(np.abs(rl_vel), axis=(0, 1))
    vla_vel_mean_joint = np.mean(np.abs(vla_vel), axis=(0, 1))
    rl_acc_mean_joint = np.mean(np.abs(rl_acc), axis=(0, 1))
    vla_acc_mean_joint = np.mean(np.abs(vla_acc), axis=(0, 1))

    summary = {
        "replay_path": str(replay_path),
        "actor_path": str(actor_path),
        "model_dir": str(model_dir),
        "actor_mode": args.actor_mode,
        "num_chunks": int(rl.shape[0]),
        "chunk_len": int(rl.shape[1]),
        "control_frequency_hz": control_hz,
        "dt_sec": dt,
        "space": "absolute_joint",
        "comparison": "open_loop_intra_chunk",
        "action_mae": {
            "overall": float(np.mean(np.abs(action_error))),
            "per_joint": {label: float(action_mae_joint[i]) for i, label in enumerate(labels)},
        },
        "velocity_abs": {
            "rl": _series_summary(np.abs(rl_vel)),
            "vla": _series_summary(np.abs(vla_vel)),
            "ratio_rl_over_vla": _ratio(float(np.mean(np.abs(rl_vel))), float(np.mean(np.abs(vla_vel)))),
            "per_joint_rl": _per_joint_summary(np.abs(rl_vel), labels),
            "per_joint_vla": _per_joint_summary(np.abs(vla_vel), labels),
        },
        "acceleration_abs": {
            "rl": _series_summary(np.abs(rl_acc)),
            "vla": _series_summary(np.abs(vla_acc)),
            "ratio_rl_over_vla": _ratio(float(np.mean(np.abs(rl_acc))), float(np.mean(np.abs(vla_acc)))),
            "per_joint_rl": _per_joint_summary(np.abs(rl_acc), labels),
            "per_joint_vla": _per_joint_summary(np.abs(vla_acc), labels),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for joint_idx, label in enumerate(labels):
        rows.append(
            {
                "joint": label,
                "action_mae": float(action_mae_joint[joint_idx]),
                "vla_vel_mean": float(vla_vel_mean_joint[joint_idx]),
                "rl_vel_mean": float(rl_vel_mean_joint[joint_idx]),
                "vel_ratio_rl_over_vla": _ratio(float(rl_vel_mean_joint[joint_idx]), float(vla_vel_mean_joint[joint_idx])),
                "vla_vel_p95": summary["velocity_abs"]["per_joint_vla"][label]["p95"],
                "rl_vel_p95": summary["velocity_abs"]["per_joint_rl"][label]["p95"],
                "vla_acc_mean": float(vla_acc_mean_joint[joint_idx]),
                "rl_acc_mean": float(rl_acc_mean_joint[joint_idx]),
                "acc_ratio_rl_over_vla": _ratio(float(rl_acc_mean_joint[joint_idx]), float(vla_acc_mean_joint[joint_idx])),
                "vla_acc_p95": summary["acceleration_abs"]["per_joint_vla"][label]["p95"],
                "rl_acc_p95": summary["acceleration_abs"]["per_joint_rl"][label]["p95"],
            }
        )
    _write_csv(output_dir / "per_joint.csv", rows)

    _dataset_bars(
        output_dir / "dataset_compare_bars.png",
        labels,
        action_mae_joint,
        vla_vel_mean_joint,
        rl_vel_mean_joint,
        vla_acc_mean_joint,
        rl_acc_mean_joint,
    )
    _paired_boxplot(
        output_dir / "velocity_boxplot.png",
        vla_vel,
        rl_vel,
        labels,
        title=f"Intra-chunk |velocity| at {control_hz:.0f} Hz",
        ylabel="|dq/dt|",
    )
    _paired_boxplot(
        output_dir / "acceleration_boxplot.png",
        vla_acc,
        rl_acc,
        labels,
        title=f"Intra-chunk |acceleration| at {control_hz:.0f} Hz",
        ylabel="|d²q/dt²|",
    )
    _hist_compare(
        output_dir / "velocity_hist.png",
        vla_vel,
        rl_vel,
        title="Velocity distribution",
        xlabel="|dq/dt|",
    )
    _hist_compare(
        output_dir / "acceleration_hist.png",
        vla_acc,
        rl_acc,
        title="Acceleration distribution",
        xlabel="|d²q/dt²|",
    )
    _horizon_profile(
        output_dir / "chunk_horizon_profile.png",
        labels,
        np.mean(action_error, axis=0),
        np.mean(np.abs(vla_vel), axis=0),
        np.mean(np.abs(rl_vel), axis=0),
    )

    by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in replay_records:
        by_episode[int(record["episode_id"])].append(record)
    if args.episode_ids:
        episode_ids = [int(episode_id) for episode_id in args.episode_ids]
        missing = [episode_id for episode_id in episode_ids if episode_id not in by_episode]
        if missing:
            raise RuntimeError(f"Episode ids not in the filtered replay: {missing}")
    else:
        ranked = sorted(by_episode.items(), key=lambda item: len(item[1]), reverse=True)
        episode_ids = [episode_id for episode_id, _ in ranked[: max(int(args.num_episode_plots), 0)]]

    for episode_id in episode_ids:
        times, vla_seq, rl_seq = _stitch_episode(by_episode[episode_id], rl, vla)
        _plot_episode(
            output_dir / f"episode_{episode_id:06d}_vla_vs_rl.png",
            episode_id,
            times,
            vla_seq,
            rl_seq,
            dt=dt,
            labels=labels,
        )

    print(f"wrote VLA vs RL comparison to: {output_dir}")
    print(
        "action mae={:.5f}  |vel| RL={:.5f} VLA={:.5f} ({:.2f}x)  |acc| RL={:.3f} VLA={:.3f} ({:.2f}x)".format(
            summary["action_mae"]["overall"],
            summary["velocity_abs"]["rl"]["mean"],
            summary["velocity_abs"]["vla"]["mean"],
            summary["velocity_abs"]["ratio_rl_over_vla"],
            summary["acceleration_abs"]["rl"]["mean"],
            summary["acceleration_abs"]["vla"]["mean"],
            summary["acceleration_abs"]["ratio_rl_over_vla"],
        )
    )


if __name__ == "__main__":
    main()
