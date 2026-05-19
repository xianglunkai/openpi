from __future__ import annotations

import argparse
import csv
import json
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
import matplotlib.pyplot as plt
import numpy as np

"""
Evaluate actor fit or recorded-action fit on replay subsets.

Supported comparison targets:

- snapshot
  - run the current actor on replay states with `mean` or `sample` inference
  - compare `predicted_chunk` against `ref_chunk`
- recorded-action
  - compare replay `action_chunk` directly against `ref_chunk`

Replay filters:

- `phase`: `all / warmup / online / unknown`
- `source`: `all / base / rl / human / mixed`
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate actor-fit or recorded-action-fit on replay subsets.")
    parser.add_argument(
        "--replay-path",
        type=Path,
        required=True,
        help="Replay journal to analyze, usually runs/<task>/replay/replay_journal.pkl",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory used to auto-resolve actor artifacts and default output locations. Defaults to <task-dir>/offline_train_bcq",
    )
    parser.add_argument(
        "--actor-path",
        type=Path,
        default=None,
        help="Optional explicit actor snapshot/checkpoint path. Overrides --model-dir auto resolution.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--disable-ref-input", action="store_true", help="Only used when compare_target=snapshot.")
    parser.add_argument("--compare-target", choices=("snapshot", "recorded-action"), default="snapshot")
    parser.add_argument(
        "--actor-mode",
        choices=("mean", "sample"),
        default="mean",
        help="Only used when compare_target=snapshot. mean: actor mean without std sampling; sample: draw one actor sample using fixed_std.",
    )
    parser.add_argument(
        "--actor-seed",
        type=int,
        default=0,
        help="Random seed used only when compare_target=snapshot and actor-mode=sample.",
    )
    parser.add_argument("--phase", choices=PHASE_CHOICES, default="all")
    parser.add_argument("--source", choices=tuple(SOURCE_CHOICES), default="all")
    return parser.parse_args()


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_counts(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    phase_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for record in records:
        phase = collection_phase(record)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        source = str(int(record["source"]))
        source_counts[source] = source_counts.get(source, 0) + 1
    return {
        "collection_phase_counts": phase_counts,
        "source_counts": source_counts,
    }


def _write_summary_json(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    all_abs_delta: np.ndarray,
    compare_target: str,
    replay_path: Path,
    records: list[dict[str, Any]],
    actor_path: Path | None,
    actor_mode: str,
    actor_seed: int,
    fixed_std: float | None,
) -> None:
    target_step_delta = np.diff(np.asarray([row["target_chunk"] for row in rows], dtype=np.float32), axis=1)
    ref_step_delta = np.diff(np.asarray([row["ref_chunk"] for row in rows], dtype=np.float32), axis=1)
    target_step_abs = np.abs(target_step_delta)
    ref_step_abs = np.abs(ref_step_delta)
    per_joint = {}
    for joint_idx, label in enumerate(JOINT_LABELS):
        joint_abs = all_abs_delta[:, :, joint_idx]
        joint_target_step = target_step_abs[:, :, joint_idx]
        joint_ref_step = ref_step_abs[:, :, joint_idx]
        per_joint[label] = {
            "mean_abs_delta": float(joint_abs.mean()),
            "median_abs_delta": float(np.median(joint_abs)),
            "p95_abs_delta": float(np.percentile(joint_abs, 95)),
            "max_abs_delta": float(joint_abs.max()),
            "target_mean_step_delta": float(joint_target_step.mean()),
            "target_p95_step_delta": float(np.percentile(joint_target_step, 95)),
            "target_max_step_delta": float(joint_target_step.max()),
            "ref_mean_step_delta": float(joint_ref_step.mean()),
            "ref_p95_step_delta": float(np.percentile(joint_ref_step, 95)),
            "ref_max_step_delta": float(joint_ref_step.max()),
        }
    payload = {
        "compare_target": compare_target,
        "replay_path": str(replay_path),
        "actor_path": str(actor_path) if actor_path is not None else None,
        "actor_mode": actor_mode,
        "actor_seed": int(actor_seed),
        "fixed_std": None if fixed_std is None else float(fixed_std),
        "num_samples": len(rows),
        **_summary_counts(records),
        "overall": {
            "mean_abs_delta": float(all_abs_delta.mean()),
            "median_abs_delta": float(np.median(all_abs_delta)),
            "p95_abs_delta": float(np.percentile(all_abs_delta, 95)),
            "max_abs_delta": float(all_abs_delta.max()),
            "target_mean_step_delta": float(target_step_abs.mean()),
            "target_p95_step_delta": float(np.percentile(target_step_abs, 95)),
            "target_max_step_delta": float(target_step_abs.max()),
            "ref_mean_step_delta": float(ref_step_abs.mean()),
            "ref_p95_step_delta": float(np.percentile(ref_step_abs, 95)),
            "ref_max_step_delta": float(ref_step_abs.max()),
        },
        "per_joint": per_joint,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _plot_joint_abs_delta_boxplot(path: Path, all_abs_delta: np.ndarray, *, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    joint_series = [all_abs_delta[:, :, joint_idx].reshape(-1) for joint_idx in range(all_abs_delta.shape[-1])]
    bp = ax.boxplot(joint_series, patch_artist=True, labels=JOINT_LABELS, showfliers=False)
    for patch, color in zip(bp["boxes"], JOINT_COLORS, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    ax.set_title(title)
    ax.set_xlabel("joint")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_joint_abs_delta_hist(path: Path, all_abs_delta: np.ndarray, *, title: str, xlabel: str) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.reshape(-1)
    for joint_idx, label in enumerate(JOINT_LABELS):
        ax = axes[joint_idx]
        values = all_abs_delta[:, :, joint_idx].reshape(-1)
        ax.hist(values, bins=40, color=JOINT_COLORS[joint_idx], alpha=0.75)
        ax.set_title(label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.2)
    axes[-1].axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_step_delta_boxplot(
    path: Path,
    target_chunks: list[np.ndarray],
    replay_records: list[dict[str, Any]],
    *,
    target_title: str,
) -> None:
    target = np.asarray(target_chunks, dtype=np.float32)
    ref = np.asarray([np.asarray(record["ref_chunk"], dtype=np.float32) for record in replay_records], dtype=np.float32)
    target_step_abs = np.abs(np.diff(target, axis=1))
    ref_step_abs = np.abs(np.diff(ref, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    target_series = [target_step_abs[:, :, joint_idx].reshape(-1) for joint_idx in range(target_step_abs.shape[-1])]
    ref_series = [ref_step_abs[:, :, joint_idx].reshape(-1) for joint_idx in range(ref_step_abs.shape[-1])]

    for ax, series, title in (
        (axes[0], target_series, f"{target_title} Step-to-Step Delta"),
        (axes[1], ref_series, "Ref Step-to-Step Delta"),
    ):
        bp = ax.boxplot(series, patch_artist=True, labels=JOINT_LABELS, showfliers=False)
        for patch, color in zip(bp["boxes"], JOINT_COLORS, strict=True):
            patch.set_facecolor(color)
            patch.set_alpha(0.4)
        ax.set_title(title)
        ax.set_xlabel("joint")
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("|step delta|")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_topk_overlay(
    path: Path,
    rows: list[dict[str, Any]],
    replay_records: list[dict[str, Any]],
    target_chunks: list[np.ndarray],
    *,
    top_k: int,
    target_label: str,
) -> None:
    top_rows = sorted(rows, key=lambda row: row["max_abs_delta"], reverse=True)[: min(top_k, len(rows))]
    if not top_rows:
        return
    fig, axes = plt.subplots(len(top_rows), 2, figsize=(14, 3.8 * len(top_rows)), squeeze=False)
    for plot_idx, row in enumerate(top_rows):
        record = replay_records[row["index"]]
        ref_chunk = np.asarray(record["ref_chunk"], dtype=np.float32)
        target_chunk = target_chunks[row["index"]]
        delta = target_chunk - ref_chunk

        ax_left = axes[plot_idx, 0]
        ax_right = axes[plot_idx, 1]
        for joint_idx, label in enumerate(JOINT_LABELS):
            ax_left.plot(
                ref_chunk[:, joint_idx],
                "--",
                color=JOINT_COLORS[joint_idx],
                alpha=0.45,
                linewidth=1.7,
                label=f"{label} ref" if plot_idx == 0 else None,
            )
            ax_left.plot(
                target_chunk[:, joint_idx],
                "-",
                color=JOINT_COLORS[joint_idx],
                alpha=0.95,
                linewidth=2.0,
                label=f"{label} {target_label}" if plot_idx == 0 else None,
            )
            ax_right.plot(
                delta[:, joint_idx],
                "-",
                color=JOINT_COLORS[joint_idx],
                alpha=0.95,
                linewidth=2.0,
                label=label if plot_idx == 0 else None,
            )

        ax_left.set_title(
            f"sample {row['index']}: episode={row['episode_id']} step={row['step_id']} phase={row['collection_phase']}"
        )
        ax_left.set_xlabel("chunk step (0-9)")
        ax_left.set_ylabel("absolute joint target")
        ax_left.grid(True, alpha=0.25)

        ax_right.axhline(0.0, color="k", lw=1, alpha=0.6)
        ax_right.set_title(f"{target_label} - ref, max|delta|={row['max_abs_delta']:.3f}")
        ax_right.set_xlabel("chunk step (0-9)")
        ax_right.set_ylabel("delta from ref")
        ax_right.grid(True, alpha=0.25)

    left_handles, left_labels = axes[0, 0].get_legend_handles_labels()
    right_handles, right_labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(left_handles + right_handles, left_labels + right_labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _default_output_name(args: argparse.Namespace) -> str:
    if args.compare_target == "recorded-action":
        base = "eval_recorded_action_fit"
    else:
        base = "eval_actor_fit"
    if args.compare_target == "snapshot" and args.disable_ref_input:
        base += "_noref"
    if args.compare_target == "snapshot" and args.actor_mode == "sample":
        base += "_sample"
    return base + default_filter_suffix(phase=args.phase, source=args.source)


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.resolve()
    task_dir = infer_task_dir_from_replay_path(replay_path)
    model_dir = (args.model_dir or (task_dir / "offline_train_bcq")).resolve()
    default_output_parent = model_dir if args.compare_target == "snapshot" else task_dir
    output_dir = (args.output_dir or (default_output_parent / _default_output_name(args))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    replay_records = filter_replay_records(
        load_replay_journal(replay_path),
        phase=args.phase,
        source=args.source,
    )
    if not replay_records:
        raise RuntimeError(f"No replay samples left after filtering: replay={replay_path}")

    actor_path: Path | None = None
    adapter: ActionRepresentationAdapter | None = None
    wrapper: RLTPolicyInferenceWrapper | None = None
    actor_params = None
    fixed_std: float | None = None
    if args.compare_target == "snapshot":
        actor_path = (
            args.actor_path.resolve() if args.actor_path is not None else resolve_default_actor_snapshot_path(model_dir)
        )
        rl_config, actor_params = load_snapshot(actor_path, task_dir)
        adapter = ActionRepresentationAdapter.from_config(rl_config)
        wrapper = RLTPolicyInferenceWrapper(rl_config)
        fixed_std = float(rl_config.fixed_std)

    rows: list[dict[str, Any]] = []
    target_chunks: list[np.ndarray] = []
    abs_deltas: list[np.ndarray] = []
    sample_base_rng = jax.random.PRNGKey(args.actor_seed)
    for index, record in enumerate(replay_records):
        if args.compare_target == "snapshot":
            assert wrapper is not None
            deterministic = args.actor_mode == "mean"
            rng = None if deterministic else jax.random.fold_in(sample_base_rng, index)
            target_chunk = predict_refined_chunk(
                wrapper,
                adapter,
                actor_params,
                record,
                disable_ref_input=args.disable_ref_input,
                deterministic=deterministic,
                rng=rng,
            )
        else:
            target_chunk = np.asarray(record["action_chunk"], dtype=np.float32)
        ref_chunk = np.asarray(record["ref_chunk"], dtype=np.float32)
        delta = target_chunk - ref_chunk
        abs_delta = np.abs(delta)
        target_chunks.append(target_chunk)
        abs_deltas.append(abs_delta)
        row: dict[str, Any] = {
            "index": index,
            "episode_id": int(record["episode_id"]),
            "step_id": int(record["step_id"]),
            "collection_phase": collection_phase(record),
            "done": bool(record["done"]),
            "success": int(record["success"]),
            "intervention_flag": bool(record["intervention_flag"]),
            "source": int(record["source"]),
            "mean_abs_delta": float(abs_delta.mean()),
            "median_abs_delta": float(np.median(abs_delta)),
            "p95_abs_delta": float(np.percentile(abs_delta, 95)),
            "max_abs_delta": float(abs_delta.max()),
            "target_chunk": target_chunk,
            "ref_chunk": ref_chunk,
        }
        for joint_idx, label in enumerate(JOINT_LABELS):
            row[f"{label}_mean_abs_delta"] = float(abs_delta[:, joint_idx].mean())
            row[f"{label}_max_abs_delta"] = float(abs_delta[:, joint_idx].max())
        rows.append(row)

    all_abs_delta = np.stack(abs_deltas, axis=0).astype(np.float32, copy=False)
    csv_rows = [{k: v for k, v in row.items() if k not in {"target_chunk", "ref_chunk"}} for row in rows]
    _write_summary_csv(output_dir / "summary.csv", csv_rows)
    _write_summary_json(
        output_dir / "summary.json",
        rows=rows,
        all_abs_delta=all_abs_delta,
        compare_target=args.compare_target,
        replay_path=replay_path,
        records=replay_records,
        actor_path=actor_path,
        actor_mode=args.actor_mode if args.compare_target == "snapshot" else "recorded-action",
        actor_seed=args.actor_seed,
        fixed_std=fixed_std,
    )

    if args.compare_target == "snapshot":
        target_title = "Actor Mean" if args.actor_mode == "mean" else "Actor Sample"
        target_label = "actor_mean" if args.actor_mode == "mean" else "actor_sample"
    else:
        target_title = "Recorded Action"
        target_label = "recorded_action"
    _plot_joint_abs_delta_boxplot(
        output_dir / "joint_abs_delta_boxplot.png",
        all_abs_delta,
        title=f"{target_title} vs Ref Absolute Delta by Joint",
        ylabel=f"|{target_label} - ref|",
    )
    _plot_joint_abs_delta_hist(
        output_dir / "joint_abs_delta_hist.png",
        all_abs_delta,
        title=f"{target_title} vs Ref Absolute Delta Histogram",
        xlabel=f"|{target_label} - ref|",
    )
    _plot_step_delta_boxplot(
        output_dir / "target_vs_ref_step_delta_boxplot.png",
        target_chunks,
        replay_records,
        target_title=target_title,
    )
    _plot_topk_overlay(
        output_dir / f"top{min(args.top_k, len(rows))}_compare_vs_ref.png",
        rows,
        replay_records,
        target_chunks,
        top_k=args.top_k,
        target_label=target_label,
    )
    print(f"wrote actor-fit analysis to: {output_dir}")


if __name__ == "__main__":
    main()
