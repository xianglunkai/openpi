from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
OFFLINE_DIR = CURRENT_DIR.parent / "offline"
ROOT = CURRENT_DIR.parents[1]
if str(OFFLINE_DIR) not in sys.path:
    sys.path.insert(0, str(OFFLINE_DIR))

if TYPE_CHECKING:
    from _common import ActionRepresentationAdapter
    from _common import RLTPolicyInferenceWrapper


JOINT_LABELS = [f"joint{i + 1}" for i in range(6)] + ["gripper"]
SOURCE_LABELS = {
    0: "BASE",
    1: "RL",
    2: "HUMAN",
    3: "MIXED",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one complete replay episode as chunk-by-chunk joint playback data for "
            "real-robot comparison between replay ref_chunk and actor deterministic output."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory that owns replay/ and actor_snapshot/.")
    parser.add_argument(
        "--replay-path",
        type=Path,
        default=None,
        help="Optional replay journal path. Defaults to run_dir/replay/replay_journal_no_rl.pkl then replay_journal.pkl.",
    )
    parser.add_argument("--episode-id", type=int, default=None, help="Episode id to export. Use --list-episodes to inspect candidates.")
    parser.add_argument("--list-episodes", action="store_true", help="Only print episode summary and exit.")
    parser.add_argument(
        "--offline-dir",
        type=Path,
        default=None,
        help="Optional offline experiment directory; if set, actor snapshot is resolved from that experiment directory.",
    )
    parser.add_argument("--snapshot-path", type=Path, default=None, help="Optional explicit actor snapshot path.")
    parser.add_argument("--disable-ref-input", action="store_true", help="Zero actor ref input when producing actor chunks.")
    parser.add_argument(
        "--keep-terminal-success",
        action="store_true",
        help="Keep the final success/done chunk. By default it is dropped for playback experiments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <actor-owner-dir>/replay_real_robot_exports/episode_<id>.",
    )
    return parser.parse_args()


def _resolve_replay_path(run_dir: Path, replay_path: Path | None) -> Path:
    if replay_path is not None:
        return replay_path
    preferred = run_dir / "replay" / "replay_journal_no_rl.pkl"
    if preferred.exists():
        return preferred
    fallback = run_dir / "replay" / "replay_journal.pkl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find replay journal under {run_dir / 'replay'}")


def _resolve_snapshot_path(args: argparse.Namespace) -> Path:
    if args.snapshot_path is not None:
        return args.snapshot_path
    common = _load_common_module()
    if args.offline_dir is not None:
        return common.resolve_default_actor_snapshot_path(args.offline_dir)
    return common.resolve_default_actor_snapshot_path(args.run_dir)


def _resolve_actor_owner_dir(args: argparse.Namespace, snapshot_path: Path) -> Path:
    if args.offline_dir is not None:
        return args.offline_dir.resolve()
    if args.snapshot_path is None:
        return args.run_dir.resolve()

    resolved = snapshot_path.resolve()
    if resolved.parent.name in {"actor_snapshot", "checkpoints"}:
        return resolved.parent.parent
    return resolved.parent


def _relativize_path(path_value: Path, anchor_path: Path) -> str:
    resolved = path_value.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError:
        return os.path.relpath(str(resolved), start=str(anchor_path.resolve().parent))
    return os.path.relpath(str(resolved), start=str(ROOT))


def _load_common_module():
    import _common

    return _common


def _group_records_by_episode(records: list[dict]) -> dict[int, list[dict]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        grouped[int(record["episode_id"])].append(record)
    for episode_id, episode_records in grouped.items():
        episode_records.sort(key=lambda item: int(item["step_id"]))
    return dict(grouped)


def _summarize_episodes(grouped: dict[int, list[dict]]) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for episode_id in sorted(grouped):
        records = grouped[episode_id]
        rows.append(
            {
                "episode_id": episode_id,
                "num_chunks": len(records),
                "done_count": int(sum(bool(np.asarray(record["done"]).item()) for record in records)),
                "success_count": int(sum(int(np.asarray(record["success"]).item()) for record in records)),
                "step_id_first": int(np.asarray(records[0]["step_id"]).item()),
                "step_id_last": int(np.asarray(records[-1]["step_id"]).item()),
            }
        )
    return rows


def _drop_terminal_success_chunk(records: list[dict]) -> list[dict]:
    if not records:
        return records
    last = records[-1]
    done = bool(np.asarray(last["done"]).item())
    success = bool(np.asarray(last["success"]).item())
    if done and success:
        return records[:-1]
    return records


def _build_wrapper_and_adapter(snapshot_path: Path, run_dir: Path) -> tuple["RLTPolicyInferenceWrapper", "ActionRepresentationAdapter | None", object]:
    common = _load_common_module()
    cfg, actor_params = common.load_snapshot(snapshot_path, run_dir)
    wrapper = common.RLTPolicyInferenceWrapper(cfg)
    adapter = common.ActionRepresentationAdapter.from_config(cfg)
    return wrapper, adapter, actor_params

def _step_delta(chunk: np.ndarray) -> np.ndarray:
    return chunk[:, 1:, :] - chunk[:, :-1, :]


def _overall_table_summary(payload: dict[str, np.ndarray]) -> dict[str, object]:
    ref = payload["ref_chunks"].astype(np.float64)
    actor = payload["actor_chunks"].astype(np.float64)
    diff = actor - ref
    abs_diff = np.abs(diff)
    ref_step_abs = np.abs(_step_delta(ref))
    actor_step_abs = np.abs(_step_delta(actor))
    per_joint: dict[str, dict[str, float]] = {}
    for joint_idx, joint_name in enumerate(JOINT_LABELS):
        joint_diff = abs_diff[:, :, joint_idx]
        joint_actor_step = actor_step_abs[:, :, joint_idx]
        joint_ref_step = ref_step_abs[:, :, joint_idx]
        per_joint[joint_name] = {
            "mean_abs_delta": float(joint_diff.mean()),
            "p95_abs_delta": float(np.percentile(joint_diff, 95)),
            "max_abs_delta": float(joint_diff.max()),
            "actor_mean_step_delta": float(joint_actor_step.mean()),
            "actor_p95_step_delta": float(np.percentile(joint_actor_step, 95)),
            "actor_max_step_delta": float(joint_actor_step.max()),
            "ref_mean_step_delta": float(joint_ref_step.mean()),
            "ref_p95_step_delta": float(np.percentile(joint_ref_step, 95)),
            "ref_max_step_delta": float(joint_ref_step.max()),
        }
    return {
        "num_chunks": int(ref.shape[0]),
        "chunk_len": int(ref.shape[1]),
        "action_dim": int(ref.shape[2]),
        "mean_abs_delta": float(abs_diff.mean()),
        "p95_abs_delta": float(np.percentile(abs_diff, 95)),
        "max_abs_delta": float(abs_diff.max()),
        "actor_mean_step_delta": float(actor_step_abs.mean()),
        "actor_p95_step_delta": float(np.percentile(actor_step_abs, 95)),
        "actor_max_step_delta": float(actor_step_abs.max()),
        "ref_mean_step_delta": float(ref_step_abs.mean()),
        "ref_p95_step_delta": float(np.percentile(ref_step_abs, 95)),
        "ref_max_step_delta": float(ref_step_abs.max()),
        "per_joint": per_joint,
    }


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_playback_tables(output_dir: Path, payload: dict[str, np.ndarray]) -> None:
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary = _overall_table_summary(payload)

    ref = payload["ref_chunks"].astype(np.float64)
    actor = payload["actor_chunks"].astype(np.float64)
    proprios = payload["proprios"].astype(np.float64)
    rewards = payload["rewards"].astype(np.float64)
    step_ids = payload["step_ids"].astype(np.int32)
    sources = payload["sources"].astype(np.int32)
    collection_phases = np.asarray(payload["collection_phases"], dtype=str)

    diff = actor - ref
    abs_diff = np.abs(diff)
    ref_step = _step_delta(ref)
    actor_step = _step_delta(actor)
    ref_step_abs = np.abs(ref_step)
    actor_step_abs = np.abs(actor_step)

    chunk_fieldnames = [
        "chunk_index",
        "step_id",
        "collection_phase",
        "source",
        "source_name",
        "reward_sum",
        "mean_abs_delta",
        "max_abs_delta",
        "actor_mean_step_delta",
        "actor_max_step_delta",
        "ref_mean_step_delta",
        "ref_max_step_delta",
    ]
    for joint_name in JOINT_LABELS:
        chunk_fieldnames.extend(
            [
                f"{joint_name}_mean_abs_delta",
                f"{joint_name}_max_abs_delta",
                f"{joint_name}_actor_mean_step_delta",
                f"{joint_name}_ref_mean_step_delta",
            ]
        )
    chunk_rows: list[dict[str, object]] = []
    for chunk_idx in range(ref.shape[0]):
        row: dict[str, object] = {
            "chunk_index": chunk_idx,
            "step_id": int(step_ids[chunk_idx]),
            "collection_phase": str(collection_phases[chunk_idx]),
            "source": int(sources[chunk_idx]),
            "source_name": SOURCE_LABELS.get(int(sources[chunk_idx]), f"UNKNOWN_{int(sources[chunk_idx])}"),
            "reward_sum": float(rewards[chunk_idx].sum()),
            "mean_abs_delta": float(abs_diff[chunk_idx].mean()),
            "max_abs_delta": float(abs_diff[chunk_idx].max()),
            "actor_mean_step_delta": float(actor_step_abs[chunk_idx].mean()),
            "actor_max_step_delta": float(actor_step_abs[chunk_idx].max()),
            "ref_mean_step_delta": float(ref_step_abs[chunk_idx].mean()),
            "ref_max_step_delta": float(ref_step_abs[chunk_idx].max()),
        }
        for joint_idx, joint_name in enumerate(JOINT_LABELS):
            row[f"{joint_name}_mean_abs_delta"] = float(abs_diff[chunk_idx, :, joint_idx].mean())
            row[f"{joint_name}_max_abs_delta"] = float(abs_diff[chunk_idx, :, joint_idx].max())
            row[f"{joint_name}_actor_mean_step_delta"] = float(actor_step_abs[chunk_idx, :, joint_idx].mean())
            row[f"{joint_name}_ref_mean_step_delta"] = float(ref_step_abs[chunk_idx, :, joint_idx].mean())
        chunk_rows.append(row)
    _write_csv(tables_dir / "chunk_summary.csv", chunk_fieldnames, chunk_rows)

    step_fieldnames = [
        "chunk_index",
        "step_id",
        "collection_phase",
        "source",
        "source_name",
        "within_chunk_step",
        "reward",
    ]
    for prefix in ("proprio", "ref", "actor", "diff"):
        for joint_name in JOINT_LABELS:
            step_fieldnames.append(f"{prefix}_{joint_name}")
    for prefix in ("ref_step", "actor_step", "step_diff"):
        for joint_name in JOINT_LABELS:
            step_fieldnames.append(f"{prefix}_{joint_name}")
    step_rows: list[dict[str, object]] = []
    for chunk_idx in range(ref.shape[0]):
        for step_idx in range(ref.shape[1]):
            row = {
                "chunk_index": chunk_idx,
                "step_id": int(step_ids[chunk_idx]),
                "collection_phase": str(collection_phases[chunk_idx]),
                "source": int(sources[chunk_idx]),
                "source_name": SOURCE_LABELS.get(int(sources[chunk_idx]), f"UNKNOWN_{int(sources[chunk_idx])}"),
                "within_chunk_step": step_idx,
                "reward": float(rewards[chunk_idx, step_idx]),
            }
            for joint_idx, joint_name in enumerate(JOINT_LABELS):
                row[f"proprio_{joint_name}"] = float(proprios[chunk_idx, joint_idx])
                row[f"ref_{joint_name}"] = float(ref[chunk_idx, step_idx, joint_idx])
                row[f"actor_{joint_name}"] = float(actor[chunk_idx, step_idx, joint_idx])
                row[f"diff_{joint_name}"] = float(diff[chunk_idx, step_idx, joint_idx])
                if step_idx == 0:
                    row[f"ref_step_{joint_name}"] = ""
                    row[f"actor_step_{joint_name}"] = ""
                    row[f"step_diff_{joint_name}"] = ""
                else:
                    ref_delta = float(ref_step[chunk_idx, step_idx - 1, joint_idx])
                    actor_delta = float(actor_step[chunk_idx, step_idx - 1, joint_idx])
                    row[f"ref_step_{joint_name}"] = ref_delta
                    row[f"actor_step_{joint_name}"] = actor_delta
                    row[f"step_diff_{joint_name}"] = actor_delta - ref_delta
            step_rows.append(row)
    _write_csv(tables_dir / "step_detail.csv", step_fieldnames, step_rows)

    top_indices = np.argsort(abs_diff.mean(axis=(1, 2)))[-5:][::-1]
    preview_lines = [
        "# Playback Preview",
        "",
        f"- num_chunks: {summary['num_chunks']}",
        f"- chunk_len: {summary['chunk_len']}",
        f"- action_dim: {summary['action_dim']}",
        f"- mean_abs_delta: {summary['mean_abs_delta']:.6f}",
        f"- p95_abs_delta: {summary['p95_abs_delta']:.6f}",
        f"- max_abs_delta: {summary['max_abs_delta']:.6f}",
        "",
        "## Top 5 Chunks By Mean |actor-ref|",
        "",
        "| chunk_index | mean_abs_delta | max_abs_delta | collection_phase | source |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for idx in top_indices:
        preview_lines.append(
            "| "
            + " | ".join(
                [
                    str(int(idx)),
                    f"{float(abs_diff[idx].mean()):.6f}",
                    f"{float(abs_diff[idx].max()):.6f}",
                    str(collection_phases[idx]),
                    SOURCE_LABELS.get(int(sources[idx]), f"UNKNOWN_{int(sources[idx])}"),
                ]
            )
            + " |"
        )
    (tables_dir / "preview.md").write_text("\n".join(preview_lines) + "\n", encoding="utf-8")
    (tables_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    replay_path = _resolve_replay_path(args.run_dir, args.replay_path)
    common = _load_common_module()
    records = common.load_replay_journal(replay_path)
    grouped = _group_records_by_episode(records)
    summary_rows = _summarize_episodes(grouped)

    if args.list_episodes:
        print(f"Replay: {replay_path}")
        print("episode_id  num_chunks  done_count  success_count  step_id_first  step_id_last")
        for row in summary_rows:
            print(
                f"{row['episode_id']:>10}  {row['num_chunks']:>10}  {row['done_count']:>10}  "
                f"{row['success_count']:>13}  {row['step_id_first']:>13}  {row['step_id_last']:>12}"
            )
        return

    if args.episode_id is None:
        raise SystemExit("Please provide --episode-id, or use --list-episodes first.")
    if args.episode_id not in grouped:
        raise SystemExit(f"Episode {args.episode_id} not found in {replay_path}")

    episode_records = list(grouped[args.episode_id])
    original_num_chunks = len(episode_records)
    if not args.keep_terminal_success:
        episode_records = _drop_terminal_success_chunk(episode_records)
    if not episode_records:
        raise SystemExit(f"Episode {args.episode_id} became empty after terminal-success filtering.")

    snapshot_path = _resolve_snapshot_path(args)
    actor_owner_dir = _resolve_actor_owner_dir(args, snapshot_path)
    wrapper, adapter, actor_params = _build_wrapper_and_adapter(snapshot_path, args.run_dir)

    ref_chunks = []
    actor_chunks = []
    proprios = []
    collection_phases = []
    step_ids = []
    sources = []
    rewards = []
    for record in episode_records:
        ref_chunk = np.asarray(record["ref_chunk"], dtype=np.float32)
        actor_chunk = common.predict_refined_chunk(
            wrapper,
            adapter,
            actor_params,
            record,
            disable_ref_input=args.disable_ref_input,
        )
        ref_chunks.append(ref_chunk)
        actor_chunks.append(actor_chunk)
        proprios.append(np.asarray(record["proprio"], dtype=np.float32))
        collection_phases.append(str(record.get("collection_phase", "unknown")))
        step_ids.append(int(np.asarray(record["step_id"]).item()))
        sources.append(int(np.asarray(record["source"]).item()))
        rewards.append(np.asarray(record["rewards"], dtype=np.float32))

    ref_chunks_np = np.stack(ref_chunks, axis=0).astype(np.float32)
    actor_chunks_np = np.stack(actor_chunks, axis=0).astype(np.float32)
    proprios_np = np.stack(proprios, axis=0).astype(np.float32)
    rewards_np = np.stack(rewards, axis=0).astype(np.float32)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = actor_owner_dir / "replay_real_robot_exports" / f"episode_{args.episode_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "playback_data.npz",
        ref_chunks=ref_chunks_np,
        actor_chunks=actor_chunks_np,
        proprios=proprios_np,
        collection_phases=np.asarray(collection_phases),
        rewards=rewards_np,
        step_ids=np.asarray(step_ids, dtype=np.int32),
        sources=np.asarray(sources, dtype=np.int32),
    )
    _write_playback_tables(
        output_dir,
        {
            "ref_chunks": ref_chunks_np,
            "actor_chunks": actor_chunks_np,
            "proprios": proprios_np,
            "collection_phases": np.asarray(collection_phases),
            "rewards": rewards_np,
            "step_ids": np.asarray(step_ids, dtype=np.int32),
            "sources": np.asarray(sources, dtype=np.int32),
        },
    )

    meta = {
        "run_dir": _relativize_path(args.run_dir, output_dir / "meta.json"),
        "replay_path": _relativize_path(replay_path, output_dir / "meta.json"),
        "snapshot_path": _relativize_path(snapshot_path, output_dir / "meta.json"),
        "actor_owner_dir": _relativize_path(actor_owner_dir, output_dir / "meta.json"),
        "episode_id": int(args.episode_id),
        "original_num_chunks": int(original_num_chunks),
        "exported_num_chunks": int(len(episode_records)),
        "collection_phase_counts": {
            phase: int(sum(item == phase for item in collection_phases))
            for phase in sorted(set(collection_phases))
        },
        "chunk_len": int(ref_chunks_np.shape[1]),
        "action_dim": int(ref_chunks_np.shape[2]),
        "drop_terminal_success": bool(not args.keep_terminal_success),
        "disable_ref_input": bool(args.disable_ref_input),
        "timing_recommendation": {
            "step_hz": 20.0,
            "step_interval_ms": 50.0,
            "chunk_boundary_interval_ms": 90.0,
            "note": "Chunk boundary interval replaces the normal 50ms step interval; it is not an extra delay on top of 50ms.",
        },
        "phase_definition": {
            "startup_reset": "Linearly interpolate from the current robot joint state to the first playback frame. This phase is not part of replay chunks.",
            "post_reset_hold": "Hold the first playback frame after startup_reset for clearer separation. This phase is not part of replay chunks.",
            "replay": "Publish exported replay chunks starting at chunk[0][0].",
        },
        "runtime_phase_plan": {
            "startup_reset_enabled": True,
            "startup_reset_duration_sec": 2.0,
            "post_reset_hold_enabled": True,
            "post_reset_hold_sec": 1.5,
            "replay_start_chunk": 0,
            "replay_start_step": 0,
        },
        "playback_files": {
            "npz": "playback_data.npz",
            "ref_chunks_key": "ref_chunks",
            "actor_chunks_key": "actor_chunks",
        },
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    print(f"Exported episode {args.episode_id} to {output_dir}")
    print(f"Replay: {replay_path}")
    print(f"Snapshot: {snapshot_path}")
    print(f"Chunks: {len(episode_records)} (original {original_num_chunks})")
    print("Files:")
    print(f"  - {output_dir / 'playback_data.npz'}")
    print(f"  - {output_dir / 'meta.json'}")
    print(f"  - {output_dir / 'tables'}")


if __name__ == "__main__":
    main()
