#!/usr/bin/env python3
"""Quantile stats for the actor's delta_chunk adapter.

The journal stores absolute joints. ``ActionRepresentationAdapter`` subtracts
the anchor proprio from the first ``pose_dim`` joints and leaves the remaining
dims (gripper) absolute, then maps ``[q01, q99]`` onto ``[-1, 1]``. These
stats are that representation, pooled over every timestep of ``action_chunk``,
``ref_chunk``, and ``next_ref_chunk``.

Joint dims use the 1% and 99% quantiles. Gripper and any other absolute dim
use min and max, because deploy clips to this range and a percentile would
cut real openings.

The previous VLA file is left untouched. Pass ``--force`` only when the output
path is intentionally that file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np

_SOURCE_IDS = {"all": None, "base": {0}, "rl": {1}, "human": {2}, "mixed": {3}, "no-rl": {0, 2, 3}}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write actor delta_chunk norm stats from a replay journal.")
    parser.add_argument("--replay-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose-dim", type=int, default=6, help="Leading dims converted to chunk deltas.")
    parser.add_argument("--phase", choices=("all", "warmup", "online", "unknown"), default="all")
    parser.add_argument("--source", choices=tuple(_SOURCE_IDS), default="all")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Existing norm json. Prints per-joint span ratios. Refuses to overwrite it without --force.",
    )
    parser.add_argument("--force", action="store_true", help="Allow --output to overwrite --compare.")
    return parser.parse_args()


def _to_representation(chunk: np.ndarray, anchor: np.ndarray, pose_dim: int) -> np.ndarray:
    represented = np.asarray(chunk, dtype=np.float32).copy()
    represented[:, :pose_dim] = represented[:, :pose_dim] - np.asarray(anchor, dtype=np.float32)[:pose_dim]
    return represented


def _keep(record: dict, *, phase: str, source_ids: set[int] | None) -> bool:
    if phase != "all" and str(record.get("collection_phase", "unknown")) != phase:
        return False
    if source_ids is not None and int(record["source"]) not in source_ids:
        return False
    return True


def _load_represented(path: Path, *, pose_dim: int, phase: str, source: str) -> tuple[np.ndarray, int]:
    source_ids = _SOURCE_IDS[source]
    pieces: list[np.ndarray] = []
    n_records = 0
    with path.open("rb") as handle:
        while True:
            try:
                record = pickle.load(handle)
            except EOFError:
                break
            if not _keep(record, phase=phase, source_ids=source_ids):
                continue
            n_records += 1
            proprio = np.asarray(record["proprio"], dtype=np.float32)
            next_proprio = np.asarray(record["next_proprio"], dtype=np.float32)
            pieces.append(_to_representation(record["action_chunk"], proprio, pose_dim))
            pieces.append(_to_representation(record["ref_chunk"], proprio, pose_dim))
            pieces.append(_to_representation(record["next_ref_chunk"], next_proprio, pose_dim))
    if n_records == 0:
        raise SystemExit(f"No records left after phase={phase} source={source}: {path}")
    return np.concatenate(pieces, axis=0), n_records


def _quantile_stats(values: np.ndarray, pose_dim: int) -> tuple[np.ndarray, np.ndarray]:
    q01 = np.quantile(values, 0.01, axis=0).astype(np.float64)
    q99 = np.quantile(values, 0.99, axis=0).astype(np.float64)
    # Absolute dims are physical commands. A percentile here becomes a hard clip.
    q01[pose_dim:] = np.min(values[:, pose_dim:], axis=0)
    q99[pose_dim:] = np.max(values[:, pose_dim:], axis=0)
    if np.any(q99 <= q01):
        flat = np.where(q99 <= q01)[0].tolist()
        print(f"warning: q99 <= q01 on dims {flat}; the adapter adds 1e-6 to the span")
    return q01, q99


def _load_previous_actions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions = payload["norm_stats"]["actions"]
    return np.asarray(actions["q01"], dtype=np.float64), np.asarray(actions["q99"], dtype=np.float64)


def _print_comparison(q01: np.ndarray, q99: np.ndarray, previous: Path) -> None:
    old_q01, old_q99 = _load_previous_actions(previous)
    if old_q01.shape != q01.shape:
        raise SystemExit(f"Action dim {q01.shape[0]} does not match {previous} dim {old_q01.shape[0]}")
    old_span = old_q99 - old_q01
    new_span = q99 - q01
    print(f"compare {previous}")
    print(f"{'dim':>4} {'new_q01':>12} {'new_q99':>12} {'new_span':>12} {'old_span':>12} {'old/new':>8}")
    for index, (lo, hi, span, old) in enumerate(zip(q01, q99, new_span, old_span)):
        ratio = float(old / span) if span > 0 else float("inf")
        print(f"{index:4d} {lo:12.6f} {hi:12.6f} {span:12.6f} {old:12.6f} {ratio:8.2f}")


def main() -> None:
    args = _parse_args()
    replay_path = args.replay_path.resolve()
    output = args.output.resolve()
    if args.compare is not None and output == args.compare.resolve() and not args.force:
        raise SystemExit(f"Refusing to overwrite {output}. Pass --force to replace that file.")
    values, n_records = _load_represented(
        replay_path,
        pose_dim=args.pose_dim,
        phase=args.phase,
        source=args.source,
    )
    if values.shape[1] <= args.pose_dim:
        raise SystemExit(f"pose_dim={args.pose_dim} leaves no absolute dims in action_dim={values.shape[1]}")
    q01, q99 = _quantile_stats(values, args.pose_dim)
    payload = {
        "norm_stats": {
            "actions": {
                "mean": values.mean(axis=0).astype(float).tolist(),
                "std": values.std(axis=0).astype(float).tolist(),
                "q01": q01.tolist(),
                "q99": q99.tolist(),
            }
        },
        "meta": {
            "replay_path": str(replay_path),
            "n_records": n_records,
            "n_samples": int(values.shape[0]),
            "phase": args.phase,
            "source": args.source,
            "pose_dim": args.pose_dim,
            "joint_range": "q01/q99 of (chunk - anchor proprio)",
            "absolute_dims": f"{args.pose_dim}: use min/max so deploy clip keeps the full command",
            "tensors": ["action_chunk@proprio", "ref_chunk@proprio", "next_ref_chunk@next_proprio"],
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"records={n_records} samples={values.shape[0]} action_dim={values.shape[1]}")
    print(f"wrote {output}")
    if args.compare is not None:
        _print_comparison(q01, q99, args.compare.resolve())


if __name__ == "__main__":
    main()
