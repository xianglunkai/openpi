"""Compute RLinf-style DynamicReturn sidecars for a LeRobot dataset.

Writes ``<dataset_root>/meta/returns_{tag}.parquet`` without mutating episode files.

Usage:
    uv run python scripts/recap/compute_returns.py \\
        --dataset-root /path/to/lerobot_dataset --dataset-type sft --tag fail300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.recap.advantages import compute_episode_returns_and_rewards, episode_boundaries

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required for recap labeling scripts") from exc


def _read_metadata_table(dataset_root: Path) -> pd.DataFrame:
    import pyarrow.parquet as pq

    data_dir = dataset_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {data_dir}")
    frames = []
    wanted = ["episode_index", "frame_index", "is_success", "task_index", "task"]
    for path in parquet_files:
        schema_names = set(pq.ParquetFile(path).schema.names)
        use = [c for c in wanted if c in schema_names]
        frames.append(pd.read_parquet(path, columns=use))
    return pd.concat(frames, ignore_index=True).sort_values(["episode_index", "frame_index"]).reset_index(drop=True)


def _episode_success(group: pd.DataFrame, dataset_type: str, success_field: str | None) -> bool:
    if dataset_type == "sft":
        return True
    if success_field and success_field in group.columns:
        return bool(group[success_field].iloc[-1])
    if "is_success" in group.columns:
        return bool(group["is_success"].iloc[-1])
    raise ValueError(
        "Rollout datasets need an is_success column or --success-field; "
        "or pass --dataset-type sft to treat every episode as successful."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-type", choices=("sft", "rollout"), default="sft")
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--failure-reward", type=float, default=-300.0)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--success-field", default=None)
    args = parser.parse_args()

    table = _read_metadata_table(args.dataset_root)
    rows = []
    for ep_id, start, end in episode_boundaries(table["episode_index"].to_numpy()):
        group = table.iloc[start:end]
        is_success = _episode_success(group, args.dataset_type, args.success_field)
        returns, rewards = compute_episode_returns_and_rewards(
            len(group),
            is_success=is_success,
            gamma=args.gamma,
            failure_reward=args.failure_reward,
        )
        prompt = ""
        if "task" in group.columns:
            prompt = str(group["task"].iloc[0])
        for local_i, (_, row) in enumerate(group.iterrows()):
            rows.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "frame_index": int(row["frame_index"]),
                    "return": float(returns[local_i]),
                    "reward": float(rewards[local_i]),
                    "is_success": bool(is_success),
                    "prompt": prompt,
                    "task_index": int(row["task_index"]) if "task_index" in row else 0,
                }
            )

    out_dir = args.dataset_root / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"returns_{args.tag}.parquet"
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    stats = {
        "tag": args.tag,
        "gamma": args.gamma,
        "failure_reward": args.failure_reward,
        "dataset_type": args.dataset_type,
        "num_rows": len(rows),
        "return_min": float(np.min([r["return"] for r in rows])),
        "return_max": float(np.max([r["return"] for r in rows])),
    }
    (out_dir / f"returns_{args.tag}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
