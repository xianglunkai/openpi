"""Binarize RECAP advantages into a sidecar parquet.

Score sources:
  - returns: quantile on G_t (π*0.6 pretrain approximation; no value model)
  - values: n-step TD using a values parquet with columns episode_index, frame_index, value

Usage:
    uv run python scripts/recap/compute_advantages.py \\
        --dataset-root /path/to/lerobot_dataset --returns-tag default --tag q30 \\
        --score-source returns --positive-fraction 0.3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.recap.advantages import (
    binarize_advantages,
    compute_n_step_advantages,
    write_advantage_sidecar,
)

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required for recap labeling scripts") from exc


def _load_returns(dataset_root: Path, tag: str) -> pd.DataFrame:
    path = dataset_root / "meta" / f"returns_{tag}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/recap/compute_returns.py first.")
    return pd.read_parquet(path)


def _load_values(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path) if path.suffix != ".csv" else pd.read_csv(path)
    if "value" not in frame.columns:
        raise ValueError(f"{path} must contain a 'value' column")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--returns-tag", default="default")
    parser.add_argument("--tag", default="q30")
    parser.add_argument("--score-source", choices=("returns", "values"), default="returns")
    parser.add_argument("--values-path", type=Path, default=None)
    parser.add_argument("--n-step", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--positive-fraction", type=float, default=0.3)
    parser.add_argument("--per-task", action="store_true", default=True)
    parser.add_argument("--no-per-task", action="store_false", dest="per_task")
    parser.add_argument("--force-intervention-positive", action="store_true", default=True)
    args = parser.parse_args()

    returns_df = _load_returns(args.dataset_root, args.returns_tag)
    episode_index = returns_df["episode_index"].to_numpy()
    frame_index = returns_df["frame_index"].to_numpy()
    rewards = returns_df["reward"].to_numpy(dtype=np.float32)
    returns = returns_df["return"].to_numpy(dtype=np.float32)
    task_indices = (
        returns_df["task_index"].to_numpy() if "task_index" in returns_df.columns else None
    )
    interventions = (
        returns_df["is_intervention"].to_numpy() if "is_intervention" in returns_df.columns else None
    )

    if args.score_source == "returns":
        scores = returns
    else:
        if args.values_path is None:
            raise ValueError("--values-path is required when --score-source=values")
        values_df = _load_values(args.values_path)
        merged = returns_df.merge(
            values_df[["episode_index", "frame_index", "value"]],
            on=["episode_index", "frame_index"],
            how="left",
        )
        if merged["value"].isna().any():
            raise ValueError("Some frames are missing values; check --values-path alignment.")
        values = merged["value"].to_numpy(dtype=np.float32)
        scores = compute_n_step_advantages(
            rewards,
            values,
            episode_index,
            frame_index,
            n_step=args.n_step,
            gamma=args.gamma,
            returns=returns,
            return_min=float(returns.min()),
            return_max=float(returns.max()),
        )

    labels, thresholds = binarize_advantages(
        scores,
        positive_fraction=args.positive_fraction,
        task_indices=task_indices,
        interventions=interventions,
        force_intervention_positive=args.force_intervention_positive,
        per_task=args.per_task and task_indices is not None,
    )

    out_path = args.dataset_root / "meta" / f"advantages_{args.tag}.parquet"
    write_advantage_sidecar(
        out_path,
        episode_index=episode_index,
        frame_index=frame_index,
        advantage=labels,
        advantage_continuous=scores,
    )
    stats = {
        "tag": args.tag,
        "score_source": args.score_source,
        "positive_fraction": args.positive_fraction,
        "positive_ratio": float(np.mean(labels)),
        "thresholds": thresholds,
        "n_step": args.n_step,
        "gamma": args.gamma,
    }
    (out_path.parent / f"advantages_{args.tag}_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"Wrote {out_path} (positive_ratio={stats['positive_ratio']:.3f})")
    print("Train with: --recap.advantage-path", out_path)


if __name__ == "__main__":
    main()
