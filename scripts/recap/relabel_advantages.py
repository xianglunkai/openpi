"""Relabel an existing advantage sidecar with a new quantile, without re-inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from openpi.recap.advantages import binarize_advantages, write_advantage_sidecar

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pandas is required for recap labeling scripts") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--positive-fraction", type=float, default=0.3)
    parser.add_argument("--per-task", action="store_true", default=False)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input) if args.input.suffix != ".csv" else pd.read_csv(args.input)
    if "advantage_continuous" not in frame.columns:
        raise ValueError("Input sidecar must contain advantage_continuous for relabeling.")
    task_indices = frame["task_index"].to_numpy() if "task_index" in frame.columns else None
    labels, _ = binarize_advantages(
        frame["advantage_continuous"].to_numpy(),
        positive_fraction=args.positive_fraction,
        task_indices=task_indices,
        per_task=args.per_task and task_indices is not None,
        force_intervention_positive=False,
    )
    extra = {}
    if "task_index" in frame.columns:
        extra["task_index"] = frame["task_index"].to_numpy()
    write_advantage_sidecar(
        args.output,
        episode_index=frame["episode_index"].to_numpy(),
        frame_index=frame["frame_index"].to_numpy(),
        advantage=labels,
        advantage_continuous=frame["advantage_continuous"].to_numpy(),
        extra=extra or None,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
