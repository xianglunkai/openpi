#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig-rlt"))

import matplotlib.pyplot as plt

"""
Plot learner metrics from an online run directory or an offline training directory.

Examples:
python3 scripts/tools/plot_learner_metrics.py runs/agilex_ethernet
python3 scripts/tools/plot_learner_metrics.py runs/agilex_ethernet/offline_train_bcq
"""

EMA_ALPHA = 0.1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot learner metrics. The input can be an online run_dir, offline training dir, "
            "logs dir, or a metrics jsonl file."
        )
    )
    parser.add_argument("input_path", type=Path, help="Run dir, offline dir, logs dir, or metrics jsonl path.")
    return parser.parse_args()


def _resolve_metrics_path(input_path: Path) -> tuple[Path, Path]:
    path = input_path.expanduser().resolve()
    if path.is_file():
        return path.parent, path
    if not path.is_dir():
        raise FileNotFoundError(f"Input not found: {path}")

    if path.name == "logs":
        path = path.parent

    candidates = (
        path / "metrics" / "learner_metrics.jsonl",
        path / "metrics.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return path, candidate
    raise FileNotFoundError(f"Could not find learner metrics under {path}")


def _load_records(metrics_path: Path) -> list[dict]:
    records: list[dict] = []
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No learner metrics found in {metrics_path}")
    return records


def _ema(values: np.ndarray, *, alpha: float = EMA_ALPHA) -> np.ndarray:
    if values.size == 0:
        return values
    out = np.empty_like(values, dtype=np.float64)
    out[0] = float(values[0])
    for i in range(1, values.size):
        out[i] = alpha * float(values[i]) + (1.0 - alpha) * float(out[i - 1])
    return out


def _masked_series(steps: np.ndarray, values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return steps[mask], values[mask]


def _series(records: list[dict], key: str, *, default: float = np.nan) -> np.ndarray:
    return np.asarray([float(record.get(key, default)) for record in records], dtype=np.float64)


def main() -> int:
    args = _parse_args()
    output_root, metrics_path = _resolve_metrics_path(args.input_path)
    records = _load_records(metrics_path)

    steps = _series(records, "global_step")
    critic_loss = _series(records, "critic_loss")
    did_actor_update = np.asarray([float(record.get("did_actor_update", 1.0)) > 0.5 for record in records], dtype=bool)
    actor_loss = _series(records, "actor_loss")
    bc_penalty = _series(records, "bc_penalty")
    actor_q = _series(records, "actor_q")
    replay_size = _series(records, "replay_size")

    actor_steps, actor_loss_values = _masked_series(steps, actor_loss, did_actor_update)
    _, bc_values = _masked_series(steps, bc_penalty, did_actor_update)
    _, actor_q_values = _masked_series(steps, actor_q, did_actor_update)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(steps, critic_loss, color="tab:blue", alpha=0.25, linewidth=1.0, label="critic_loss")
    ax.plot(steps, _ema(critic_loss), color="tab:blue", linewidth=2.0, label="critic_loss_ema")
    ax.set_title("Critic Loss")
    ax.set_xlabel("global_step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    if actor_steps.size > 0:
        ax.plot(actor_steps, actor_loss_values, color="tab:orange", alpha=0.3, linewidth=1.0, label="actor_loss")
        ax.plot(actor_steps, _ema(actor_loss_values), color="tab:orange", linewidth=2.0, label="actor_loss_ema")
    ax.set_title("Actor Loss")
    ax.set_xlabel("global_step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    if actor_steps.size > 0:
        ax.plot(actor_steps, bc_values, color="tab:green", alpha=0.3, linewidth=1.0, label="bc_penalty")
        ax.plot(actor_steps, _ema(bc_values), color="tab:green", linewidth=2.0, label="bc_penalty_ema")
        ax.plot(actor_steps, actor_q_values, color="tab:red", alpha=0.25, linewidth=1.0, label="actor_q")
        ax.plot(actor_steps, _ema(actor_q_values), color="tab:red", linewidth=2.0, label="actor_q_ema")
    ax.set_title("Actor Q / BC")
    ax.set_xlabel("global_step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(steps, replay_size, color="tab:purple", linewidth=2.0, label="replay_size")
    ax.set_title("Replay Size")
    ax.set_xlabel("global_step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(output_root.name)
    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / "learner_curves.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
