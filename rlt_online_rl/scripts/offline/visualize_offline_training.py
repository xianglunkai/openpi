from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

"""
Visualize offline training results.

The script reads an offline training directory, plots loss/Q/BC/delta/replay-fit
metrics, and helps check whether offline training is converging and whether the
actor is moving closer to the current BC target.

Default inputs:

- metrics: <train-dir>/metrics.jsonl
- status:  <train-dir>/status.json

Default output:

- <train-dir>/analysis/

Outputs:

- loss_curves.png
- fit_curves.png
- joint_fit_curves.png
- summary.json

Example:

python3 scripts/offline/visualize_offline_training.py \
  --train-dir runs/agilex_ethernet/offline_train_bcq
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize offline training metrics.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Offline training directory that contains metrics.jsonl and status.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Optional output directory. Defaults to <train-dir>/analysis"
    )
    return parser.parse_args()


def _load_metrics(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No metrics found in {path}")
    return rows


def _series(rows: list[dict], key: str, *, actor_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for row in rows:
        if key not in row:
            continue
        if actor_only and float(row.get("did_actor_update", 0.0)) < 0.5:
            continue
        value = float(row[key])
        if not np.isfinite(value):
            continue
        xs.append(float(row["global_step"]))
        ys.append(value)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    if y.size == 0 or window <= 1 or y.size < window:
        return y
    kernel = np.ones(window, dtype=np.float32) / float(window)
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _plot_losses(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    plot_specs = [
        ("critic_loss", "Critic Loss", False),
        ("actor_loss", "Actor Loss", True),
        ("actor_q", "Actor Q", True),
        ("bc_penalty", "BC Penalty", True),
        ("delta_penalty", "Delta Penalty", True),
        ("weighted_delta", "Weighted Delta", True),
    ]
    for ax, (key, title, actor_only) in zip(axes.reshape(-1), plot_specs, strict=True):
        x, y = _series(rows, key, actor_only=actor_only)
        if x.size == 0:
            ax.set_title(title)
            ax.set_xlabel("global_step")
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.25)
            continue
        ax.plot(x, y, linewidth=1.8)
        if actor_only:
            smooth = _moving_average(y, window=min(101, max(5, (y.size // 20) * 2 + 1)))
            ax.plot(x, smooth, linewidth=2.2, color="tab:red", alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("global_step")
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_fit(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key, title in (
        (axes[0], "train_mean_abs_delta", "Train Fit"),
        (axes[0], "val_mean_abs_delta", "Val Fit"),
        (axes[1], "val_max_abs_delta", "Val Max Delta"),
    ):
        x, y = _series(rows, key)
        if x.size == 0:
            continue
        ax.plot(x, y, linewidth=1.8, label=title if ax is axes[0] else key)
        ax.set_xlabel("global_step")
        ax.grid(True, alpha=0.25)
    axes[0].set_title("Mean Absolute Delta")
    axes[0].set_ylabel("|actor - ref|")
    axes[0].legend(frameon=False)
    axes[1].set_title("Validation Max Absolute Delta")
    axes[1].set_ylabel("max |actor - ref|")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_joint_fit(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(12, 12))
    axes = axes.reshape(-1)
    labels = [f"joint{i + 1}" for i in range(6)] + ["gripper"]
    for joint_idx, label in enumerate(labels):
        ax = axes[joint_idx]
        x, y = _series(rows, f"val_{label}_mean_abs_delta")
        if x.size:
            ax.plot(x, y, linewidth=1.8)
        ax.set_title(label)
        ax.set_xlabel("global_step")
        ax.set_ylabel("val mean |actor - ref|")
        ax.grid(True, alpha=0.25)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_summary(rows: list[dict], status_path: Path, output_path: Path) -> None:
    last = rows[-1]
    _, actor_loss_vals = _series(rows, "actor_loss", actor_only=True)
    _, actor_q_vals = _series(rows, "actor_q", actor_only=True)
    _, bc_penalty_vals = _series(rows, "bc_penalty", actor_only=True)
    _, delta_penalty_vals = _series(rows, "delta_penalty", actor_only=True)
    _, weighted_delta_vals = _series(rows, "weighted_delta", actor_only=True)
    payload = {
        "last_global_step": int(last["global_step"]),
        "last_actor_version": int(last["actor_version"]),
        "last_critic_loss": float(last["critic_loss"]),
        "last_actor_loss": float(actor_loss_vals[-1]) if actor_loss_vals.size else float("nan"),
        "last_actor_q": float(actor_q_vals[-1]) if actor_q_vals.size else float("nan"),
        "last_bc_penalty": float(bc_penalty_vals[-1]) if bc_penalty_vals.size else float("nan"),
        "last_delta_penalty": float(delta_penalty_vals[-1]) if delta_penalty_vals.size else float("nan"),
        "last_weighted_delta": float(weighted_delta_vals[-1]) if weighted_delta_vals.size else float("nan"),
        "last_train_mean_abs_delta": float(last.get("train_mean_abs_delta", np.nan)),
        "last_val_mean_abs_delta": float(last.get("val_mean_abs_delta", np.nan)),
        "last_val_max_abs_delta": float(last.get("val_max_abs_delta", np.nan)),
    }
    if status_path.exists():
        payload["status"] = json.loads(status_path.read_text(encoding="utf-8"))
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    train_dir = args.train_dir.resolve()
    metrics_path = train_dir / "metrics.jsonl"
    status_path = train_dir / "status.json"
    output_dir = (args.output_dir or (train_dir / "analysis")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_metrics(metrics_path)
    _plot_losses(rows, output_dir / "loss_curves.png")
    _plot_fit(rows, output_dir / "fit_curves.png")
    _plot_joint_fit(rows, output_dir / "joint_fit_curves.png")
    _write_summary(rows, status_path, output_dir / "summary.json")
    print(f"wrote offline training visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
