#!/usr/bin/env python3
"""Visual demo for time-axis action chunk optimization.

X-axis spans the original chunk duration. Optimized trajectory uses its own
timestamps but is clipped to t <= T_orig for comparison.

Run:
  cd rlt_online_rl
  pip install osqp scipy matplotlib
  PYTHONPATH=src python scripts/tools/demo_time_axis_optimization.py
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig-rlt-timeaxis"))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "matplotlib is required. Install with: pip install matplotlib osqp scipy"
    ) from exc

from rlt_online_rl.temporal_profile_utils import TimeParameterizationMPC


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo time-axis optimization.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/time_axis_demo"))
    parser.add_argument("--dt-ref", type=float, default=0.033)
    parser.add_argument("--horizon", type=int, default=50)
    return parser.parse_args()


def _make_trajectory(T: int, dt: float) -> np.ndarray:
    t = np.arange(T, dtype=np.float32) * dt
    pos = np.zeros(T, dtype=np.float32)
    for i, ti in enumerate(t):
        if ti < 0.52:
            pos[i] = 0.68 * ti
        elif ti < 1.02:
            base = 0.35 + 0.40 * (ti - 0.52) / 0.50
            pos[i] = base + 0.22 * np.sin(2 * np.pi * (ti - 0.52) / 0.09)
        else:
            pos[i] = 0.75 + (ti - 1.02) / (t[-1] - 1.02 + 1e-6) * 0.25
    return pos


def _vel_acc(t: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt_seg = np.diff(t)
    vel = np.diff(pos) / dt_seg
    acc = np.diff(vel) / dt_seg[1:]
    t_vel = t[:-1] + 0.5 * dt_seg
    t_acc = t[1:-1]
    return t_vel, vel, t_acc, acc


def _clip_by_time(
    t: np.ndarray, *series: np.ndarray, t_max: float
) -> tuple[np.ndarray, ...]:
    """Keep samples with t <= t_max (always include at least two points)."""
    mask = t <= t_max + 1e-9
    if np.count_nonzero(mask) < 2:
        mask = np.zeros_like(t, dtype=bool)
        mask[:2] = True
    out: list[np.ndarray] = [t[mask]]
    for s in series:
        out.append(s[mask])
    return tuple(out)


def main() -> None:
    args = _parse_args()
    dt = args.dt_ref
    T = args.horizon
    original_pos = _make_trajectory(T, dt)

    t_orig = np.arange(T, dtype=np.float64) * dt
    T_orig = float(t_orig[-1])

    mpc = TimeParameterizationMPC(
        dt_ref=dt,
        dt_min=0.01,
        dt_max=0.35,
        lambda_acc=500.0,
        lambda_time=0.001,
        v_max=0.75,
        stride=T,
        horizon=T,
        optim_dims=[0],
    )
    opt_pos, t_opt, T_opt = mpc.resample_on_optimal_duration(original_pos[:, None])
    opt_pos = opt_pos[:, 0]

    tv0, v0, ta0, a0 = _vel_acc(t_orig, original_pos)
    tv1, v1, ta1, a1 = _vel_acc(t_opt, opt_pos)

    t_plot_opt, opt_pos_plot = _clip_by_time(t_opt, opt_pos, t_max=T_orig)
    tv1, v1 = _clip_by_time(tv1, v1, t_max=T_orig)
    ta1, a1 = _clip_by_time(ta1, a1, t_max=T_orig)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))

    axes[0].plot(t_orig, original_pos, "o--", color="C0", markersize=4, label="Original", alpha=0.85)
    axes[0].plot(
        t_plot_opt,
        opt_pos_plot,
        "s-",
        color="C1",
        markersize=4,
        label=f"Optimized (own clock, T_full={T_opt:.2f}s)",
        linewidth=2,
    )
    axes[0].set_ylabel("Position (m)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(tv0, v0, "o--", color="C0", markersize=3, alpha=0.85)
    axes[1].plot(tv1, v1, "s-", color="C1", markersize=3, linewidth=2)
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ta0, a0, "o--", color="C0", markersize=3, alpha=0.85)
    axes[2].plot(ta1, a1, "s-", color="C1", markersize=3, linewidth=2)
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlim(0.0, T_orig * 1.02)

    fig.tight_layout()
    out_path = args.output_dir / "time_axis_demo.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"plot window: 0–{T_orig:.2f}s (original duration)")
    print(f"optimized full duration: {T_opt:.2f}s (shown up to {T_orig:.2f}s)")
    print(f"|acc| max: {np.max(np.abs(a0)):.1f} -> {np.max(np.abs(a1)):.1f}")
    print(f"saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
