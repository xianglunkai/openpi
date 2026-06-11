from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

osqp = pytest.importorskip("osqp")
scipy = pytest.importorskip("scipy")  # noqa: F401

from rlt_online_rl.action_chunk_smoother import ActionChunkTimeAxisSmoother
from rlt_online_rl.temporal_profile_utils import TimeParameterizationMPC


def _finite_diff(x: np.ndarray, dt: float, order: int) -> np.ndarray:
    if order == 1:
        return np.diff(x, axis=0) / dt
    v = np.diff(x, axis=0) / dt
    return np.diff(v, axis=0) / dt


def test_optimize_reduces_speed_variance_on_temporal_mismatch() -> None:
    """Time-axis should flatten segment speeds when path geometry is smooth but unevenly sampled."""
    u_fine = np.linspace(0.0, 1.0, 500)
    path = np.stack([np.sin(2 * np.pi * u_fine), np.cos(2 * np.pi * u_fine)], axis=-1)
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    arc /= arc[-1]
    u_sample = 0.15 + 0.70 * 0.5 * (1.0 - np.cos(np.pi * np.linspace(0, 1, 40)))
    idx = np.clip(np.searchsorted(arc, u_sample), 0, len(path) - 1)
    traj = path[idx].astype(np.float32)

    optimizer = TimeParameterizationMPC(
        dt_ref=0.033,
        dt_min=0.02,
        dt_max=0.06,
        lambda_acc=50.0,
        lambda_time=0.3,
        stride=40,
        optim_dims=[0, 1],
        horizon=40,
    )
    out = np.asarray(optimizer.optimize(traj.tolist()), dtype=np.float32)
    dt = 0.033
    speed_std_before = float(np.std(np.linalg.norm(np.diff(traj, axis=0), axis=1) / dt))
    speed_std_after = float(np.std(np.linalg.norm(np.diff(out, axis=0), axis=1) / dt))
    assert speed_std_after < speed_std_before * 0.9


def test_reference_trajectory_flattens_oscillations() -> None:
    """On each trajectory's own time axis, optimized acc/vel spikes collapse."""
    dt = 0.033
    t = np.arange(50, dtype=np.float32) * dt
    pos = np.zeros(50, dtype=np.float32)
    for i, ti in enumerate(t):
        if ti < 0.52:
            pos[i] = 0.68 * ti
        elif ti < 1.02:
            base = 0.35 + 0.40 * (ti - 0.52) / 0.50
            pos[i] = base + 0.22 * np.sin(2 * np.pi * (ti - 0.52) / 0.09)
        else:
            pos[i] = 0.75 + (ti - 1.02) / (t[-1] - 1.02 + 1e-6) * 0.25
    traj = np.stack([pos, np.zeros(50, dtype=np.float32)], axis=-1)

    mpc = TimeParameterizationMPC(
        dt_ref=dt,
        dt_min=0.01,
        dt_max=0.35,
        lambda_acc=500.0,
        lambda_time=0.001,
        v_max=0.75,
        stride=50,
        horizon=50,
        optim_dims=[0, 1],
    )
    out, t_opt, _ = mpc.resample_on_optimal_duration(traj)

    dt_orig = np.diff(t.astype(np.float64))
    dt_opt = np.diff(t_opt)
    v0 = np.diff(traj[:, 0]) / dt_orig
    v1 = np.diff(out[:, 0]) / dt_opt
    a0 = np.diff(v0) / dt_orig[1:]
    a1 = np.diff(v1) / dt_opt[1:]

    assert np.max(np.abs(a1)) < np.max(np.abs(a0)) * 0.05
    assert t_opt[-1] > t[-1] * 2.0


def test_solve_qp_respects_speed_bounds() -> None:
    waypoints = np.stack([np.linspace(0.0, 1.0, 10), np.zeros(10)], axis=-1)
    optimizer = TimeParameterizationMPC(
        dt_ref=0.05,
        dt_min=0.02,
        dt_max=0.08,
        horizon=8,
        stride=8,
        optim_dims=[0, 1],
    )
    optimizer.dp = waypoints[1:] - waypoints[:-1]
    optimizer.N = len(optimizer.dp)
    s = optimizer.solve_qp(0)
    assert np.all(s >= optimizer.s_min - 1e-6)
    assert np.all(s <= optimizer.s_max + 1e-6)


def test_action_chunk_smoother_batch_and_disabled() -> None:
    traj = np.ones((20, 3), dtype=np.float32)
    traj[:, 0] = np.linspace(0.0, 1.0, 20)

    disabled = ActionChunkTimeAxisSmoother(enabled=False)
    assert np.allclose(disabled(traj), traj)

    enabled = ActionChunkTimeAxisSmoother(
        enabled=True,
        dt_ref=0.033,
        dt_min=0.02,
        dt_max=0.06,
        optim_dims=[0, 1, 2],
        horizon=20,
        stride=20,
        lambda_acc=10.0,
    )
    batch = np.stack([traj, traj + 0.1], axis=0)
    out = enabled(batch)
    assert out.shape == batch.shape
