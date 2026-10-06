"""Time-axis reparameterization for action chunks via convex QP (OSQP).

Optimizes per-segment speeds s_i = 1/dt_i along the polyline through input
waypoints, then resamples at uniform dt_ref.

Typical effect (with v_max, large dt_max, low lambda_time):
  - High-frequency position oscillations are pulled flat on the output grid
  - Velocity / acceleration spikes are strongly reduced

Mechanism: v_max forces slow traversal of high-dp segments; the optimizer
compresses time spent in oscillating regions; uniform resampling then
interpolates over skipped vertices (not pure time-shift).

Tuning guide:
  - dt_max (large, e.g. 0.2–0.5): allow slowing down through spikes
  - v_max: cap ||dp||/dt; required for strong flattening
  - lambda_time (small, e.g. 0.001): allow time warping away from dt_ref
  - lambda_acc (moderate, e.g. 200–500): smooth segment-speed transitions
  - dt_min (small): do not confuse with dt_max — dt_min caps max speed, not min
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Sequence

import numpy as np
import osqp
import scipy.sparse as sp


class BaseOptimizer(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, inference_cfg) -> BaseOptimizer:
        raise NotImplementedError

    @abstractmethod
    def optimize(self, action_list: list) -> list:
        raise NotImplementedError


class PassThroughOptimizer(BaseOptimizer):
    @classmethod
    def from_config(cls, inference_cfg) -> PassThroughOptimizer:
        del inference_cfg
        return cls()

    def optimize(self, action_list: list) -> list:
        return action_list


class TimeParameterizationMPC(BaseOptimizer):
    """MPC-style rolling-horizon time parameterization along a fixed path."""

    @classmethod
    def from_config(cls, inference_cfg) -> TimeParameterizationMPC:
        return cls(
            dt_ref=inference_cfg.timeaxis_dt_ref_s,
            dt_min=inference_cfg.timeaxis_dt_min_s,
            dt_max=inference_cfg.timeaxis_dt_max_s,
            lambda_acc=inference_cfg.timeaxis_lambda_acc,
            lambda_time=inference_cfg.timeaxis_lambda_time,
            stride=inference_cfg.timeaxis_stride,
            optim_dims=inference_cfg.timeaxis_optdims,
            v_max=inference_cfg.timeaxis_v_max,
            lambda_v=inference_cfg.timeaxis_lambda_v,
            horizon=inference_cfg.timeaxis_horizon,
            logging=inference_cfg.timeaxis_logging,
        )

    def __init__(
        self,
        dt_ref: float = 0.05,
        dt_min: float = 0.01,
        dt_max: float = 0.3,
        lambda_acc: float = 1.0,
        lambda_time: float = 0.1,
        stride: int = 10,
        optim_dims: Sequence[int] | None = None,
        v_max: float | None = None,
        lambda_v: float = 10.0,
        horizon: int = 20,
        logging: bool = False,
    ) -> None:
        self.dt_ref = float(dt_ref)
        self.s_ref = 1.0 / self.dt_ref
        self.s_min = 1.0 / float(dt_max)
        self.s_max = 1.0 / float(dt_min)
        self.lambda_acc = float(lambda_acc)
        self.lambda_time = float(lambda_time)
        self.solve_stride = int(stride)
        self.optim_dims = list(optim_dims) if optim_dims is not None else [0, 1, 2, 3, 4, 5, 6]
        self.v_max = v_max
        self.lambda_v = float(lambda_v)
        self.H = int(horizon)
        self.logging = bool(logging)

    def _run_osqp(
        self,
        P: sp.csc_matrix,
        q: np.ndarray,
        A: sp.csc_matrix,
        l: np.ndarray,
        u: np.ndarray,
    ) -> np.ndarray:
        prob = osqp.OSQP()
        prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
        res = prob.solve()
        if res.info.status not in ("solved", "solved_inaccurate"):
            raise RuntimeError(f"OSQP failed with status={res.info.status}")
        return np.asarray(res.x, dtype=np.float64)

    def solve_qp(self, k: int) -> np.ndarray:
        H = min(self.H, self.N - k - 1)
        if H <= 0:
            return np.array([], dtype=np.float64)
        dp = self.dp[k : k + H]
        dp_norm = np.linalg.norm(dp, axis=1)
        scale_time = self.s_ref**2 + 1e-6
        scale_acc = np.mean((dp_norm * self.s_ref) ** 2) + 1e-6
        lambda_time = self.lambda_time / scale_time
        lambda_acc = self.lambda_acc / scale_acc

        n_var = H
        P = np.zeros((n_var, n_var))
        P += 2 * lambda_time * np.eye(n_var)
        for i in range(H - 1):
            P[i, i] += 2 * lambda_acc * np.sum(dp[i] ** 2)
            P[i + 1, i + 1] += 2 * lambda_acc * np.sum(dp[i + 1] ** 2)
            cross = 2 * lambda_acc * np.dot(dp[i], dp[i + 1])
            P[i, i + 1] -= cross
            P[i + 1, i] -= cross
        P = sp.csc_matrix(P)

        q = -2 * lambda_time * self.s_ref * np.ones(n_var)

        A = sp.eye(n_var)
        l = self.s_min * np.ones(n_var)
        u = self.s_max * np.ones(n_var)

        if self.v_max is not None:
            u = np.minimum(u, self.v_max / (dp_norm + 1e-8))

        # Spike segments can require s < s_min (slow down below dt_max floor) to satisfy v_max.
        l = np.minimum(l, u)

        try:
            return self._run_osqp(P, q, A, l, u)
        except RuntimeError:
            if self.v_max is None:
                raise
            if self.logging:
                print("TimeParameterizationMPC: infeasible v_max; retrying without v_max.")
            u = self.s_max * np.ones(n_var)
            return self._run_osqp(P, q, A, l, u)

    def re_allocate(self, waypoints: np.ndarray, ts: np.ndarray) -> np.ndarray:
        ts_out = np.arange(len(waypoints)) * self.dt_ref
        return np.apply_along_axis(lambda col: np.interp(ts_out, ts, col), axis=0, arr=waypoints)

    def resample_on_optimal_duration(
        self, waypoints: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Resample path on uniform grid spanning the optimized total duration.

        Returns (positions, timestamps, total_duration). Unlike ``re_allocate``,
        timestamps span ``[0, sum(dt_opt)]`` rather than ``[0, (T-1)*dt_ref]``.
        """
        waypoints = np.asarray(waypoints, dtype=np.float64)
        if waypoints.ndim == 1:
            waypoints = waypoints[:, None]
        n = len(waypoints)
        _, t_wp, dt_arr = self.solve(waypoints, 0, n)
        seg_dt = dt_arr[1:-1] if len(dt_arr) > 2 else dt_arr
        total_duration = float(np.sum(seg_dt))
        if total_duration <= 0.0:
            t_out = np.arange(n, dtype=np.float64) * self.dt_ref
            total_duration = float(t_out[-1]) if n > 1 else 0.0
            return waypoints.astype(np.float32), t_out.astype(np.float64), total_duration
        t_out = np.linspace(0.0, total_duration, n, dtype=np.float64)
        resampled = np.apply_along_axis(
            lambda col: np.interp(t_out, t_wp[:n], col), axis=0, arr=waypoints
        )
        return resampled.astype(np.float32), t_out, total_duration

    def solve(
        self, waypoints: np.ndarray, st_roll: int, end_roll: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.dp = waypoints[1:] - waypoints[:-1]
        self.N = len(self.dp)
        s_traj: list[float] = []
        k = st_roll

        while k < end_roll:
            st = time.time()
            s_opt = self.solve_qp(k)
            end = time.time()
            if self.logging:
                print(end - st)
            if len(s_opt) == 0:
                break
            s_traj += s_opt[: self.solve_stride].tolist()
            k += self.solve_stride

        s_traj_arr = np.asarray(s_traj, dtype=np.float64)[: (end_roll - st_roll)]
        if len(s_traj_arr) == 0:
            ts_out = np.arange(end_roll - st_roll) * self.dt_ref
            return waypoints[st_roll:end_roll].copy(), ts_out, np.full(end_roll - st_roll, self.dt_ref)

        dt_traj = 1.0 / s_traj_arr
        dt_traj = np.concatenate((dt_traj[:1], dt_traj, dt_traj[-1:]), axis=0)
        t = np.concatenate([[0.0], np.cumsum(dt_traj)])[:-1]
        optim_wp = self.re_allocate(waypoints[st_roll:end_roll], t[: end_roll - st_roll])
        return optim_wp, t, dt_traj

    def optimize(self, action_list: list) -> list:
        if not action_list:
            return action_list
        waypoints = np.asarray(action_list, dtype=np.float32)
        if waypoints.ndim == 1:
            waypoints = waypoints[:, None]
        n = len(waypoints)
        optim_wp, _, _ = self.solve(waypoints[:, self.optim_dims], 0, n)
        waypoints[: len(optim_wp), self.optim_dims] = optim_wp
        return np.asarray(waypoints, dtype=np.float32).tolist()
