"""Post-processing wrapper for action chunks used in rollout / actor inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rlt_online_rl.temporal_profile_utils import PassThroughOptimizer
from rlt_online_rl.temporal_profile_utils import TimeParameterizationMPC


@dataclass
class ActionChunkTimeAxisSmoother:
    """Re-time action chunks; with v_max + dt_max can flatten high-freq spikes.

    Expects action chunks shaped (T, A) or (B, T, A). Disabled by default.

    For reference-style spike flattening, use ``from_reference_preset()``.
    """

    enabled: bool = False
    dt_ref: float = 0.033
    dt_min: float = 0.025
    dt_max: float = 0.038
    lambda_acc: float = 10.0
    lambda_time: float = 1.0
    stride: int = 50
    optim_dims: list[int] = field(default_factory=lambda: list(range(14)))
    v_max: float | None = None
    lambda_v: float = 1.0
    horizon: int = 50
    logging: bool = False
    _optimizer: Any = field(init=False, repr=False, default=None)

    @classmethod
    def from_reference_preset(
        cls,
        *,
        dt_ref: float = 0.033,
        horizon: int = 50,
        optim_dims: list[int] | None = None,
        enabled: bool = True,
    ) -> ActionChunkTimeAxisSmoother:
        """Preset tuned for oscillation flattening (see demo 00_reference_effect.png)."""
        return cls(
            enabled=enabled,
            dt_ref=dt_ref,
            dt_min=0.025,
            dt_max=0.038,
            lambda_acc=10.0,
            lambda_time=1.0,
            v_max=None,
            stride=horizon,
            horizon=horizon,
            optim_dims=list(optim_dims) if optim_dims is not None else list(range(14)),
        )

    def __post_init__(self) -> None:
        self.dt_ref = float(self.dt_ref)
        self.dt_min = float(self.dt_min)
        self.dt_max = float(self.dt_max)
        self.lambda_acc = float(self.lambda_acc)
        self.lambda_time = float(self.lambda_time)
        self.stride = int(self.stride)
        self.horizon = int(self.horizon)
        self.lambda_v = float(self.lambda_v)
        self.logging = bool(self.logging)

    def _ensure_optimizer(self) -> None:
        if self._optimizer is None:
            self._optimizer = (
                TimeParameterizationMPC(
                    dt_ref=self.dt_ref,
                    dt_min=self.dt_min,
                    dt_max=self.dt_max,
                    lambda_acc=self.lambda_acc,
                    lambda_time=self.lambda_time,
                    stride=self.stride,
                    optim_dims=self.optim_dims,
                    v_max=self.v_max,
                    lambda_v=self.lambda_v,
                    horizon=self.horizon,
                    logging=self.logging,
                )
                if self.enabled
                else PassThroughOptimizer()
            )

    def __call__(self, action_chunk: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.asarray(action_chunk, dtype=np.float32)

        chunk = np.asarray(action_chunk, dtype=np.float32)
        batched = chunk.ndim == 3
        if not batched:
            chunk = chunk[None, ...]

        self._ensure_optimizer()
        out_batches: list[np.ndarray] = []
        for b in range(chunk.shape[0]):
            traj = chunk[b]
            try:
                optimized = np.asarray(self._optimizer.optimize(traj.tolist()), dtype=np.float32)
            except Exception as exc:
                if self.logging:
                    print(f"ActionChunkTimeAxisSmoother: batch {b} failed ({exc}); using original.")
                optimized = traj
            out_batches.append(optimized)

        out = np.stack(out_batches, axis=0)
        return out if batched else out[0]
