from typing import Dict, Optional, Tuple

import numpy as np
import tree
import time
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.

    Optionally applies VLA-RAIL intra-chunk trajectory smoothing.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        use_smoothing: bool = False,
        polynomial_order: int = 3,
        preserve_boundaries: bool = True
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0
        self._use_smoothing = use_smoothing
        self._polynomial_order = polynomial_order
        self._preserve_boundaries = preserve_boundaries

        self._last_results: Dict[str, np.ndarray] | None = None

    @staticmethod
    def _slice_action_chunk(results: Dict[str, np.ndarray], cur_step: int) -> Dict[str, np.ndarray]:
        """Slice action chunk from cached results."""
        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[cur_step, ...]
            else:
                return x
        return tree.map_structure(slicer, results)

    @staticmethod
    def intra_chunk_smoothing_vla_rail(
        actions: np.ndarray,
        polynomial_order: int = 3,
        preserve_boundaries: bool = True
    ) -> np.ndarray:
        """
        VLA-RAIL intra-chunk trajectory smoothing using polynomial fitting (NumPy version).

        Args:
            actions: Action tensor of shape (T, A), where
                T = chunk size (time steps),
                A = action dimension
            polynomial_order: Order of polynomial for fitting (default=3 for cubic)
            preserve_boundaries: If True, keep the first and last points unchanged

        Returns:
            Smoothed action tensor of shape (T, A)
        """
        # Handle different input shapes
        if actions.ndim == 1:
            # Shape (T,) - single action dimension
            actions = actions.reshape(-1, 1)
        elif actions.ndim != 2:
            # Unsupported shape, return as-is
            return actions

        T, A = actions.shape

        # Ensure we have enough points for polynomial fitting
        if T < polynomial_order + 1:
            return actions

        # Create time indices (0, 1, 2, ..., T-1)
        t = np.arange(T, dtype=np.float32).reshape(-1, 1)  # (T, 1)

        # Build Vandermonde matrix for polynomial basis [1, t, t^2, t^3, ...]
        powers = np.arange(polynomial_order + 1, dtype=np.float32)
        V = t ** powers  # (T, polynomial_order+1)

        # Compute Moore-Penrose pseudoinverse: (V^T V)^(-1) V^T
        VtV = V.T @ V  # (polynomial_order+1, polynomial_order+1)

        # Add small regularization for numerical stability
        reg = np.eye(polynomial_order + 1, dtype=np.float32) * 1e-6
        VtV_reg = VtV + reg

        # Compute pseudoinverse using Cholesky decomposition for stability
        try:
            L = np.linalg.cholesky(VtV_reg)
            # Solve L @ L.T @ X = I => X = (L.T)^(-1) @ L^(-1)
            L_inv = np.linalg.solve(L, np.eye(polynomial_order + 1))
            VtV_inv = L_inv.T @ L_inv
            V_pinv = VtV_inv @ V.T  # (polynomial_order+1, T)
        except np.linalg.LinAlgError:
            # Fallback to SVD-based pseudoinverse
            U, S, Vh = np.linalg.svd(V, full_matrices=False)
            S_inv = np.diag(1.0 / (S + 1e-6))
            V_pinv = Vh.T @ S_inv @ U.T

        # Compute polynomial coefficients
        # V_pinv shape: (polynomial_order+1, T)
        # actions.T shape: (A, T)
        # coefficients shape: (polynomial_order+1, A)
        # print(f"V_pinv.shape {V_pinv.shape}, actions.shape: {actions.shape}")
        coefficients = V_pinv @ actions  # (polynomial_order+1, A)

        # Evaluate polynomial at original time points
        # V shape: (T, polynomial_order+1)
        # coefficients shape: (polynomial_order+1, A)
        # smoothed shape: (T, A)
        smoothed = V @ coefficients  # (T, A)

        # Preserve boundary points if requested (NumPy style)
        if preserve_boundaries:
            smoothed[0, :] = actions[0, :]
            smoothed[-1, :] = actions[-1, :]

        return smoothed

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            t0 = time.perf_counter()
            self._last_results = self._policy.infer(obs)
            tf = time.perf_counter()
            print(f"inference take time {(tf - t0) * 1000 }ms")

            # Apply smoothing to action chunks if enabled
            if self._use_smoothing:
                self._last_results = tree.map_structure(
                    lambda x: self.intra_chunk_smoothing_vla_rail(
                        x, polynomial_order=self._polynomial_order,
                        preserve_boundaries=self._preserve_boundaries
                    ) if isinstance(x, np.ndarray) and x.ndim >= 2 else x,
                    self._last_results
                )

            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
