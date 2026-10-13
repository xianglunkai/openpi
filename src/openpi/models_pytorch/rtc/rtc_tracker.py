"""RTC Tracker for debugging and visualization."""

import logging
from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class RTCStepData:
    """Data for a single RTC denoising step."""

    step: int
    time: float
    x_t: Tensor
    v_t: Tensor
    x1_t: Tensor | None = None
    correction: Tensor | None = None
    error: Tensor | None = None
    weights: Tensor | None = None
    guidance_weight: float | None = None


class RTCTracker:
    """Tracks RTC parameters across denoising steps for debugging and visualization.

    This class collects data from each denoising step when debug mode is enabled,
    allowing for detailed analysis and visualization of the RTC process.
    """

    def __init__(self, enabled: bool = False):
        """Initialize RTCTracker.

        Args:
            enabled: Whether to track RTC data. If False, all tracking operations are no-ops.
        """
        self.enabled = enabled
        self.steps: List[RTCStepData] = []
        self._current_step = 0

    def reset(self):
        """Reset tracker for a new inference run."""
        self.steps = []
        self._current_step = 0

    def record_step(
        self,
        time: float,
        x_t: Tensor,
        v_t: Tensor,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        error: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | None = None,
    ):
        """Record data from a single denoising step.

        Args:
            time: Current diffusion timestep
            x_t: Current noisy actions
            v_t: Predicted velocity (possibly with RTC corrections)
            x1_t: Predicted clean actions (x_t - time * v_t)
            correction: RTC correction term
            error: Error between prev_chunk and x1_t
            weights: Prefix attention weights
            guidance_weight: Guidance weight applied to correction
        """
        if not self.enabled:
            return

        # Clone tensors to prevent memory issues and detach from computation graph
        step_data = RTCStepData(
            step=self._current_step,
            time=time,
            x_t=x_t.detach().cpu().clone() if x_t is not None else None,
            v_t=v_t.detach().cpu().clone() if v_t is not None else None,
            x1_t=x1_t.detach().cpu().clone() if x1_t is not None else None,
            correction=correction.detach().cpu().clone() if correction is not None else None,
            error=error.detach().cpu().clone() if error is not None else None,
            weights=weights.detach().cpu().clone() if weights is not None else None,
            guidance_weight=guidance_weight,
        )

        self.steps.append(step_data)
        self._current_step += 1

    def get_last_step_data(self) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Get correction, x1_t, and error from the last recorded step.

        Returns:
            Tuple of (correction, x1_t, error) from the last step. Returns (None, None, None) if no steps recorded.
        """
        if not self.enabled or len(self.steps) == 0:
            return None, None, None

        last_step = self.steps[-1]
        # Return tensors - they're already detached and on CPU from record_step
        correction = last_step.correction
        x1_t = last_step.x1_t
        error = last_step.error

        return correction, x1_t, error

    def get_tracking_history(self) -> dict:
        """Get tracking history in a format compatible with visualization code.

        Returns:
            Dictionary with lists of tracked parameters, compatible with existing visualization code.
        """
        if not self.enabled or len(self.steps) == 0:
            return None

        history = {
            'x_t': [],
            'v_t': [],
            'time': [],
            'x_1': [],
            'error': [],
            'weights': [],
            'guidance_weight': [],
            'pinv_correction': []
        }

        for step_data in self.steps:
            history['x_t'].append(step_data.x_t.numpy() if step_data.x_t is not None else None)
            history['v_t'].append(step_data.v_t.numpy() if step_data.v_t is not None else None)
            history['time'].append(step_data.time)
            history['x_1'].append(step_data.x1_t.numpy() if step_data.x1_t is not None else None)
            history['error'].append(step_data.error.numpy() if step_data.error is not None else None)
            history['weights'].append(step_data.weights.numpy() if step_data.weights is not None else None)
            history['guidance_weight'].append(step_data.guidance_weight)
            history['pinv_correction'].append(step_data.correction.numpy() if step_data.correction is not None else None)

        return history

    def __len__(self) -> int:
        """Return number of recorded steps."""
        return len(self.steps)

    def __repr__(self) -> str:
        """String representation."""
        status = "enabled" if self.enabled else "disabled"
        return f"RTCTracker({status}, {len(self.steps)} steps recorded)"