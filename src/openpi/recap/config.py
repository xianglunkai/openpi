"""RECAP hyperparameters shared by JAX training and serving."""

from __future__ import annotations

import dataclasses
from typing import Literal


@dataclasses.dataclass(frozen=True)
class RecapConfig:
    """Advantage-conditioned CFGRL settings for OpenPI JAX.

    Training (Evo-RL ACP tags + RLinf routing):
      - Positive frames get ``\\nAdvantage: positive`` appended to the prompt.
      - Negative frames stay untagged when ``positive_only_conditional`` is True
        (RLinf default); otherwise they get ``\\nAdvantage: negative``.
      - Positive tags are dropped to unconditional with ``unconditional_prob``.

    Inference (RLinf in-loop CFG, Evo-RL β=1 shortcut):
      - ``cfg_guidance_scale == 1`` samples the positive-conditioned policy.
      - ``cfg_guidance_scale > 1`` combines cond/uncond velocities each step:
        ``v = (1-w)*v_uncond + w*v_cond``.
    """

    enable: bool = False

    # Sidecar parquet written by scripts/recap/compute_advantages.py. If None,
    # labels are read from the LeRobot sample (``advantage_column``).
    advantage_path: str | None = None
    advantage_column: str = "advantage"
    # Used when the sidecar only has ``advantage_continuous`` (no bool ``advantage``).
    positive_fraction: float = 0.3

    # RLinf CFG training defaults.
    positive_only_conditional: bool = True
    unconditional_prob: float = 0.1
    dropout_seed: int = 0

    # Inference.
    cfg_enable: bool = True
    cfg_guidance_scale: float = 1.5
    guidance_type: Literal["positive", "no_guide"] = "positive"

    @property
    def active_at_inference(self) -> bool:
        """Whether serving should condition / CFG-sample from this config."""
        return self.enable and self.cfg_enable

    def __post_init__(self) -> None:
        if not 0.0 < self.positive_fraction < 1.0:
            raise ValueError(f"positive_fraction must be in (0, 1), got {self.positive_fraction}")
        if not 0.0 <= self.unconditional_prob <= 1.0:
            raise ValueError(f"unconditional_prob must be in [0, 1], got {self.unconditional_prob}")
        if self.cfg_guidance_scale < 0.0:
            raise ValueError(f"cfg_guidance_scale must be >= 0, got {self.cfg_guidance_scale}")
        if self.guidance_type not in ("positive", "no_guide"):
            raise ValueError(f"guidance_type must be 'positive' or 'no_guide', got {self.guidance_type}")
