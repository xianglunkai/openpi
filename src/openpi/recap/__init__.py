"""JAX RECAP (π*0.6): advantage-conditioned policy training and CFG inference.

Takes prompt-side advantage tags from Evo-RL and sidecar / positive-only CFG
routing from RLinf. The value critic itself stays optional: this package trains
and serves the JAX OpenPI policy given precomputed boolean labels.
"""

from openpi.recap.config import RecapConfig
from openpi.recap.tags import (
    ACP_NEGATIVE_TAG,
    ACP_POSITIVE_TAG,
    build_acp_tagged_task,
)

__all__ = [
    "ACP_NEGATIVE_TAG",
    "ACP_POSITIVE_TAG",
    "RecapConfig",
    "build_acp_tagged_task",
]
