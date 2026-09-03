"""CFG sample routing, ported from RLinf ``compute_cfg_routing_masks``."""

from __future__ import annotations

import numpy as np


def compute_cfg_routing_masks(
    advantage: np.ndarray,
    *,
    positive_only_conditional: bool,
    unconditional_prob: float,
    random_values: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-sample CFG routing masks.

    Args:
        advantage: Boolean array, True = positive (high-advantage) sample.
        positive_only_conditional: If True, only positives may be conditional;
            negatives are always unconditional.
        unconditional_prob: Dropout probability for the conditional branch.
        random_values: Optional uniform samples in ``[0, 1)`` for tests.
        rng: RNG used when ``random_values`` is None.
    """
    advantage = np.asarray(advantage, dtype=np.bool_)
    batch_size = int(advantage.shape[0])
    if random_values is None:
        if rng is None:
            rng = np.random.default_rng()
        random_values = rng.random(batch_size)
    else:
        random_values = np.asarray(random_values, dtype=np.float64)

    positive_mask = advantage
    negative_mask = ~positive_mask
    if positive_only_conditional:
        positive_conditional_mask = positive_mask & (random_values > unconditional_prob)
        negative_conditional_mask = np.zeros_like(positive_mask)
    else:
        guidance_mask = random_values > unconditional_prob
        positive_conditional_mask = positive_mask & guidance_mask
        negative_conditional_mask = negative_mask & guidance_mask

    conditional_mask = positive_conditional_mask | negative_conditional_mask
    return {
        "positive_mask": positive_mask,
        "negative_mask": negative_mask,
        "conditional_mask": conditional_mask,
        "positive_conditional_mask": positive_conditional_mask,
        "positive_unconditional_mask": positive_mask & ~positive_conditional_mask,
        "negative_conditional_mask": negative_conditional_mask,
        "negative_unconditional_mask": negative_mask & ~negative_conditional_mask,
    }
