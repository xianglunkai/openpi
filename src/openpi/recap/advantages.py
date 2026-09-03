"""Return / n-step advantage / quantile labeling (RLinf + Evo-RL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

logger = logging.getLogger("openpi")


def compute_episode_returns_and_rewards(
    episode_length: int,
    *,
    is_success: bool,
    gamma: float,
    failure_reward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """DynamicReturn used by RLinf RECAP.

    Per-step reward is -1; terminal reward is 0 on success and ``failure_reward``
    on failure. ``G_t = r_t + gamma * G_{t+1}``.
    """
    if episode_length <= 0:
        raise ValueError("episode_length must be > 0")
    rewards = np.full(episode_length, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if is_success else float(failure_reward)
    returns = np.zeros(episode_length, dtype=np.float32)
    returns[-1] = rewards[-1]
    for t in range(episode_length - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]
    return returns, rewards


def episode_boundaries(episode_indices: np.ndarray) -> list[tuple[int, int, int]]:
    """Return ``(episode_id, start, end)`` half-open spans."""
    episode_indices = np.asarray(episode_indices)
    if episode_indices.size == 0:
        return []
    change_mask = np.diff(episode_indices) != 0
    change_positions = np.where(change_mask)[0] + 1
    starts = np.concatenate([[0], change_positions])
    ends = np.concatenate([change_positions, [len(episode_indices)]])
    ep_ids = episode_indices[starts]
    return list(zip(ep_ids.tolist(), starts.tolist(), ends.tolist(), strict=True))


def quantile_threshold(scores, positive_fraction: float) -> float:
    """Threshold so the top ``positive_fraction`` of scores are positive."""
    if not 0.0 < positive_fraction < 1.0:
        raise ValueError(f"positive_fraction must be in (0, 1), got {positive_fraction}")
    return float(np.percentile(np.asarray(scores), (1.0 - float(positive_fraction)) * 100.0))


def apply_boolean_label(continuous, threshold: float, *, inclusive: bool = True):
    """RECAP uses ``>=``; STEAM uses ``>``."""
    if inclusive:
        return continuous >= threshold
    return continuous > threshold


def compute_n_step_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    episode_indices: np.ndarray,
    frame_indices: np.ndarray,
    *,
    n_step: int,
    gamma: float = 1.0,
    returns: np.ndarray | None = None,
    return_min: float | None = None,
    return_max: float | None = None,
    discount_next_value: bool = True,
) -> np.ndarray:
    """N-step advantage.

    RLinf (gamma==1): ``A = normalize(G_t - G_{t+N}) + gamma^N V_{t+N} - V_t``.
    Evo-RL (general): discounted n-step reward + bootstrap - V_t.
    """
    if n_step <= 0:
        raise ValueError("n_step must be > 0")
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    episode_indices = np.asarray(episode_indices)
    frame_indices = np.asarray(frame_indices)
    n = rewards.shape[0]
    advantages = np.zeros(n, dtype=np.float32)
    gamma_powers = np.array([gamma**i for i in range(n_step)], dtype=np.float64)

    def _normalize(x: float) -> float:
        if return_min is None or return_max is None:
            return float(x)
        ret_range = float(return_max) - float(return_min)
        if ret_range <= 0:
            return -0.5
        return (float(x) - float(return_min)) / ret_range - 1.0

    for i in range(n):
        ep_i = episode_indices[i]
        fi = int(frame_indices[i])
        steps = 0
        j = i
        while steps < n_step and j < n:
            if episode_indices[j] != ep_i or int(frame_indices[j]) != fi + steps:
                break
            steps += 1
            j += 1

        bootstrapped = (
            steps == n_step
            and j < n
            and episode_indices[j] == ep_i
            and int(frame_indices[j]) == fi + n_step
        )
        if abs(gamma - 1.0) < 1e-8 and returns is not None:
            if bootstrapped:
                reward_sum_raw = float(returns[i]) - float(returns[j])
            else:
                reward_sum_raw = float(returns[i])
        else:
            reward_slice = rewards[i : i + steps]
            reward_sum_raw = float(np.sum(gamma_powers[:steps] * reward_slice))

        v_next = float(values[j]) if bootstrapped else 0.0
        gamma_k = (gamma**steps) if discount_next_value else 1.0
        advantages[i] = _normalize(reward_sum_raw) + gamma_k * v_next - float(values[i])
    return advantages


def binarize_advantages(
    scores: np.ndarray,
    *,
    positive_fraction: float,
    task_indices: np.ndarray | None = None,
    interventions: np.ndarray | None = None,
    force_intervention_positive: bool = True,
    per_task: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Quantile-binarize scores; optionally force intervention frames positive."""
    scores = np.asarray(scores, dtype=np.float32)
    indicators = np.zeros(scores.shape[0], dtype=np.bool_)
    thresholds: dict[str, float] = {}
    if task_indices is not None and per_task:
        task_indices = np.asarray(task_indices)
        for task_idx in np.unique(task_indices):
            mask = task_indices == task_idx
            if not np.any(mask):
                continue
            threshold = quantile_threshold(scores[mask], positive_fraction)
            thresholds[str(int(task_idx))] = threshold
            indicators[mask] = apply_boolean_label(scores[mask], threshold)
    else:
        threshold = quantile_threshold(scores, positive_fraction)
        thresholds["all"] = threshold
        indicators = np.asarray(apply_boolean_label(scores, threshold), dtype=np.bool_)

    if force_intervention_positive and interventions is not None:
        indicators = indicators | (np.asarray(interventions).astype(np.float32) > 0.5)
    return indicators, thresholds


def load_advantage_lookup(
    path: str | Path,
    *,
    positive_fraction: float = 0.3,
) -> dict[tuple[int, int], bool]:
    """Load sidecar parquet/csv keyed by ``(episode_index, frame_index)``.

    Accepts a bool ``advantage`` column, or only ``advantage_continuous``
    (top ``positive_fraction`` of scores become True).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Advantage sidecar not found: {path}")
    if pd is None:
        raise ImportError("pandas is required to load RECAP advantage sidecars")
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        frame = pd.read_parquet(path)
    if "episode_index" not in frame.columns or "frame_index" not in frame.columns:
        raise ValueError(f"{path} must contain episode_index and frame_index")

    if "advantage" in frame.columns:
        labels = np.asarray(frame["advantage"]).astype(np.bool_)
    elif "advantage_continuous" in frame.columns:
        task_indices = frame["task_index"].to_numpy() if "task_index" in frame.columns else None
        labels, thresholds = binarize_advantages(
            frame["advantage_continuous"].to_numpy(),
            positive_fraction=positive_fraction,
            task_indices=task_indices,
            per_task=task_indices is not None,
            force_intervention_positive=False,
        )
        logger.info(
            "Binarized %s from advantage_continuous (positive_fraction=%.2f, positive_ratio=%.3f, thresholds=%s)",
            path,
            positive_fraction,
            float(np.mean(labels)),
            thresholds,
        )
    else:
        raise ValueError(
            f"{path} needs a boolean 'advantage' column or a float 'advantage_continuous' column"
        )

    lookup: dict[tuple[int, int], bool] = {}
    for ep, fr, adv in zip(
        frame["episode_index"].tolist(),
        frame["frame_index"].tolist(),
        labels.tolist(),
        strict=True,
    ):
        lookup[(int(ep), int(fr))] = bool(adv)
    return lookup


def write_advantage_sidecar(
    path: str | Path,
    *,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    advantage: np.ndarray,
    advantage_continuous: np.ndarray | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> Path:
    """Write RLinf-style ``meta/advantages_*.parquet`` without mutating the dataset."""
    if pd is None:
        raise ImportError("pandas is required to write RECAP advantage sidecars")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "episode_index": np.asarray(episode_index),
        "frame_index": np.asarray(frame_index),
        "advantage": np.asarray(advantage).astype(np.bool_),
    }
    if advantage_continuous is not None:
        payload["advantage_continuous"] = np.asarray(advantage_continuous, dtype=np.float32)
    if extra:
        payload.update(extra)
    frame = pd.DataFrame(payload)
    if path.suffix == ".csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path
