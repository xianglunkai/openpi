#!/usr/bin/env python

"""Unit tests for JAX RECAP helpers (no GPU / no policy weights)."""

import numpy as np
import pytest

from openpi.recap.advantages import (
    binarize_advantages,
    compute_episode_returns_and_rewards,
    compute_n_step_advantages,
    quantile_threshold,
)
from openpi.recap.routing import compute_cfg_routing_masks
from openpi.recap.tags import ACP_POSITIVE_TAG, build_acp_tagged_task
from openpi.recap.transforms import InjectAdvantagePrompt, LoadAdvantageLabel


def test_recap_config_inference_flag():
    from openpi.recap.config import RecapConfig

    assert RecapConfig().active_at_inference is False
    assert RecapConfig(enable=True).active_at_inference is True
    assert RecapConfig(enable=True, cfg_enable=False).active_at_inference is False


def test_build_acp_tagged_task():
    assert build_acp_tagged_task("fold the shirt", True) == f"fold the shirt\n{ACP_POSITIVE_TAG}"
    assert build_acp_tagged_task("", False) == "Advantage: negative"


def test_dynamic_return_success_and_failure():
    success, success_r = compute_episode_returns_and_rewards(3, is_success=True, gamma=1.0, failure_reward=-300)
    failure, failure_r = compute_episode_returns_and_rewards(3, is_success=False, gamma=1.0, failure_reward=-300)
    np.testing.assert_allclose(success_r, [-1, -1, 0])
    np.testing.assert_allclose(success, [-2, -1, 0])
    np.testing.assert_allclose(failure_r, [-1, -1, -300])
    np.testing.assert_allclose(failure, [-302, -301, -300])


def test_quantile_and_binarize():
    scores = np.arange(10, dtype=np.float32)
    threshold = quantile_threshold(scores, 0.3)
    labels, _ = binarize_advantages(scores, positive_fraction=0.3, per_task=False)
    assert threshold == pytest.approx(float(np.percentile(scores, 70)))
    assert int(labels.sum()) >= 3


def test_n_step_advantage_gamma_one():
    rewards = np.array([-1, -1, -1, 0], dtype=np.float32)
    returns = np.array([-3, -2, -1, 0], dtype=np.float32)
    values = returns.copy()
    episode = np.zeros(4, dtype=np.int32)
    frames = np.arange(4)
    adv = compute_n_step_advantages(
        rewards,
        values,
        episode,
        frames,
        n_step=2,
        gamma=1.0,
        returns=returns,
        return_min=float(returns.min()),
        return_max=float(returns.max()),
    )
    assert adv.shape == (4,)


def test_positive_only_conditional_routing():
    advantage = np.array([True, True, False, False])
    random_values = np.array([0.0, 0.99, 0.0, 0.99])
    routing = compute_cfg_routing_masks(
        advantage,
        positive_only_conditional=True,
        unconditional_prob=0.1,
        random_values=random_values,
    )
    assert routing["positive_conditional_mask"].tolist() == [False, True, False, False]
    assert routing["negative_conditional_mask"].tolist() == [False, False, False, False]
    assert routing["positive_unconditional_mask"].tolist() == [True, False, False, False]
    assert routing["negative_unconditional_mask"].tolist() == [False, False, True, True]


def test_inject_advantage_prompt_positive_only():
    transform = InjectAdvantagePrompt(positive_only_conditional=True, unconditional_prob=0.0, seed=0)
    positive = transform({"prompt": "sort screws", "advantage": True, "index": 0})
    negative = transform({"prompt": "sort screws", "advantage": False, "index": 1})
    assert ACP_POSITIVE_TAG in str(positive["prompt"])
    assert "Advantage:" not in str(negative["prompt"])
    assert "advantage" not in positive


def test_load_advantage_from_column():
    transform = LoadAdvantageLabel(path=None, column="advantage")
    out = transform({"advantage": 1, "episode_index": 0, "frame_index": 3})
    assert bool(out["advantage"]) is True


def test_sidecar_roundtrip(tmp_path):
    from openpi.recap.advantages import load_advantage_lookup, write_advantage_sidecar

    path = tmp_path / "advantages.parquet"
    write_advantage_sidecar(
        path,
        episode_index=np.array([0, 0, 1]),
        frame_index=np.array([0, 1, 0]),
        advantage=np.array([True, False, True]),
        advantage_continuous=np.array([0.2, -0.1, 0.4], dtype=np.float32),
    )
    lookup = load_advantage_lookup(path)
    loaded = LoadAdvantageLabel(path=str(path))
    out = loaded({"episode_index": 0, "frame_index": 1})
    assert lookup[(0, 1)] is False
    assert bool(out["advantage"]) is False


def test_sidecar_binarizes_continuous_only(tmp_path):
    from openpi.recap.advantages import load_advantage_lookup
    import pandas as pd

    path = tmp_path / "advantages_continuous.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0, 0, 1],
            "frame_index": [0, 1, 2, 0],
            "advantage_continuous": [0.0, 0.1, 0.9, 0.8],
        }
    ).to_parquet(path, index=False)
    lookup = load_advantage_lookup(path, positive_fraction=0.5)
    assert set(lookup) == {(0, 0), (0, 1), (0, 2), (1, 0)}
    assert lookup[(0, 2)] is True
    assert lookup[(1, 0)] is True
    assert lookup[(0, 0)] is False
