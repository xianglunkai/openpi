from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.chunk_horizon import resolve_ref_chunk_horizon
from rlt_online_rl.config import EnvDriverConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.inference import ChunkFeatures
from rlt_online_rl.inference import PolicyPlan
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.rtc_runtime import RtcActionRuntime
from rlt_online_rl.rtc_runtime import resolve_rtc_execution_horizon
from rlt_online_rl.rtc_runtime import resolve_rtc_refill_threshold


def _make_plan(*, length: int, source: int) -> PolicyPlan:
    actions = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    refs = actions + 100.0
    features = ChunkFeatures(
        z_rl=np.zeros(4, dtype=np.float32),
        proprio=np.zeros(2, dtype=np.float32),
        ref_chunk=refs,
    )
    return PolicyPlan(
        action_chunk=actions,
        ref_chunk=refs,
        source=source,
        start_features=features,
        actor_param_version=3,
    )


def test_rtc_ref_horizon_stays_long_when_enabled() -> None:
    rl_config = RLTOnlineRLConfig(chunk_len=10)
    env_config = EnvDriverConfig(vla_chunk_exec_horizon=50, use_rtc=True)
    assert resolve_ref_chunk_horizon(env_config, rl_config, uses_rl_actor=True) == 50
    assert resolve_ref_chunk_horizon(env_config, rl_config, uses_rl_actor=False) == 50


def test_rtc_thresholds() -> None:
    # Evo collect defaults: s_vla=25, s_rl=10, refill=30.
    env_config = EnvDriverConfig()
    rl_config = RLTOnlineRLConfig(chunk_len=10)
    assert resolve_rtc_refill_threshold(env_config, rl_config) == 30
    assert resolve_rtc_execution_horizon(env_config, rl_config, uses_rl_actor=False) == 25
    assert resolve_rtc_execution_horizon(env_config, rl_config, uses_rl_actor=True) == 10
    assert env_config.rtc_inference_delay is None
    assert env_config.control_frequency_hz == 30.0


def test_rtc_refill_threshold_none_falls_back_to_chunk_len_minus_one() -> None:
    env_config = EnvDriverConfig(action_queue_size_to_get_new_actions=None)
    rl_config = RLTOnlineRLConfig(chunk_len=10)
    assert resolve_rtc_refill_threshold(env_config, rl_config) == 9
    # Explicit small threshold is kept (Evo only warns vs s+delay; does not raise).
    env_small = EnvDriverConfig(action_queue_size_to_get_new_actions=5)
    assert resolve_rtc_refill_threshold(env_small, rl_config) == 5


def test_prepare_request_hold_pads_short_leftover_to_s() -> None:
    runtime = RtcActionRuntime(
        fps=30.0,
        action_queue_size_to_get_new_actions=2,
        inference_warmup_steps=0,
    )
    plan = _make_plan(length=8, source=int(TransitionSource.BASE))
    runtime.merge_plan(
        plan,
        request_start_time=0.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=10,
    )
    _obs, prev, *_rest = runtime._prepare_request(
        {"state": np.zeros(2, dtype=np.float32)},
        allow_warmup_prev=False,
        execution_horizon=10,
    )
    assert prev is not None
    assert prev.shape[0] == 10
    np.testing.assert_array_equal(prev[:8], plan.action_chunk)
    np.testing.assert_array_equal(prev[8:], np.repeat(plan.action_chunk[-1:], 2, axis=0))


def test_prepare_request_empty_leftover_is_none() -> None:
    runtime = RtcActionRuntime(
        fps=30.0,
        action_queue_size_to_get_new_actions=2,
        inference_warmup_steps=0,
    )
    runtime.merge_plan(
        _make_plan(length=1, source=int(TransitionSource.BASE)),
        request_start_time=0.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=10,
    )
    assert runtime.pop() is not None
    _obs, prev, *_rest = runtime._prepare_request(
        {"state": np.zeros(2, dtype=np.float32)},
        allow_warmup_prev=False,
        execution_horizon=10,
    )
    assert prev is None


def test_rtc_runtime_phase_reset_and_pop_metadata() -> None:
    runtime = RtcActionRuntime(
        fps=30.0,
        action_queue_size_to_get_new_actions=2,
        inference_warmup_steps=0,
    )
    calls = {"n": 0}

    def planner(obs, local_step, **kwargs):
        del obs, local_step, kwargs
        calls["n"] += 1
        # Evo RL phase: short actor chunk only.
        return _make_plan(length=5, source=int(TransitionSource.RL))

    first = runtime.ensure_action(
        observation={"state": np.zeros(2, dtype=np.float32)},
        local_step=0,
        planner=planner,
        use_vla_guidance=False,
        execution_horizon=2,
        rl_refine_steps=None,
    )
    assert first.metadata.source == int(TransitionSource.RL)
    assert first.metadata.is_plan_anchor
    assert calls["n"] == 1
    assert runtime.qsize() == 4

    assert runtime.note_phase("online", True, "actor", True) is False
    assert runtime.note_phase("online", True, "actor", False) is True  # phase flip -> clear
    assert runtime.empty()
    assert runtime._phase_key == ("online", True, "actor", False)

    second = runtime.ensure_action(
        observation={"state": np.zeros(2, dtype=np.float32)},
        local_step=0,
        planner=planner,
        use_vla_guidance=False,
        execution_horizon=2,
        rl_refine_steps=None,
    )
    assert second.metadata.actor_param_version == 3
    assert calls["n"] == 2


def test_rtc_episode_reset_clears_phase_identity() -> None:
    runtime = RtcActionRuntime(fps=30.0, action_queue_size_to_get_new_actions=2)
    runtime.note_phase("online", True, "actor", True)
    runtime.merge_plan(
        _make_plan(length=3, source=int(TransitionSource.RL)),
        request_start_time=0.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=2,
    )
    assert not runtime.empty()
    runtime.reset()
    assert runtime.empty()
    assert runtime._phase_key is None
    # Same phase after episode reset must re-bind without requiring a key change.
    assert runtime.note_phase("online", True, "actor", True) is False


def test_rtc_policy_disable_clears_queue_without_dropping_phase() -> None:
    runtime = RtcActionRuntime(fps=30.0, action_queue_size_to_get_new_actions=2)
    runtime.note_phase("online", False, "base", False)
    runtime.merge_plan(
        _make_plan(length=3, source=int(TransitionSource.BASE)),
        request_start_time=0.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=2,
    )
    assert runtime.note_policy_enabled(False) is True
    assert runtime.empty()
    assert runtime._phase_key == ("online", False, "base", False)


def test_rtc_merge_clamps_metadata_with_delay() -> None:
    runtime = RtcActionRuntime(fps=30.0, action_queue_size_to_get_new_actions=100)
    plan_a = _make_plan(length=4, source=int(TransitionSource.BASE))
    runtime.merge_plan(
        plan_a,
        request_start_time=0.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=2,
    )
    assert runtime.pop() is not None  # consume one -> delay=1 on next merge

    plan_b = _make_plan(length=4, source=int(TransitionSource.RL))
    runtime.merge_plan(
        plan_b,
        request_start_time=0.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=2,
    )
    # delay skipped the first action of plan_b; remaining steps stay RL (Evo-style).
    result = runtime.pop()
    assert result is not None
    assert result.metadata.source == int(TransitionSource.RL)
    assert not result.metadata.is_plan_anchor
    sources = []
    while not runtime.empty():
        sources.append(runtime.pop().metadata.source)
    assert sources == [int(TransitionSource.RL), int(TransitionSource.RL)]


def test_rtc_fixed_guided_inference_delay() -> None:
    runtime = RtcActionRuntime(
        fps=30.0,
        action_queue_size_to_get_new_actions=2,
        guided_inference_delay=7,
        latency_skip_samples=0,
    )
    runtime._latency.add(1.0)  # would imply a large estimated delay
    assert runtime.guided_inference_delay_steps() == 7
    assert runtime.estimated_inference_delay_steps() >= 7


def test_rtc_skips_first_latency_samples_for_auto_d() -> None:
    runtime = RtcActionRuntime(
        fps=30.0,
        action_queue_size_to_get_new_actions=2,
        inference_warmup_steps=0,
        latency_skip_samples=2,
    )
    plan = _make_plan(length=4, source=int(TransitionSource.BASE))
    now = time.perf_counter()
    runtime.merge_plan(
        plan,
        request_start_time=now - 8.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=25,
    )
    assert runtime.estimated_inference_delay_steps() == 0
    assert runtime.guided_inference_delay_steps() == 0

    runtime.merge_plan(
        plan,
        request_start_time=now - 8.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=25,
    )
    assert runtime.estimated_inference_delay_steps() == 0

    runtime.merge_plan(
        plan,
        request_start_time=now - 0.25,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=25,
    )
    # 250ms at 30Hz → 8 steps; the 8s JIT samples must not set d.
    assert runtime.estimated_inference_delay_steps() == 8
    assert runtime.guided_inference_delay_steps() == 8


def test_rtc_latency_skip_resets_with_queue() -> None:
    runtime = RtcActionRuntime(
        fps=30.0,
        action_queue_size_to_get_new_actions=2,
        inference_warmup_steps=0,
        latency_skip_samples=1,
    )
    plan = _make_plan(length=3, source=int(TransitionSource.BASE))
    now = time.perf_counter()
    runtime.merge_plan(
        plan,
        request_start_time=now - 8.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=25,
    )
    runtime.merge_plan(
        plan,
        request_start_time=now - 0.25,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=25,
    )
    assert runtime.estimated_inference_delay_steps() == 8
    runtime.reset()
    assert runtime.estimated_inference_delay_steps() == 0
    runtime.merge_plan(
        plan,
        request_start_time=time.perf_counter() - 8.0,
        action_index_before_inference=0,
        generation=runtime._generation,
        execution_horizon=25,
    )
    # Cold-start skip quota is restored; 8s sample is dropped again.
    assert runtime.estimated_inference_delay_steps() == 0


def test_rtc_refill_uses_fixed_threshold() -> None:
    """Evo compares qsize to a fixed threshold (no runtime d+1 bump)."""
    runtime = RtcActionRuntime(
        fps=30.0,
        action_queue_size_to_get_new_actions=3,
        inference_warmup_steps=0,
    )
    calls = {"n": 0}

    def planner(obs, local_step, **kwargs):
        del obs, local_step, kwargs
        calls["n"] += 1
        return _make_plan(length=5, source=int(TransitionSource.RL))

    runtime.ensure_action(
        observation={"state": np.zeros(2, dtype=np.float32)},
        local_step=0,
        planner=planner,
        use_vla_guidance=False,
        execution_horizon=10,
        rl_refine_steps=None,
    )
    # After sync fill qsize=4 > threshold=3, so async worker must not start.
    assert runtime.qsize() == 4
    assert calls["n"] == 1
    assert runtime._worker is None or not runtime._worker.is_alive()
