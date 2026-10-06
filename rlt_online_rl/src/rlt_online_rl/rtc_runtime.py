"""Client-side Real-Time Chunking (RTC) runtime for online RL deployment.

Mirrors openpi ``RuntimeRTC`` / ChunkACPolicy RTC:
  * one shared ``ActionQueue`` for VLA and RL phases
  * async refill when the queue is low
  * latency-aware merge via ``real_delay``
  * hard reset on phase switch / interrupt / episode boundary
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
import logging
import math
import threading
import time
from typing import Any

import numpy as np
from openpi_client.action_queue import ActionQueue
from openpi_client.action_queue import pad_rtc_prev_actions_hold_last
from openpi_client.latency_tracker import LatencyTracker

from rlt_online_rl.inference import ChunkFeatures
from rlt_online_rl.inference import PolicyPlan
from rlt_online_rl.replay import TransitionSource

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RtcStepMetadata:
    """Per-step metadata aligned with one queued action."""

    ref_action: np.ndarray
    source: int
    actor_param_version: int
    start_features: ChunkFeatures
    is_plan_anchor: bool


@dataclass(slots=True)
class RtcPopResult:
    action: np.ndarray
    metadata: RtcStepMetadata


class RtcActionRuntime:
    """Thread-safe overlapping action queue for deploy-time RTC."""

    def __init__(
        self,
        *,
        fps: float,
        action_queue_size_to_get_new_actions: int,
        guided_inference_delay: int | None = None,
        inference_warmup_steps: int = 4,
        latency_skip_samples: int = 4,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if action_queue_size_to_get_new_actions < 0:
            raise ValueError(
                f"action_queue_size_to_get_new_actions must be >= 0, got {action_queue_size_to_get_new_actions}"
            )
        if guided_inference_delay is not None and int(guided_inference_delay) < 0:
            raise ValueError(f"guided_inference_delay must be >= 0, got {guided_inference_delay}")
        if inference_warmup_steps < 0:
            raise ValueError(f"inference_warmup_steps must be >= 0, got {inference_warmup_steps}")
        if latency_skip_samples < 0:
            raise ValueError(f"latency_skip_samples must be >= 0, got {latency_skip_samples}")
        self._fps = float(fps)
        self._refill_threshold = int(action_queue_size_to_get_new_actions)
        self._guided_inference_delay = (
            None if guided_inference_delay is None else int(guided_inference_delay)
        )
        self._inference_warmup_total = int(inference_warmup_steps)
        self._inference_warmup_remaining = int(inference_warmup_steps)
        self._latency_skip_total = int(latency_skip_samples)
        self._latency_skip_remaining = int(latency_skip_samples)
        self._warmup_prev_actions: np.ndarray | None = None
        self._queue = ActionQueue(enabled=True)
        self._latency = LatencyTracker()
        self._lock = threading.Lock()
        self._generation = 0
        self._phase_key: tuple[Any, ...] | None = None
        self._worker: threading.Thread | None = None
        self._worker_error: Exception | None = None
        self._step_metadata: deque[RtcStepMetadata] = deque()
        self._policy_was_enabled = True
        self._inference_count = 0

    @property
    def refill_threshold(self) -> int:
        return self._refill_threshold

    def set_refill_threshold(self, value: int) -> None:
        self._refill_threshold = max(0, int(value))

    def note_phase(self, *phase_key: Any) -> bool:
        """Reset the queue when the deploy phase identity changes.

        Returns True if the phase identity changed (queue was cleared), matching
        Evo-RLT ``set_rl_mode`` / ``set_vla_mode`` / ``trigger_critical_phase``.
        """
        key = tuple(phase_key)
        changed = self._phase_key is not None and key != self._phase_key
        if changed:
            logger.info("[RLT RTC] phase change %s -> %s; resetting action queue", self._phase_key, key)
            self._invalidate_queue()
        self._phase_key = key
        return changed

    def note_policy_enabled(self, enabled: bool) -> bool:
        """Human interrupt / resume: clear leftovers so guidance cannot cross the gap.

        Returns True if policy was disabled and the queue was cleared (Evo ``interrupt_chunk``).
        """
        changed = bool(self._policy_was_enabled and not enabled)
        if changed:
            logger.info("[RLT RTC] policy disabled; resetting action queue")
            self._invalidate_queue()
        self._policy_was_enabled = bool(enabled)
        return changed

    def reset(self) -> None:
        """Episode boundary hard clear (Evo ``ChunkACPolicy.reset`` / ``_reset_rtc_runtime``)."""
        self._invalidate_queue()
        self._phase_key = None
        self._policy_was_enabled = True
        self._inference_warmup_remaining = self._inference_warmup_total
        self._warmup_prev_actions = None
        self._inference_count = 0

    def _invalidate_queue(self) -> None:
        """Drop queued actions / metadata and cancel in-flight refill workers."""
        with self._lock:
            self._generation += 1
            self._queue = ActionQueue(enabled=True)
            self._latency.reset()
            self._latency_skip_remaining = self._latency_skip_total
            self._step_metadata.clear()
            self._worker_error = None
        self._join_finished_worker()

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    def estimated_inference_delay_steps(self) -> int:
        """Latency-derived delay after skipping the first cold-start samples."""
        if len(self._latency) == 0:
            return 0
        time_per_step = 1.0 / self._fps
        return int(math.ceil(self._latency.max() / time_per_step))

    def guided_inference_delay_steps(self) -> int:
        """``d`` sent to Machine A ``guided_inference`` (fixed when configured)."""
        if self._guided_inference_delay is not None:
            return int(self._guided_inference_delay)
        return self.estimated_inference_delay_steps()

    def get_left_over(self) -> np.ndarray | None:
        leftover = self._queue.get_left_over()
        if leftover is None:
            return None
        return np.asarray(leftover, dtype=np.float32).copy()

    def pop(self) -> RtcPopResult | None:
        with self._lock:
            action = self._queue.get()
            if action is None:
                return None
            if not self._step_metadata:
                raise RuntimeError("RTC metadata queue empty while action queue had an action")
            metadata = self._step_metadata.popleft()
        return RtcPopResult(action=np.asarray(action, dtype=np.float32), metadata=metadata)

    def merge_plan(
        self,
        plan: PolicyPlan,
        *,
        request_start_time: float,
        action_index_before_inference: int,
        generation: int,
        execution_horizon: int | None = None,
        rl_refine_steps: int | None = None,
    ) -> None:
        if generation != self._generation:
            return
        actions = np.asarray(plan.action_chunk, dtype=np.float32)
        refs = np.asarray(plan.ref_chunk, dtype=np.float32)
        if actions.ndim != 2 or refs.ndim != 2:
            raise ValueError(f"RTC plan chunks must be rank-2, got actions={actions.shape} refs={refs.shape}")
        if actions.shape[0] != refs.shape[0]:
            raise ValueError(
                f"RTC action/ref length mismatch: actions={actions.shape[0]} refs={refs.shape[0]}"
            )

        metadata = _plan_to_step_metadata(
            plan,
            rl_refine_steps=plan.rl_refine_steps if plan.rl_refine_steps is not None else rl_refine_steps,
        )
        new_latency = time.perf_counter() - request_start_time
        time_per_step = 1.0 / self._fps
        estimated_delay = int(math.ceil(new_latency / time_per_step))
        # Drop the first few samples so JIT / cold-start (~seconds) cannot set RTC ``d``.
        skipped_latency_sample = self._latency_skip_remaining > 0
        if skipped_latency_sample:
            self._latency_skip_remaining -= 1
            logger.info(
                "[RLT RTC] skip latency sample remaining=%d latency=%.1fms estimated_delay=%d "
                "(excluded from auto d)",
                self._latency_skip_remaining,
                new_latency * 1000.0,
                estimated_delay,
            )
        else:
            self._latency.add(new_latency)

        with self._lock:
            if generation != self._generation:
                return
            real_delay = max(0, self._queue.get_action_index() - action_index_before_inference)
            # Mirror ActionQueue._replace_actions_queue clamping so metadata stays aligned.
            clamped_delay = max(0, min(real_delay, actions.shape[0], refs.shape[0]))
            queue_before = self._queue.qsize()
            self._queue.merge(
                actions,
                actions,
                real_delay,
                action_index_before_inference,
            )
            self._step_metadata = deque(metadata[clamped_delay:])
            self._inference_count += 1

        ls = self._queue.qsize()
        print(
            f"[RltRtc] Inference {self._inference_count}: "
            f"time={new_latency * 1000.0:.1f}ms, "
            f"delay={estimated_delay} steps, "
            f"real_delay={real_delay} steps, "
            f"start queue_size={queue_before}, "
            f"end queue_size={ls}",
            flush=True,
        )
        logger.info(
            "[RLT RTC] latency=%.1fms estimated_delay=%d real_delay=%d queue_before=%d queued=%d",
            new_latency * 1000.0,
            estimated_delay,
            real_delay,
            queue_before,
            ls,
        )
        # Match Evo-RLT ChunkACPolicy: warn when refill fires too late vs s + delay.
        horizon = int(execution_horizon) if execution_horizon is not None else 0
        delay_for_warn = estimated_delay if not skipped_latency_sample else self.estimated_inference_delay_steps()
        min_refill_threshold = horizon + delay_for_warn
        if horizon > 0 and self._refill_threshold < min_refill_threshold:
            logger.warning(
                "[RLT RTC] action_queue_size_to_get_new_actions=%d is smaller than "
                "execution_horizon + delay (%d + %d). The queue may run dry under load.",
                self._refill_threshold,
                horizon,
                estimated_delay,
            )

    def ensure_action(
        self,
        *,
        observation: dict[str, Any],
        local_step: int,
        planner: Callable[..., PolicyPlan],
        use_vla_guidance: bool,
        execution_horizon: int,
        rl_refine_steps: int | None,
        observation_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> RtcPopResult:
        """Pop one action; sync-refill if empty, else optionally kick an async worker.

        ``observation_fn`` (if set) is called at inference start so async refill uses a
        fresh obs like openpi ``RuntimeRTC`` (not the stale trigger-time clone).
        """
        self._join_finished_worker()
        self._raise_worker_error()

        computed_sync = False
        if self.empty():
            self._wait_for_worker()
            self._raise_worker_error()
        if self.empty():
            self._run_startup_warmup(
                observation=observation,
                local_step=local_step,
                planner=planner,
                use_vla_guidance=use_vla_guidance,
                execution_horizon=execution_horizon,
                rl_refine_steps=rl_refine_steps,
                observation_fn=observation_fn,
            )
            self._predict_and_merge_sync(
                observation=observation,
                local_step=local_step,
                planner=planner,
                use_vla_guidance=use_vla_guidance,
                execution_horizon=execution_horizon,
                rl_refine_steps=rl_refine_steps,
                observation_fn=observation_fn,
            )
            computed_sync = True

        result = self.pop()
        if result is None:
            self._predict_and_merge_sync(
                observation=observation,
                local_step=local_step,
                planner=planner,
                use_vla_guidance=use_vla_guidance,
                execution_horizon=execution_horizon,
                rl_refine_steps=rl_refine_steps,
                observation_fn=observation_fn,
            )
            result = self.pop()
        if result is None:
            raise RuntimeError("RTC action queue empty after refill")

        if not computed_sync:
            self._maybe_start_worker(
                observation=observation,
                local_step=local_step,
                planner=planner,
                use_vla_guidance=use_vla_guidance,
                execution_horizon=execution_horizon,
                rl_refine_steps=rl_refine_steps,
                observation_fn=observation_fn,
            )
        return result

    def _resolve_observation(
        self,
        observation: dict[str, Any],
        observation_fn: Callable[[], dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if observation_fn is None:
            return _clone_obs(observation)
        return _clone_obs(observation_fn())

    def _prepare_request(
        self,
        observation: dict[str, Any],
        observation_fn: Callable[[], dict[str, Any]] | None = None,
        *,
        allow_warmup_prev: bool = True,
        execution_horizon: int,
    ) -> tuple[dict[str, Any], np.ndarray | None, int, int, float, int]:
        with self._lock:
            prev_actions = self._queue.get_left_over()
            if prev_actions is not None:
                prev_actions = np.asarray(prev_actions, dtype=np.float32).copy()
            elif allow_warmup_prev and self._warmup_prev_actions is not None:
                n = max(1, int(self._refill_threshold))
                cached = np.asarray(self._warmup_prev_actions, dtype=np.float32)
                prev_actions = cached[: min(n, cached.shape[0])].copy()
            if prev_actions is not None and int(np.asarray(prev_actions).shape[0]) == 0:
                prev_actions = None
            if prev_actions is not None:
                # Hold-pad to ``s`` so RTC weights on ``[T, s)`` lock the last command
                # instead of shrinking ``s`` or leaking zeros into the prefix.
                prev_actions = pad_rtc_prev_actions_hold_last(prev_actions, execution_horizon)
            action_index_before = self._queue.get_action_index()
            generation = self._generation
        inference_delay = self.guided_inference_delay_steps()
        return (
            self._resolve_observation(observation, observation_fn),
            prev_actions,
            inference_delay,
            action_index_before,
            time.perf_counter(),
            generation,
        )

    def _run_startup_warmup(
        self,
        *,
        observation: dict[str, Any],
        local_step: int,
        planner: Callable[..., PolicyPlan],
        use_vla_guidance: bool,
        execution_horizon: int,
        rl_refine_steps: int | None,
        observation_fn: Callable[[], dict[str, Any]] | None,
    ) -> None:
        """Discard first N inferences without executing (openpi RuntimeRTC warmup)."""
        del rl_refine_steps
        while self._inference_warmup_remaining > 0:
            (
                obs,
                prev_actions,
                inference_delay,
                _action_index_before,
                _request_start,
                generation,
            ) = self._prepare_request(
                observation,
                observation_fn,
                allow_warmup_prev=True,
                execution_horizon=execution_horizon,
            )
            if generation != self._generation:
                return
            guidance_s = max(1, int(execution_horizon))
            plan = planner(
                obs,
                local_step,
                prev_actions=prev_actions if use_vla_guidance else None,
                use_rtc=bool(use_vla_guidance),
                inference_delay=inference_delay,
                rtc_execution_horizon=guidance_s,
            )
            actions = np.asarray(plan.action_chunk, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[np.newaxis, ...]
            self._warmup_prev_actions = actions
            self._inference_warmup_remaining -= 1
            logger.info(
                "[RLT RTC] startup warmup discard remaining=%d chunk_len=%d",
                self._inference_warmup_remaining,
                int(actions.shape[0]),
            )
        self._warmup_prev_actions = None

    def _predict_and_merge_sync(
        self,
        *,
        observation: dict[str, Any],
        local_step: int,
        planner: Callable[..., PolicyPlan],
        use_vla_guidance: bool,
        execution_horizon: int,
        rl_refine_steps: int | None,
        observation_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        (
            obs,
            prev_actions,
            inference_delay,
            action_index_before,
            request_start,
            generation,
        ) = self._prepare_request(
            observation,
            observation_fn,
            allow_warmup_prev=False,
            execution_horizon=execution_horizon,
        )
        guidance_s = max(1, int(execution_horizon))
        plan = planner(
            obs,
            local_step,
            prev_actions=prev_actions if use_vla_guidance else None,
            use_rtc=bool(use_vla_guidance),
            inference_delay=inference_delay,
            rtc_execution_horizon=guidance_s,
        )
        self.merge_plan(
            plan,
            request_start_time=request_start,
            action_index_before_inference=action_index_before,
            generation=generation,
            execution_horizon=execution_horizon,
            rl_refine_steps=rl_refine_steps,
        )

    def _maybe_start_worker(
        self,
        *,
        observation: dict[str, Any],
        local_step: int,
        planner: Callable[..., PolicyPlan],
        use_vla_guidance: bool,
        execution_horizon: int,
        rl_refine_steps: int | None,
        observation_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._join_finished_worker()
        self._raise_worker_error()
        if self._queue.qsize() > self._refill_threshold:
            return
        # Snapshot leftover/index now; refresh images/state inside the worker.
        request = self._prepare_request(
            observation,
            observation_fn=None,
            allow_warmup_prev=False,
            execution_horizon=execution_horizon,
        )
        self._worker = threading.Thread(
            target=self._run_worker,
            args=(
                request,
                local_step,
                planner,
                use_vla_guidance,
                execution_horizon,
                rl_refine_steps,
                observation_fn,
            ),
            daemon=True,
            name="RltRtcWorker",
        )
        self._worker.start()

    def _run_worker(
        self,
        request: tuple[dict[str, Any], np.ndarray | None, int, int, float, int],
        local_step: int,
        planner: Callable[..., PolicyPlan],
        use_vla_guidance: bool,
        execution_horizon: int,
        rl_refine_steps: int | None,
        observation_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        obs, prev_actions, inference_delay, action_index_before, request_start, generation = request
        try:
            if generation != self._generation:
                return
            if observation_fn is not None:
                obs = _clone_obs(observation_fn())
                request_start = time.perf_counter()

            guidance_s = max(1, int(execution_horizon))

            plan = planner(
                obs,
                local_step,
                prev_actions=prev_actions if use_vla_guidance else None,
                use_rtc=bool(use_vla_guidance),
                inference_delay=inference_delay,
                rtc_execution_horizon=guidance_s,
            )

            self.merge_plan(
                plan,
                request_start_time=request_start,
                action_index_before_inference=action_index_before,
                generation=generation,
                execution_horizon=execution_horizon,
                rl_refine_steps=rl_refine_steps,
            )
        except Exception as error:  # noqa: BLE001 - surface to control thread
            with self._lock:
                if generation == self._generation:
                    self._worker_error = error

    def _join_finished_worker(self) -> None:
        if self._worker is None or self._worker.is_alive():
            return
        self._worker.join()
        self._worker = None

    def _wait_for_worker(self) -> None:
        if self._worker is None:
            return
        self._worker.join()
        self._worker = None
        self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self._worker_error is None:
            return
        error = self._worker_error
        self._worker_error = None
        raise error


def _plan_to_step_metadata(
    plan: PolicyPlan,
    *,
    rl_refine_steps: int | None,
) -> list[RtcStepMetadata]:
    actions = np.asarray(plan.action_chunk, dtype=np.float32)
    refs = np.asarray(plan.ref_chunk, dtype=np.float32)
    refine_n = actions.shape[0] if rl_refine_steps is None else max(0, min(int(rl_refine_steps), actions.shape[0]))
    out: list[RtcStepMetadata] = []
    for idx in range(actions.shape[0]):
        if idx < refine_n:
            source = int(plan.source)
        else:
            source = int(TransitionSource.BASE)
        out.append(
            RtcStepMetadata(
                ref_action=refs[idx].copy(),
                source=source,
                actor_param_version=int(plan.actor_param_version),
                start_features=plan.start_features,
                is_plan_anchor=(idx == 0),
            )
        )
    return out


def _clone_obs(observation: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in observation.items():
        if isinstance(value, np.ndarray):
            cloned[key] = value.copy()
        elif isinstance(value, dict):
            cloned[key] = {
                nested_key: (nested_val.copy() if isinstance(nested_val, np.ndarray) else nested_val)
                for nested_key, nested_val in value.items()
            }
        else:
            cloned[key] = value
    return cloned


def resolve_rtc_execution_horizon(env_config: Any, rl_config: Any | None = None, *, uses_rl_actor: bool = False) -> int:
    """Preferred guidance ``s`` for the current phase."""
    if uses_rl_actor:
        if env_config.rtc_execution_horizon_rl is not None:
            return max(1, int(env_config.rtc_execution_horizon_rl))
        if env_config.rl_chunk_exec_horizon is not None:
            return max(1, int(env_config.rl_chunk_exec_horizon))
        if rl_config is not None:
            return max(1, int(rl_config.chunk_len))
        return 10
    return max(1, int(env_config.rtc_execution_horizon_vla))


def resolve_rtc_refill_threshold(env_config: Any, rl_config: Any | None = None) -> int:
    """Shared refill threshold (Evo-RLT ``action_queue_size_to_get_new_actions``).

    One value for VLA and RL. Collect default is 30; ``None`` → ``max(1, chunk_len - 1)``
    (Evo ``configure_rtc`` when the CLI flag is omitted). Too-small vs ``s + delay``
    is warned at merge time, not auto-raised here.
    """
    configured = getattr(env_config, "action_queue_size_to_get_new_actions", None)
    if configured is not None:
        return max(0, int(configured))
    chunk_len = 10
    if rl_config is not None and getattr(rl_config, "chunk_len", None) is not None:
        chunk_len = int(rl_config.chunk_len)
    elif getattr(env_config, "rl_chunk_exec_horizon", None) is not None:
        chunk_len = int(env_config.rl_chunk_exec_horizon)
    return max(1, chunk_len - 1)
