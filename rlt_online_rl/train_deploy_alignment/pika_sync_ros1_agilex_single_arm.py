#!/usr/bin/env python3
"""ROS1 AgileX single-arm rollout adapter for RLT online RL.

This file keeps the same online-RL orchestration semantics as
`train_deploy_alignment/pika_sync_ros.py`, but swaps the robot/ROS integration
to ROS1 (`rospy`) and AgileX Aloha-style topics via
`examples/mobile_aloha_AgileX/robot_utils.py`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import dataclasses
import importlib
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from examples.mobile_aloha_AgileX import robot_utils
from openpi_client import image_tools

from rlt_online_rl.chunk_horizon import ChunkHorizonEnvMixin
from rlt_online_rl.config import OnlineRLSystemConfig
from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.inference import ActorClient
from rlt_online_rl.inference import ActorResponse
from rlt_online_rl.inference import ChunkFeatures
from rlt_online_rl.inference import EnvDriver
from rlt_online_rl.inference import MachineAFeatureClient
from rlt_online_rl.inference import PolicyPlan
from rlt_online_rl.replay import NullReplayClient
from rlt_online_rl.replay import ReplayClient
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.rtc_runtime import RtcActionRuntime
from rlt_online_rl.rtc_runtime import resolve_rtc_execution_horizon
from rlt_online_rl.rtc_runtime import resolve_rtc_refill_threshold
from rlt_online_rl.runtime_logging import metrics_path_for
from rlt_online_rl.runtime_logging import setup_process_logging

RewardFn = Callable[[dict[str, Any], np.ndarray, dict[str, Any], dict[str, Any]], np.ndarray | list[float] | float]
SuccessFn = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool | int]
DoneFn = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool | int]
ActionFilterFn = Callable[[np.ndarray], np.ndarray]

logger = logging.getLogger("pika_sync_ros1_agilex_single_arm")

TELEOP_STATUS_SERVICE = "/teleop_status"
SHUTDOWN_ROLLOUT_SERVICE = "/shutdown_rollout"
REQUEST_NEXT_EPISODE_SERVICE = "/request_next_episode"
RECORD_SUCCESS_SERVICE = "/record_success"
RECORD_FAILURE_SERVICE = "/record_failure"
RECORD_DONE_SERVICE = "/record_done"
ENTER_CRITICAL_PHASE_SERVICE = "/enter_critical_phase"
TOGGLE_CRITICAL_PHASE_SERVICE = "/toggle_critical_phase"
SET_CRITICAL_POLICY_ACTOR_SERVICE = "/select_critical_policy_actor"
SET_CRITICAL_POLICY_BASE_SERVICE = "/select_critical_policy_base"

SIGNAL_NEXT_EPISODE_REQUESTED = "next_episode_requested"
SIGNAL_MANUAL_SUCCESS_PENDING = "manual_success_pending"
SIGNAL_MANUAL_FAILURE_PENDING = "manual_failure_pending"
SIGNAL_MANUAL_DONE_PENDING = "manual_done_pending"
SIGNAL_CRITICAL_STARTED = "critical_started"
SIGNAL_SELECTED_CRITICAL_POLICY = "selected_critical_policy"
SIGNAL_EPISODE_CRITICAL_POLICY = "episode_critical_policy"
SIGNAL_TASK_MODE = "task_mode"
SIGNAL_STOP_REQUESTED = "stop_requested"


@dataclasses.dataclass(slots=True)
class RolloutRuntimeContext:
    system: OnlineRLSystemConfig
    task_state: "TaskState"
    intervention_state: "HumanInterventionState"
    signal_values: dict[str, Any] = dataclasses.field(default_factory=dict)
    _lock: threading.RLock = dataclasses.field(default_factory=threading.RLock, init=False, repr=False)
    _condition: threading.Condition = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_condition", threading.Condition(self._lock))
        self.reset_episode_state()

    def set_signal(self, name: str, value: Any) -> None:
        with self._condition:
            self.signal_values[name] = value
            self._condition.notify_all()

    def get_signal(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self.signal_values.get(name, default)

    def snapshot_signals(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.signal_values)

    def clear_signals(self, *names: str) -> None:
        with self._condition:
            for name in names:
                self.signal_values[name] = False
            self._condition.notify_all()

    def task_mode(self) -> str:
        return str(self.get_signal(SIGNAL_TASK_MODE, self.system.env_driver.task_mode))

    def in_critical_phase(self) -> bool:
        return bool(self.get_signal(SIGNAL_CRITICAL_STARTED, False))

    def selected_critical_policy_mode(self) -> str:
        return str(self.get_signal(SIGNAL_SELECTED_CRITICAL_POLICY, "actor"))

    def episode_critical_policy_mode(self) -> str:
        return str(self.get_signal(SIGNAL_EPISODE_CRITICAL_POLICY, self.selected_critical_policy_mode()))

    def reset_episode_state(self) -> None:
        task_mode = self.system.env_driver.task_mode
        with self._condition:
            selected_policy = str(self.signal_values.get(SIGNAL_SELECTED_CRITICAL_POLICY, "actor"))
            self.signal_values[SIGNAL_TASK_MODE] = task_mode
            self.signal_values[SIGNAL_CRITICAL_STARTED] = task_mode == "critical_phase"
            self.signal_values[SIGNAL_SELECTED_CRITICAL_POLICY] = selected_policy
            self.signal_values[SIGNAL_EPISODE_CRITICAL_POLICY] = selected_policy
            self.signal_values[SIGNAL_NEXT_EPISODE_REQUESTED] = False
            self.signal_values[SIGNAL_MANUAL_SUCCESS_PENDING] = False
            self.signal_values[SIGNAL_MANUAL_FAILURE_PENDING] = False
            self.signal_values[SIGNAL_MANUAL_DONE_PENDING] = False
            self.signal_values[SIGNAL_STOP_REQUESTED] = bool(self.signal_values.get(SIGNAL_STOP_REQUESTED, False))
            self._condition.notify_all()

    def set_selected_critical_policy_mode(self, mode: str) -> None:
        selected = "actor" if str(mode) != "base" else "base"
        self.set_signal(SIGNAL_SELECTED_CRITICAL_POLICY, selected)

    def lock_episode_critical_policy_mode(self) -> str:
        mode = self.selected_critical_policy_mode()
        self.set_signal(SIGNAL_EPISODE_CRITICAL_POLICY, mode)
        return mode

    def request_next_episode(self) -> None:
        self.set_signal(SIGNAL_NEXT_EPISODE_REQUESTED, True)

    def request_stop(self) -> None:
        with self._condition:
            self.signal_values[SIGNAL_STOP_REQUESTED] = True
            self._condition.notify_all()

    def stop_requested(self) -> bool:
        with self._lock:
            return bool(self.signal_values.get(SIGNAL_STOP_REQUESTED, False))

    def wait_for_next_episode_request(self) -> None:
        with self._condition:
            while (
                not rospy.is_shutdown()
                and not bool(self.signal_values.get(SIGNAL_NEXT_EPISODE_REQUESTED, False))
                and not bool(self.signal_values.get(SIGNAL_STOP_REQUESTED, False))
            ):
                self._condition.wait(timeout=0.1)
            self.signal_values[SIGNAL_NEXT_EPISODE_REQUESTED] = False

    def mark_manual_success(self) -> None:
        self.intervention_state.enter_episode_reset()
        with self._condition:
            self.signal_values[SIGNAL_MANUAL_SUCCESS_PENDING] = True
            self.signal_values[SIGNAL_MANUAL_FAILURE_PENDING] = False
            self.signal_values[SIGNAL_MANUAL_DONE_PENDING] = True
            self._condition.notify_all()

    def mark_manual_failure(self) -> None:
        self.intervention_state.enter_episode_reset()
        with self._condition:
            self.signal_values[SIGNAL_MANUAL_FAILURE_PENDING] = True
            self.signal_values[SIGNAL_MANUAL_SUCCESS_PENDING] = False
            self.signal_values[SIGNAL_MANUAL_DONE_PENDING] = True
            self._condition.notify_all()

    def mark_manual_done(self) -> None:
        self.intervention_state.enter_episode_reset()
        self.set_signal(SIGNAL_MANUAL_DONE_PENDING, True)

    def enter_critical_phase(self) -> bool:
        with self._condition:
            already_active = bool(self.signal_values.get(SIGNAL_CRITICAL_STARTED, False))
            if not already_active:
                self.signal_values[SIGNAL_CRITICAL_STARTED] = True
                self._condition.notify_all()
            return not already_active

    def toggle_critical_phase(self) -> bool:
        with self._condition:
            active = not bool(self.signal_values.get(SIGNAL_CRITICAL_STARTED, False))
            self.signal_values[SIGNAL_CRITICAL_STARTED] = active
            self._condition.notify_all()
            return active


class RolloutPhaseController:
    def __init__(
        self,
        replay_client: ReplayClient,
        warmup_min_size: int,
        *,
        min_online_actor_version: int,
        logger_: logging.Logger,
    ):
        self._replay_client = replay_client
        self._warmup_min_size = max(int(warmup_min_size), 0)
        self._min_online_actor_version = max(int(min_online_actor_version), 0)
        self._logger = logger_
        self._status = "warmup_collect" if self._warmup_min_size > 0 else "online"
        self._episode_phase = "warmup" if self._warmup_min_size > 0 else "online"
        self._warmup_data_ready = self._warmup_min_size <= 0
        self._actor_version_getter: Callable[[], int] | None = None
        self._learner_status_getter: Callable[[], dict[str, Any]] | None = None
        self._logged_initial = False
        self._last_wait_actor_version = -1
        self._last_wait_global_step = -1

    @property
    def episode_phase(self) -> str:
        return self._episode_phase

    def bind_actor_version_getter(self, getter: Callable[[], int]) -> None:
        self._actor_version_getter = getter

    def bind_learner_status_getter(self, getter: Callable[[], dict[str, Any]]) -> None:
        self._learner_status_getter = getter

    def begin_episode(self) -> str:
        if self._warmup_min_size <= 0:
            self._status = "online"
            self._episode_phase = "online"
            if not self._logged_initial:
                self._logger.info("Rollout phase=online warmup disabled")
                self._logged_initial = True
            return self._episode_phase
        self.observe_progress()
        if self._warmup_data_ready and not self._is_online_ready():
            self._status = "warmup_wait_online"
            self._wait_until_online_ready()
        elif self._warmup_data_ready:
            self._status = "online"
        else:
            self._status = "warmup_collect"
        self._episode_phase = "online" if self._status == "online" else "warmup"
        if not self._logged_initial:
            self._logger.info(
                "Rollout phase=%s replay_size=%s/%s actor_version=%s required_actor_version=%s learner_updates=%s/%s",
                self._status,
                self._replay_size(),
                self._warmup_min_size,
                self._safe_actor_version(),
                self._min_online_actor_version,
                self._learner_global_step(),
                self._learner_warmup_required_updates(),
            )
            self._logged_initial = True
        return self._episode_phase

    def observe_progress(self) -> None:
        if self._warmup_min_size <= 0 or self._warmup_data_ready:
            return
        replay_size = self._replay_size()
        if replay_size < self._warmup_min_size:
            return
        self._warmup_data_ready = True
        self._logger.info(
            "Warmup data ready latched replay_size=%s/%s current_episode_phase=%s",
            replay_size,
            self._warmup_min_size,
            self._episode_phase,
        )

    def finish_episode(self) -> None:
        if self._warmup_min_size <= 0:
            self._status = "online"
            return
        self.observe_progress()
        if not self._warmup_data_ready:
            self._status = "warmup_collect"
            return
        if self._is_online_ready():
            self._status = "online"
            return
        self._status = "warmup_wait_online"

    def _replay_size(self) -> int:
        return int(self._replay_client.stats()["size"])

    def _safe_actor_version(self) -> int:
        if self._actor_version_getter is None:
            return -1
        try:
            return int(self._actor_version_getter())
        except RuntimeError:
            return -1

    def _is_actor_ready(self) -> bool:
        return self._safe_actor_version() >= self._min_online_actor_version

    def _safe_learner_status(self) -> dict[str, Any]:
        if self._learner_status_getter is None:
            return {}
        try:
            return dict(self._learner_status_getter())
        except RuntimeError:
            return {}

    def _learner_global_step(self) -> int:
        return int(self._safe_learner_status().get("global_step", 0))

    def _learner_warmup_required_updates(self) -> int:
        return int(self._safe_learner_status().get("warmup_required_updates", 0))

    def _is_training_ready(self) -> bool:
        return bool(self._safe_learner_status().get("ready_for_online", False))

    def _is_online_ready(self) -> bool:
        return self._is_actor_ready() and self._is_training_ready()

    def _wait_until_online_ready(self) -> None:
        while not rospy.is_shutdown():
            actor_version = self._safe_actor_version()
            learner_status = self._safe_learner_status()
            learner_global_step = int(learner_status.get("global_step", 0))
            warmup_required_updates = int(learner_status.get("warmup_required_updates", 0))
            if actor_version >= self._min_online_actor_version and bool(learner_status.get("ready_for_online", False)):
                self._status = "online"
                return
            if actor_version != self._last_wait_actor_version or learner_global_step != self._last_wait_global_step:
                self._logger.info(
                    "Waiting for online readiness actor_version=%s required=%s learner_updates=%s/%s",
                    actor_version,
                    self._min_online_actor_version,
                    learner_global_step,
                    warmup_required_updates,
                )
                self._last_wait_actor_version = actor_version
                self._last_wait_global_step = learner_global_step
            time.sleep(0.25)


def _resolve_min_online_actor_version(system: OnlineRLSystemConfig) -> int:
    if system.rl.warmup_min_size <= 0:
        return 0
    push_interval = max(int(system.learner_service.push_actor_interval_steps), 1)
    actor_period = max(int(system.rl.actor_update_period), 1)
    return max(push_interval // actor_period, 1)


def _make_learner_status_reader(path: Path) -> Callable[[], dict[str, Any]]:
    def _read() -> dict[str, Any]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return dict(payload) if isinstance(payload, dict) else {}

    return _read


class StaticOnlinePhaseController:
    @property
    def episode_phase(self) -> str:
        return "online"

    def bind_actor_version_getter(self, _getter: Callable[[], int]) -> None:
        return

    def bind_learner_status_getter(self, _getter: Callable[[], dict[str, Any]]) -> None:
        return

    def begin_episode(self) -> str:
        return "online"

    def observe_progress(self) -> None:
        return

    def finish_episode(self) -> None:
        return


class PhaseAwareActorClient:
    def __init__(
        self,
        actor_client: ActorClient,
        phase_controller: RolloutPhaseController | StaticOnlinePhaseController,
        runtime_context: RolloutRuntimeContext,
    ):
        self._actor_client = actor_client
        self._phase_controller = phase_controller
        self._runtime_context = runtime_context

    def infer(self, request: Any) -> ActorResponse:
        phase = self._phase_controller.episode_phase
        if (
            phase == "warmup"
            or not self._runtime_context.in_critical_phase()
            or self._runtime_context.episode_critical_policy_mode() != "actor"
        ):
            return ActorResponse(
                refined_chunk=np.asarray(request.ref_chunk, dtype=np.float32),
                actor_param_version=-1,
                request_id=request.request_id,
                timestamp=time.time(),
                source=int(TransitionSource.BASE),
            )
        return self._actor_client.infer(request)

    def get_actor_param_version(self) -> int:
        return self._actor_client.get_actor_param_version()


def _load_callable(path: str | None) -> Callable[..., Any] | None:
    if path is None:
        return None
    module_name, attr_name = path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class TaskState:
    def __init__(self, task: str):
        self._lock = threading.Lock()
        self._task = task

    def get(self) -> str:
        with self._lock:
            return self._task


class HumanInterventionState:
    def __init__(self, *, policy_enabled: bool = True):
        self._lock = threading.Lock()
        self._policy_enabled = policy_enabled
        self._episode_active = False
        self._need_reset_on_resume = False
        self._resume_until = 0.0

    def _mode_name_locked(self) -> str:
        if not self._episode_active:
            return "reset"
        return "policy" if self._policy_enabled else "teleop"

    def current_mode(self) -> str:
        with self._lock:
            return self._mode_name_locked()

    def is_policy_enabled(self) -> bool:
        with self._lock:
            return self._policy_enabled

    def in_resume_cooldown(self) -> bool:
        with self._lock:
            return time.time() < self._resume_until

    def toggle_policy(self, *, resume_delay_s: float) -> bool | None:
        with self._lock:
            if not self._episode_active:
                return None
            self._policy_enabled = not self._policy_enabled
            if self._policy_enabled:
                self._need_reset_on_resume = True
                self._resume_until = time.time() + max(float(resume_delay_s), 0.0)
            else:
                self._need_reset_on_resume = False
                self._resume_until = 0.0
            return self._policy_enabled

    def set_policy_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self._episode_active = True
            self._policy_enabled = bool(enabled)
            self._need_reset_on_resume = False
            self._resume_until = 0.0
            return self._policy_enabled

    def enter_episode_reset(self) -> str:
        with self._lock:
            self._episode_active = False
            self._policy_enabled = True
            self._need_reset_on_resume = False
            self._resume_until = 0.0
            return self._mode_name_locked()

    def consume_reset_request(self) -> bool:
        with self._lock:
            need = self._need_reset_on_resume
            self._need_reset_on_resume = False
            return need


class Ros1TeleopServices:
    def __init__(self, state: HumanInterventionState, service_name: str, resume_delay_s: float):
        self._state = state
        self._resume_delay_s = float(resume_delay_s)
        self._srv_toggle = rospy.Service(service_name, Trigger, self._on_toggle)
        self._srv_status = rospy.Service(TELEOP_STATUS_SERVICE, Trigger, self._on_status)
        logger.info("Human intervention service ready: %s", service_name)
        logger.info("Teleop status service ready: %s", TELEOP_STATUS_SERVICE)

    def _on_toggle(self, _request: Any) -> TriggerResponse:
        policy_enabled = self._state.toggle_policy(resume_delay_s=self._resume_delay_s)
        if policy_enabled is None:
            msg = "mode=reset Episode inactive/reset in progress; teleop toggle ignored."
            return TriggerResponse(success=False, message=msg)
        if policy_enabled:
            msg = f"mode=policy Policy ENABLED (delay {self._resume_delay_s:.2f}s)"
        else:
            msg = "mode=teleop Policy DISABLED, teleop enabled"
        return TriggerResponse(success=True, message=msg)

    def _on_status(self, _request: Any) -> TriggerResponse:
        mode = self._state.current_mode()
        return TriggerResponse(success=True, message=f"mode={mode}")

    def shutdown(self) -> None:
        self._srv_toggle.shutdown("shutdown")
        self._srv_status.shutdown("shutdown")


class Ros1ManualSignalServices:
    def __init__(self, runtime_context: RolloutRuntimeContext):
        self._runtime_context = runtime_context
        self._services = [
            rospy.Service(SHUTDOWN_ROLLOUT_SERVICE, Trigger, self._on_shutdown_rollout),
            rospy.Service(REQUEST_NEXT_EPISODE_SERVICE, Trigger, self._on_request_next_episode),
            rospy.Service(RECORD_SUCCESS_SERVICE, Trigger, self._on_record_success),
            rospy.Service(RECORD_FAILURE_SERVICE, Trigger, self._on_record_failure),
            rospy.Service(RECORD_DONE_SERVICE, Trigger, self._on_record_done),
            rospy.Service(ENTER_CRITICAL_PHASE_SERVICE, Trigger, self._on_enter_critical_phase),
            rospy.Service(TOGGLE_CRITICAL_PHASE_SERVICE, Trigger, self._on_toggle_critical_phase),
            rospy.Service(SET_CRITICAL_POLICY_ACTOR_SERVICE, Trigger, self._on_select_actor),
            rospy.Service(SET_CRITICAL_POLICY_BASE_SERVICE, Trigger, self._on_select_base),
        ]
        logger.info(
            "Manual services shutdown=%s next=%s success=%s failure=%s done=%s critical=%s toggle_critical=%s select_actor=%s select_base=%s",
            SHUTDOWN_ROLLOUT_SERVICE,
            REQUEST_NEXT_EPISODE_SERVICE,
            RECORD_SUCCESS_SERVICE,
            RECORD_FAILURE_SERVICE,
            RECORD_DONE_SERVICE,
            ENTER_CRITICAL_PHASE_SERVICE,
            TOGGLE_CRITICAL_PHASE_SERVICE,
            SET_CRITICAL_POLICY_ACTOR_SERVICE,
            SET_CRITICAL_POLICY_BASE_SERVICE,
        )

    def _on_shutdown_rollout(self, _request: Any) -> TriggerResponse:
        self._runtime_context.request_stop()
        return TriggerResponse(success=True, message="Rollout shutdown requested; robot will reset before exit.")

    def _on_request_next_episode(self, _request: Any) -> TriggerResponse:
        self._runtime_context.request_next_episode()
        return TriggerResponse(success=True, message="Next episode requested.")

    def _on_record_success(self, _request: Any) -> TriggerResponse:
        self._runtime_context.mark_manual_success()
        return TriggerResponse(success=True, message="Manual success recorded.")

    def _on_record_failure(self, _request: Any) -> TriggerResponse:
        self._runtime_context.mark_manual_failure()
        return TriggerResponse(success=True, message="Manual failure recorded.")

    def _on_record_done(self, _request: Any) -> TriggerResponse:
        self._runtime_context.mark_manual_done()
        return TriggerResponse(success=True, message="Manual done recorded.")

    def _on_enter_critical_phase(self, _request: Any) -> TriggerResponse:
        if self._runtime_context.task_mode() == "critical_phase":
            return TriggerResponse(success=True, message="Critical phase mode is already active for this episode.")
        if self._runtime_context.enter_critical_phase():
            return TriggerResponse(success=True, message="Entered critical phase.")
        return TriggerResponse(success=True, message="Critical phase was already active.")

    def _on_toggle_critical_phase(self, _request: Any) -> TriggerResponse:
        if self._runtime_context.task_mode() == "critical_phase":
            return TriggerResponse(
                success=True,
                message="Critical phase task mode is fixed for this episode; toggle is ignored.",
            )
        active = self._runtime_context.toggle_critical_phase()
        return TriggerResponse(success=True, message="Entered critical phase." if active else "Exited critical phase.")

    def _on_select_actor(self, _request: Any) -> TriggerResponse:
        self._runtime_context.set_selected_critical_policy_mode("actor")
        return TriggerResponse(success=True, message="Selected critical policy mode=actor for the next episode.")

    def _on_select_base(self, _request: Any) -> TriggerResponse:
        self._runtime_context.set_selected_critical_policy_mode("base")
        return TriggerResponse(success=True, message="Selected critical policy mode=base for the next episode.")

    def shutdown(self) -> None:
        for service in self._services:
            service.shutdown("shutdown")


class Ros1HumanActionRecorder:
    def __init__(self, cmd_topic: str):
        self._lock = threading.Lock()
        self._latest_action: np.ndarray | None = None
        self._latest_seq = -1
        self._sub = rospy.Subscriber(cmd_topic, JointState, self._on_action, queue_size=50)

    def _on_action(self, msg: JointState) -> None:
        action = np.asarray(msg.position, dtype=np.float32).reshape(-1)
        if action.shape[0] < 7:
            return
        with self._lock:
            self._latest_action = action[:7].copy()
            self._latest_seq += 1

    def snapshot_latest(self) -> tuple[np.ndarray | None, int]:
        with self._lock:
            if self._latest_action is None:
                return None, self._latest_seq
            return self._latest_action.copy(), self._latest_seq

    def shutdown(self) -> None:
        self._sub.unregister()


def _coerce_reward_output(reward: np.ndarray | list[float] | float, *, executed_steps: int) -> list[float]:
    if executed_steps <= 0:
        return []
    if np.isscalar(reward):
        return [float(reward)] * executed_steps
    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    if reward_array.shape[0] == 1:
        return [float(reward_array[0])] * executed_steps
    if reward_array.shape[0] != executed_steps:
        raise ValueError(f"Reward callback must return scalar or length {executed_steps}, got shape {reward_array.shape}.")
    return [float(x) for x in reward_array]


def _manual_terminal_events(signals: dict[str, Any]) -> tuple[bool, bool, bool]:
    success = bool(signals.get(SIGNAL_MANUAL_SUCCESS_PENDING, False))
    failure = bool(signals.get(SIGNAL_MANUAL_FAILURE_PENDING, False))
    done = bool(signals.get(SIGNAL_MANUAL_DONE_PENDING, False))
    return success, failure, done


def _default_reward_fn(
    _observation: dict[str, Any],
    _action_chunk: np.ndarray,
    _next_observation: dict[str, Any],
    context: dict[str, Any],
) -> np.ndarray:
    executed_steps = int(context["executed_steps"])
    rewards = np.zeros((executed_steps,), dtype=np.float32)
    success, _, _ = _manual_terminal_events(context["signals"])
    if executed_steps > 0 and success:
        rewards[-1] = 1.0
    return rewards


def _default_success_fn(
    _observation: dict[str, Any],
    _next_observation: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    success, _, _ = _manual_terminal_events(context["signals"])
    return success


def _default_done_fn(
    _observation: dict[str, Any],
    _next_observation: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    _, failure, done = _manual_terminal_events(context["signals"])
    return bool(failure or done)


class AgilexSingleArmROS1Bridge:
    def __init__(self, args: argparse.Namespace):
        self._args = args
        ros_args = robot_utils.get_arguments()
        self._ros_args = ros_args
        self._ros_operator = robot_utils.RosOperator(ros_args)
        self._single_arm = args.single_arm
        self._max_gripper_m = float(args.max_gripper_m)
        self._gripper_offset = float(args.gripper_offset_m)
        self._left_reset = np.asarray(args.left_reset, dtype=np.float32).reshape(7)
        self._right_reset = np.asarray(args.right_reset, dtype=np.float32).reshape(7)
        self._last_left = self._left_reset.copy()
        self._last_right = self._right_reset.copy()

    @staticmethod
    def _to_rgb_u8_hwc(image: np.ndarray, resize_hw: tuple[int, int]) -> np.ndarray:
        img = np.asarray(image)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape {img.shape}")
        img = image_tools.resize_with_pad(img, resize_hw[0], resize_hw[1])
        return image_tools.convert_to_uint8(img)

    def wait_for_observation_ready(self, timeout_s: float | None = None) -> None:
     
        try:
            _ = robot_utils.get_ros_observation(self._ros_args, self._ros_operator)
        except Exception:
    
            raise RuntimeError("Timeout waiting ROS1 observation.")
    

    def get_observation(self, resize_hw: tuple[int, int], task: str) -> dict[str, Any]:
        img_front, img_left, img_right, puppet_left, puppet_right, _base = robot_utils.get_ros_observation(
            self._ros_args,
            self._ros_operator,
        )
        left_pos = np.asarray(puppet_left.position, dtype=np.float32).reshape(-1)
        right_pos = np.asarray(puppet_right.position, dtype=np.float32).reshape(-1)
        if left_pos.shape[0] < 7 or right_pos.shape[0] < 7:
            raise RuntimeError("Expected both arm JointState.position dims >= 7.")
        self._last_left = left_pos[:7].copy()
        self._last_right = right_pos[:7].copy()
        if self._single_arm == "left":
            state = self._last_left
            wrist_img = img_left
        else:
            state = self._last_right
            wrist_img = img_right
        return {
            "state": state.astype(np.float32, copy=True),
            "images": {
                # Cobot single-arm policy expected keys (e.g., screw_sorting).
                "cam_high": self._to_rgb_u8_hwc(img_front, resize_hw),
                "cam_right_wrist": self._to_rgb_u8_hwc(wrist_img, resize_hw),
            },
            "prompt": task,
        }

    def send_action(self, action7: np.ndarray) -> None:
        action = np.asarray(action7, dtype=np.float32).reshape(-1)
        if action.shape[0] < 7:
            raise ValueError(f"Expected action dim >= 7, got {action.shape}")
        arm_target = action[:7].copy()
        arm_target[6] = float(np.clip(arm_target[6] + self._gripper_offset, 0.0, self._max_gripper_m))
        left_target = self._last_left.copy()
        right_target = self._last_right.copy()
        if self._single_arm == "left":
            left_target = arm_target
        else:
            right_target = arm_target
        self._ros_operator.puppet_arm_publish(left_target.tolist(), right_target.tolist())

    def move_to_reset(self, target_action: np.ndarray | None) -> None:
        left_target = self._left_reset.copy()
        right_target = self._right_reset.copy()
        print(f"left_target: {left_target}")
        if target_action is not None:
            target = np.asarray(target_action, dtype=np.float32).reshape(-1)
            if target.shape[0] >= 7:
                target[6] = float(np.clip(target[6] + self._gripper_offset, 0.0, self._max_gripper_m))
                if self._single_arm == "left":
                    left_target = target[:7]
                else:
                    right_target = target[:7]
        self._ros_operator.puppet_arm_publish_continuous(left_target.tolist(), right_target.tolist())

    def set_policy_control_active(self, _enabled: bool) -> None:
        return

    def shutdown(self) -> None:
        return


class AgilexChunkEnvAdapter(ChunkHorizonEnvMixin):
    def __init__(
        self,
        *,
        system: OnlineRLSystemConfig,
        robot: AgilexSingleArmROS1Bridge,
        task_state: TaskState,
        intervention_state: HumanInterventionState,
        human_action_recorder: Ros1HumanActionRecorder,
        phase_controller: RolloutPhaseController | StaticOnlinePhaseController,
        runtime_context: RolloutRuntimeContext,
        reward_fn: RewardFn,
        success_fn: SuccessFn,
        done_fn: DoneFn,
        safe_action_filter: ActionFilterFn | None = None,
        max_chunk_steps_per_episode: int = 200,
        idle_sleep_sec: float = 0.02,
        action_delta_limits: np.ndarray | None = None,
        resize_hw: tuple[int, int] = (224, 224),
        obs_ready_timeout_s: float | None = None,
    ):
        self._system = system
        self._robot = robot
        self._task_state = task_state
        self._intervention_state = intervention_state
        self._human_action_recorder = human_action_recorder
        self._phase_controller = phase_controller
        self._runtime_context = runtime_context
        self._reward_fn = reward_fn
        self._success_fn = success_fn
        self._done_fn = done_fn
        self._safe_action_filter = safe_action_filter
        self._max_chunk_steps_per_episode = max_chunk_steps_per_episode
        self._idle_sleep_sec = idle_sleep_sec
        self._resize_hw = resize_hw
        self._obs_ready_timeout_s = obs_ready_timeout_s
        self._episode_chunk_step = 0
        self._last_sent_action: np.ndarray | None = None
        self._last_human_seq = -1
        self._last_human_action: np.ndarray | None = None
        self._task_mode = str(system.env_driver.task_mode)
        if self._task_mode not in {"full_task", "critical_phase"}:
            raise ValueError(f"Unsupported task_mode={self._task_mode!r}")
        self._action_delta_limits = None
        if action_delta_limits is not None:
            limits = np.asarray(action_delta_limits, dtype=np.float32).reshape(-1)
            if limits.shape[0] != system.rl.action_dim:
                raise ValueError(f"action_delta_limits must have {system.rl.action_dim} entries, got {limits.shape[0]}")
            self._action_delta_limits = limits

        self._rtc: RtcActionRuntime | None = None
        if bool(system.env_driver.use_rtc):
            self._rtc = RtcActionRuntime(
                fps=float(system.env_driver.control_frequency_hz),
                action_queue_size_to_get_new_actions=resolve_rtc_refill_threshold(
                    system.env_driver,
                    system.rl,
                ),
                guided_inference_delay=system.env_driver.rtc_inference_delay,
            )
            logger.info(
                "RTC enabled fps=%.1f vla_guidance=%s guided_d=%s",
                float(system.env_driver.control_frequency_hz),
                bool(system.env_driver.rtc_vla_guidance),
                system.env_driver.rtc_inference_delay,
            )

    def move_to_home_on_shutdown(self) -> None:
        """Move the robot to the configured reset pose during rollout shutdown."""
        self._intervention_state.enter_episode_reset()
        self._reset_robot_to_mode_start()

    def reset(self) -> dict[str, Any]:
        if self._runtime_context.stop_requested():
            raise KeyboardInterrupt
        if self._rtc is not None:
            self._rtc.reset()
        self._episode_chunk_step = 0
        self._last_sent_action = None
        latest_human_action, latest_human_seq = self._human_action_recorder.snapshot_latest()
        self._last_human_seq = latest_human_seq
        self._last_human_action = None if latest_human_action is None else latest_human_action.astype(np.float32, copy=False)
        self._intervention_state.enter_episode_reset()
        self._runtime_context.reset_episode_state()
    
        self._robot.wait_for_observation_ready(timeout_s=self._obs_ready_timeout_s)
  
        self._reset_robot_to_mode_start()
        self._phase_controller.begin_episode()
        logger.info("Waiting for next episode request task_mode=%s", self._task_mode)
        self._runtime_context.wait_for_next_episode_request()
        if self._runtime_context.stop_requested():
            raise KeyboardInterrupt
        locked_mode = self._runtime_context.lock_episode_critical_policy_mode()
        logger.info("Episode critical policy mode=%s", locked_mode)
        self._apply_episode_start_control_mode()
        return self._robot.get_observation(self._resize_hw, self._task_state.get())

    def current_phase_name(self) -> str:
        segment = "critical" if self._runtime_context.in_critical_phase() else "base"
        return f"{self._phase_controller.episode_phase}:{self._task_mode}:{segment}"

    def execute_chunk(
        self,
        *,
        observation: dict[str, Any] | None = None,
        control_hz: float,
        policy_planner: Callable[..., PolicyPlan] | None = None,
    ) -> tuple[dict[str, Any], list[float], bool, dict[str, Any]]:
        """Per-tick: obs → act → sleep. Trailing get gives o_H for step_trace; next chunk reuses it as o0."""
        if self._runtime_context.stop_requested():
            raise KeyboardInterrupt
        self._phase_controller.observe_progress()
        phase = self._phase_controller.episode_phase
        critical_started = self._runtime_context.in_critical_phase()
        chunk_phase = phase
        chunk_critical_started = critical_started
        self.refresh_chunk_horizon_state()
        uses_rl = bool(self.policy_uses_rl_actor())
        period = 1.0 / max(float(control_hz), 1e-6)
        horizon = self.current_chunk_exec_horizon()

        if self._rtc is not None:
            # Evo: one refill threshold for the whole episode; only RTC ``s`` changes per phase.
            self._rtc.note_phase(
                phase,
                critical_started,
                self._runtime_context.episode_critical_policy_mode(),
                uses_rl,
            )

        executed: list[np.ndarray] = []
        ref_actions: list[np.ndarray] = []
        human_controlled: list[bool] = []
        step_sources: list[int] = []
        actor_param_versions: list[int] = []
        # Length H while looping; append o_H after loop → H+1 for step_trace next_obs.
        step_observations: list[dict[str, Any]] = []
        policy_anchor_offsets: list[int] = []
        policy_anchor_features: list[ChunkFeatures] = []
        chunk_start_features = None
        current_plan: PolicyPlan | None = None
        plan_cursor = 0
        phase_interrupted = False
        # First tick may reuse EnvDriver's observation (Evo has no chunk seam).
        pending_obs = observation

        for local_step in range(horizon):
            tick_start = time.perf_counter()

            if self._runtime_context.stop_requested():
                raise KeyboardInterrupt
            if self._manual_terminal_requested():
                break

            # Re-sample phase each tick (Evo resets RTC on set_rl/set_vla/critical).
            phase = self._phase_controller.episode_phase
            critical_started = self._runtime_context.in_critical_phase()
            prev_uses_rl = uses_rl
            self.refresh_chunk_horizon_state()
            uses_rl = bool(self.policy_uses_rl_actor())
            if self._rtc is not None:
                phase_changed = self._rtc.note_phase(
                    phase,
                    critical_started,
                    self._runtime_context.episode_critical_policy_mode(),
                    uses_rl,
                )
                if phase_changed or uses_rl != prev_uses_rl:
                    current_plan = None
                    plan_cursor = 0
                    if local_step > 0 and phase_changed:
                        # End this env chunk early so replay windows stay phase-pure.
                        phase_interrupted = True
                        break

            policy_enabled = bool(
                self._intervention_state.is_policy_enabled() and not self._intervention_state.in_resume_cooldown()
            )
            if self._rtc is not None:
                if self._rtc.note_policy_enabled(policy_enabled):
                    current_plan = None
                    plan_cursor = 0
            if not policy_enabled:
                current_plan = None
                plan_cursor = 0

            if pending_obs is not None:
                step_observation = pending_obs
                pending_obs = None
            else:
                step_observation = self._robot.get_observation(self._resize_hw, self._task_state.get())

            if policy_enabled and policy_planner is not None and self._rtc is not None:
                # RL: async ActionQueue only — skip leftover-guided VLA (short C thrash / clamp d).
                rtc_result = self._rtc.ensure_action(
                    observation=step_observation,
                    local_step=local_step,
                    planner=policy_planner,
                    # use vla guidance if enabled and not using rl
                    use_vla_guidance=bool(self._system.env_driver.rtc_vla_guidance) and not uses_rl,
                    execution_horizon=resolve_rtc_execution_horizon(
                        self._system.env_driver,
                        self._system.rl,
                        uses_rl_actor=uses_rl,
                    ),
                    rl_refine_steps=None,
                )
                meta = rtc_result.metadata
                if local_step == 0 and meta.is_plan_anchor:
                    chunk_start_features = meta.start_features
                elif meta.is_plan_anchor:
                    policy_anchor_offsets.append(local_step)
                    policy_anchor_features.append(meta.start_features)
                raw_action = np.asarray(rtc_result.action, dtype=np.float32)[: self._system.rl.action_dim]
                bounded = self._apply_action_limits(raw_action)
                self._robot.send_action(bounded)
                executed.append(bounded)
                ref_actions.append(np.asarray(meta.ref_action, dtype=np.float32)[: self._system.rl.action_dim])
                human_controlled.append(bool(meta.source == int(TransitionSource.HUMAN)))
                step_sources.append(int(meta.source))
                actor_param_versions.append(int(meta.actor_param_version))
            elif policy_enabled and policy_planner is not None:
                if current_plan is None or plan_cursor >= current_plan.action_chunk.shape[0]:
                    current_plan = policy_planner(step_observation, local_step)
                    plan_cursor = 0
                    if current_plan.source != int(TransitionSource.HUMAN):
                        if local_step == 0:
                            chunk_start_features = current_plan.start_features
                        else:
                            policy_anchor_offsets.append(local_step)
                            policy_anchor_features.append(current_plan.start_features)

                if current_plan is not None and plan_cursor < current_plan.action_chunk.shape[0]:
                    raw_action = np.asarray(current_plan.action_chunk[plan_cursor], dtype=np.float32)[
                        : self._system.rl.action_dim
                    ]
                    bounded = self._apply_action_limits(raw_action)
                    self._robot.send_action(bounded)
                    executed.append(bounded)
                    ref_actions.append(
                        np.asarray(current_plan.ref_chunk[plan_cursor], dtype=np.float32)[
                            : self._system.rl.action_dim
                        ]
                    )
                    human_controlled.append(bool(current_plan.source == int(TransitionSource.HUMAN)))
                    step_sources.append(int(current_plan.source))
                    actor_param_versions.append(int(current_plan.actor_param_version))
                    plan_cursor += 1
            else:
                human_action = self._sample_latest_human_action(step_observation)
                bounded_human = self._apply_action_limits(human_action)
                # SpaceMouse already commands /master; skip duplicate puppet publish to reduce jitter.
                executed.append(bounded_human)
                ref_actions.append(bounded_human.copy())
                human_controlled.append(True)
                step_sources.append(int(TransitionSource.HUMAN))
                actor_param_versions.append(-1)

            elapsed = time.perf_counter() - tick_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            else:
                logger.warning("Tick took too long: %.4f s (period=%.4f)", elapsed, period)
            step_observations.append(step_observation)

        # o_H after last sleep; EnvDriver reuses it as next chunk o0 (no second get).
        next_observation = self._robot.get_observation(self._resize_hw, self._task_state.get())
        if executed:
            step_observations.append(next_observation)

        signal_snapshot = self._runtime_context.snapshot_signals()
        context = {
            "episode_chunk_step": self._episode_chunk_step,
            "executed_steps": len(executed),
            "interrupted": bool(any(human_controlled) or phase_interrupted),
            "phase": chunk_phase,
            "task_mode": self._task_mode,
            "critical_started": chunk_critical_started,
            "phase_interrupted": phase_interrupted,
            "runtime": self._runtime_context,
            "signals": signal_snapshot,
        }
        chunk_start_obs = step_observations[0] if step_observations else next_observation
        rewards = _coerce_reward_output(
            self._reward_fn(
                chunk_start_obs,
                np.asarray(executed, dtype=np.float32)
                if executed
                else np.zeros((0, self._system.rl.action_dim), dtype=np.float32),
                next_observation,
                context,
            ),
            executed_steps=len(executed),
        )
        success = int(bool(self._success_fn(chunk_start_obs, next_observation, context)))
        manual_done = bool(self._done_fn(chunk_start_obs, next_observation, context))
        terminal_requested = bool(success or manual_done)
        self._consume_manual_terminal_events(signal_snapshot)
        human_intervened = any(human_controlled)
        step_trace = [
            {
                "observation": step_observations[idx],
                "action": executed[idx],
                "ref_action": ref_actions[idx],
                "reward": rewards[idx],
                "next_observation": step_observations[idx + 1],
                "human_controlled": human_controlled[idx],
                "source": step_sources[idx],
                "actor_param_version": actor_param_versions[idx],
                "done": False,
            }
            for idx in range(len(executed))
        ]

        if not executed:
            self._last_sent_action = None
            if terminal_requested:
                self._on_episode_done(terminal_requested=True)
            return next_observation, rewards, terminal_requested, {
                "drop_transition": True,
                "intervention_flag": False,
                "source": int(TransitionSource.HUMAN),
                "success": success,
                "step_trace": step_trace,
                "policy_anchor_offsets": policy_anchor_offsets,
                "policy_anchor_features": policy_anchor_features,
                "chunk_start_features": chunk_start_features,
                "phase_interrupted": phase_interrupted,
            }

        self._episode_chunk_step += 1
        done = bool(terminal_requested or (self._episode_chunk_step >= self._max_chunk_steps_per_episode))
        if done:
            self._on_episode_done(terminal_requested=terminal_requested)
        if step_trace:
            step_trace[-1]["done"] = done
        if human_intervened:
            source = int(TransitionSource.MIXED if any(not flag for flag in human_controlled) else TransitionSource.HUMAN)
        else:
            source = int(step_sources[0])
        info = {
            "intervention_flag": human_intervened,
            "source": source,
            "success": success,
            "step_trace": step_trace,
            "policy_anchor_offsets": policy_anchor_offsets,
            "policy_anchor_features": policy_anchor_features,
            "chunk_start_features": chunk_start_features,
            "phase_interrupted": phase_interrupted,
        }
        if not chunk_critical_started:
            info["drop_transition"] = True
        return next_observation, rewards, done, info

    def _wait_until_policy_active(self) -> None:
        while not rospy.is_shutdown():
            if self._intervention_state.consume_reset_request():
                self._last_sent_action = None
                logger.info("Policy resumed: runtime state cleared.")
            if self._intervention_state.is_policy_enabled() and not self._intervention_state.in_resume_cooldown():
                return
            time.sleep(self._idle_sleep_sec)

    def _apply_episode_start_control_mode(self) -> None:
        start_mode = self._system.env_driver.episode_start_control_mode
        if start_mode == "sticky":
            policy_enabled = self._intervention_state.is_policy_enabled()
        else:
            policy_enabled = start_mode == "policy"
        resolved_mode = self._intervention_state.set_policy_enabled(policy_enabled)
        self._robot.set_policy_control_active(resolved_mode)
        if resolved_mode:
            self._wait_until_policy_active()

    def _consume_manual_terminal_events(self, signal_snapshot: dict[str, Any]) -> None:
        to_clear = [
            name
            for name in (SIGNAL_MANUAL_SUCCESS_PENDING, SIGNAL_MANUAL_FAILURE_PENDING, SIGNAL_MANUAL_DONE_PENDING)
            if bool(signal_snapshot.get(name, False))
        ]
        if to_clear:
            self._runtime_context.clear_signals(*to_clear)

    def _manual_terminal_requested(self) -> bool:
        success, failure, done = _manual_terminal_events(self._runtime_context.snapshot_signals())
        return bool(success or failure or done)

    def _reset_target_for_mode(self) -> np.ndarray | None:
        raw = (
            self._system.env_driver.critical_phase_reset_action
            if self._task_mode == "critical_phase"
            else self._system.env_driver.full_task_reset_action
        )
        if raw is None:
            return None
        target = np.asarray(raw, dtype=np.float32).reshape(-1)
        if target.shape[0] != self._system.rl.action_dim:
            raise ValueError(
                f"{self._task_mode} reset action must have {self._system.rl.action_dim} entries, got {target.shape[0]}"
            )
        return target

    def _reset_robot_to_mode_start(self) -> None:
        self._robot.move_to_reset(self._reset_target_for_mode())

    def _on_episode_done(self, *, terminal_requested: bool) -> None:
        if self._rtc is not None:
            self._rtc.reset()
        self._intervention_state.enter_episode_reset()
        self._phase_controller.finish_episode()
        if terminal_requested:
            logger.info("Episode terminal signal received; moving robot to reset pose.")
            try:
                self._reset_robot_to_mode_start()
            except Exception as exc:
                logger.warning("Failed to move robot to reset pose after episode end: %s", exc)

    def _apply_action_limits(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(-1)[: self._system.rl.action_dim]
        if self._action_delta_limits is None or self._last_sent_action is None:
            self._last_sent_action = action.copy()
            return action
        delta = np.clip(action - self._last_sent_action, -self._action_delta_limits, self._action_delta_limits)
        bounded = self._last_sent_action + delta
        self._last_sent_action = bounded.copy()
        return bounded

    def _sample_latest_human_action(self, observation: dict[str, Any]) -> np.ndarray:
        latest_action, latest_seq = self._human_action_recorder.snapshot_latest()
        if latest_action is not None and latest_seq != self._last_human_seq:
            self._last_human_seq = latest_seq
            self._last_human_action = latest_action.astype(np.float32, copy=False)
        if self._last_human_action is None:
            state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            self._last_human_action = state[: self._system.rl.action_dim].astype(np.float32, copy=True)
        self._last_sent_action = self._last_human_action.copy()
        return self._last_human_action.copy()


def parse_args() -> argparse.Namespace:
    default_config = REPO_ROOT / "configs" / "tasks" / "screw_sorting" / "online_rl.yaml"
    parser = argparse.ArgumentParser(description="ROS1 AgileX single-arm runner for OpenPI RLT online RL.")
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--task", type=str, default="Please sort and return the silver screws in the grey box to their proper places.")
    parser.add_argument("--single_arm", type=str, choices=("left", "right"), default="right")
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Max episodes per rollout session (default: unlimited; press 'o' between episodes). "
        "Use a finite value only for one-shot eval runs.",
    )
    parser.add_argument("--max_chunk_steps_per_episode", type=int, default=20000)
    parser.add_argument("--idle_sleep_sec", type=float, default=0.02)
    parser.add_argument("--machine_a_ws_url", type=str, default=None)
    parser.add_argument("--actor_service_url", type=str, default=None)
    parser.add_argument("--replay_service_url", type=str, default=None)
    parser.add_argument("--reward_factory", type=str, default=None)
    parser.add_argument("--success_factory", type=str, default=None)
    parser.add_argument("--done_factory", type=str, default=None)
    parser.add_argument("--safe_action_filter_factory", type=str, default=None)
    parser.add_argument("--action_delta_limits", type=float, nargs=7, default=None)
    parser.add_argument("--image_h", type=int, default=224)
    parser.add_argument("--image_w", type=int, default=224)
    parser.add_argument("--max_gripper_m", type=float, default=0.097)
    parser.add_argument("--gripper_offset_m", type=float, default=-0.005)
    parser.add_argument("--obs_ready_timeout_s", type=float, default=None)
    parser.add_argument("--global_image_topic", type=str, default="/camera_f/color/image_raw")
    parser.add_argument("--left_image_topic", type=str, default="/camera_l/color/image_raw")
    parser.add_argument("--right_image_topic", type=str, default="/camera_r/color/image_raw")
    parser.add_argument("--left_joint_topic", type=str, default="/puppet/joint_left")
    parser.add_argument("--right_joint_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--left_cmd_topic", type=str, default="/master/joint_left")
    parser.add_argument("--right_cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--teleop_trigger_service", type=str, default="/teleop_trigger_rl")
    parser.add_argument("--policy_resume_delay_s", type=float, default=0.2)
    parser.add_argument("--start_in_human_mode", action="store_true")
    parser.add_argument("--step_trace_stride", type=int, default=None)
    parser.add_argument("--eval_actor_only", action="store_true")
    parser.add_argument(
        "--left_reset",
        type=float,
        nargs=7,
        default=[-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.09],
    )
    parser.add_argument(
        "--right_reset",
        type=float,
        nargs=7,
        default=[-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.09],
        
    )
    return parser.parse_args()


def _override_system_urls(system: OnlineRLSystemConfig, args: argparse.Namespace) -> OnlineRLSystemConfig:
    env_driver = system.env_driver
    updates: dict[str, Any] = {}
    if args.machine_a_ws_url is not None:
        updates["machine_a_ws_url"] = args.machine_a_ws_url
    if args.actor_service_url is not None:
        updates["actor_service_url"] = args.actor_service_url
    if args.replay_service_url is not None:
        updates["replay_service_url"] = args.replay_service_url
    if not updates:
        return system
    return dataclasses.replace(system, env_driver=dataclasses.replace(env_driver, **updates))


def main() -> None:
    args = parse_args()
    system = _override_system_urls(load_system_config_yaml(args.config), args)
    configured_stride = int(system.env_driver.step_trace_stride)
    override_stride = configured_stride if args.step_trace_stride is None else max(int(args.step_trace_stride), 0)
    effective_step_trace_stride = 0 if args.eval_actor_only else override_stride
    effective_actor_deterministic = True if args.eval_actor_only else system.env_driver.actor_deterministic
    system = dataclasses.replace(
        system,
        env_driver=dataclasses.replace(
            system.env_driver,
            actor_deterministic=effective_actor_deterministic,
            step_trace_stride=effective_step_trace_stride,
        ),
    )
    log_path = setup_process_logging("pika_sync_ros1_agilex_single_arm", system, console_level=logging.INFO)

    reward_fn = _load_callable(args.reward_factory) or _default_reward_fn
    success_fn = _load_callable(args.success_factory) or _default_success_fn
    done_fn = _load_callable(args.done_factory) or _default_done_fn
    safe_action_filter = _load_callable(args.safe_action_filter_factory)

    task_state = TaskState(args.task)
    intervention_state = HumanInterventionState(policy_enabled=not args.start_in_human_mode)
    robot = AgilexSingleArmROS1Bridge(args)
    human_action_topic = args.right_cmd_topic if args.single_arm == "right" else args.left_cmd_topic
    human_action_recorder = Ros1HumanActionRecorder(human_action_topic)
    runtime_context = RolloutRuntimeContext(
        system=system,
        task_state=task_state,
        intervention_state=intervention_state,
    )
    teleop_services = Ros1TeleopServices(
        state=intervention_state,
        service_name=args.teleop_trigger_service,
        resume_delay_s=args.policy_resume_delay_s,
    )
    manual_services = Ros1ManualSignalServices(runtime_context)

    feature_provider = MachineAFeatureClient(
        system.env_driver.machine_a_ws_url,
        connect_timeout_sec=system.env_driver.machine_a_connect_timeout_sec,
        recv_timeout_sec=system.env_driver.machine_a_recv_timeout_sec,
        retry_interval_sec=system.env_driver.machine_a_retry_interval_sec,
    )
    replay_client = (
        NullReplayClient()
        if args.eval_actor_only
        else ReplayClient(
            system.env_driver.replay_service_url,
            timeout_sec=system.env_driver.replay_request_timeout_sec,
        )
    )
    min_online_actor_version = 0 if args.eval_actor_only else _resolve_min_online_actor_version(system)
    learner_status_path = metrics_path_for(system, "learner_status.json")
    phase_controller: RolloutPhaseController | StaticOnlinePhaseController
    phase_controller = (
        StaticOnlinePhaseController()
        if args.eval_actor_only
        else RolloutPhaseController(
            replay_client,
            system.rl.warmup_min_size,
            min_online_actor_version=min_online_actor_version,
            logger_=logger,
        )
    )
    base_actor_client = ActorClient(
        system.env_driver.actor_service_url,
        timeout_sec=system.env_driver.actor_request_timeout_sec,
    )
    phase_controller.bind_actor_version_getter(base_actor_client.get_actor_param_version)
    phase_controller.bind_learner_status_getter(_make_learner_status_reader(learner_status_path))
    actor_client = PhaseAwareActorClient(base_actor_client, phase_controller, runtime_context)
    env = AgilexChunkEnvAdapter(
        system=system,
        robot=robot,
        task_state=task_state,
        intervention_state=intervention_state,
        human_action_recorder=human_action_recorder,
        phase_controller=phase_controller,
        runtime_context=runtime_context,
        reward_fn=reward_fn,
        success_fn=success_fn,
        done_fn=done_fn,
        safe_action_filter=safe_action_filter,
        max_chunk_steps_per_episode=args.max_chunk_steps_per_episode,
        idle_sleep_sec=args.idle_sleep_sec,
        action_delta_limits=args.action_delta_limits,
        resize_hw=(args.image_h, args.image_w),
        obs_ready_timeout_s=args.obs_ready_timeout_s,
    )
    driver = EnvDriver(
        env=env,
        feature_provider=feature_provider,
        actor_client=actor_client,
        replay_client=replay_client,
        rl_config=system.rl,
        env_config=system.env_driver,
        eval_actor_only=args.eval_actor_only,
        metrics_path=str(metrics_path_for(system, "robot_rollout_metrics.jsonl")),
    )
  
    logger.info("Starting ROS1 AgileX rollout log=%s config=%s", log_path, args.config)
    logger.info("Machine A ws: %s", system.env_driver.machine_a_ws_url)
    logger.info("Actor service: %s", system.env_driver.actor_service_url)
    logger.info("Replay service: %s", system.env_driver.replay_service_url)
    logger.info("Task mode: %s single_arm=%s", system.env_driver.task_mode, args.single_arm)
    logger.info("Eval actor only: %s", args.eval_actor_only)

    try:
        driver.run_forever(num_episodes=args.num_episodes)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down.")
    finally:
        if runtime_context.stop_requested():
            logger.info("Stop requested; moving robot to reset pose before exit.")
            try:
                env.move_to_home_on_shutdown()
            except Exception as exc:
                logger.warning("Failed to move robot home on shutdown: %s", exc)
        human_action_recorder.shutdown()
        manual_services.shutdown()
        teleop_services.shutdown()
        robot.shutdown()


if __name__ == "__main__":
    main()

