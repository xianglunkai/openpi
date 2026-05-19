#!/usr/bin/env python3
"""Self-contained ROS real-robot adapter for standalone RLT online RL.

This script owns the robot-side ROS integration only:
- subscribes observations
- publishes actions
- manages teleop / human takeover state
- exposes a PikaChunkEnvAdapter to the generic EnvDriver

The online RL rollout orchestration remains in `rlt_online_rl.inference.EnvDriver`.
"""

from __future__ import annotations

import argparse
from collections import deque
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
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

try:
    from data_msgs.msg import Gripper as GripperMsg
except Exception:
    GripperMsg = None

from manual_signal_bridge import ENTER_CRITICAL_PHASE_SERVICE
from manual_signal_bridge import RECORD_DONE_SERVICE
from manual_signal_bridge import RECORD_FAILURE_SERVICE
from manual_signal_bridge import RECORD_SUCCESS_SERVICE
from manual_signal_bridge import REQUEST_NEXT_EPISODE_SERVICE
from manual_signal_bridge import SET_CRITICAL_POLICY_ACTOR_SERVICE
from manual_signal_bridge import SET_CRITICAL_POLICY_BASE_SERVICE
from manual_signal_bridge import SIGNAL_CRITICAL_STARTED
from manual_signal_bridge import SIGNAL_EPISODE_CRITICAL_POLICY
from manual_signal_bridge import SIGNAL_MANUAL_DONE_PENDING
from manual_signal_bridge import SIGNAL_MANUAL_FAILURE_PENDING
from manual_signal_bridge import SIGNAL_MANUAL_SUCCESS_PENDING
from manual_signal_bridge import SIGNAL_NEXT_EPISODE_REQUESTED
from manual_signal_bridge import SIGNAL_SELECTED_CRITICAL_POLICY
from manual_signal_bridge import SIGNAL_TASK_MODE
from manual_signal_bridge import TOGGLE_CRITICAL_PHASE_SERVICE
from manual_signal_bridge import ManualSignalBridge

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openpi_client import image_tools

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
from rlt_online_rl.runtime_logging import metrics_path_for
from rlt_online_rl.runtime_logging import setup_process_logging

RewardFn = Callable[[dict[str, Any], np.ndarray, dict[str, Any], dict[str, Any]], np.ndarray | list[float] | float]
SuccessFn = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool | int]
DoneFn = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool | int]
ActionFilterFn = Callable[[np.ndarray], np.ndarray]


logger = logging.getLogger("pika_sync_ros")


TELEOP_STATUS_SERVICE = "/teleop_status"


@dataclasses.dataclass(slots=True)
class RolloutRuntimeContext:
    system: OnlineRLSystemConfig
    obs_node: ROSObsBuffer
    task_state: TaskState
    intervention_state: HumanInterventionState
    robot: PikaRobotROSBridge
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

    def wait_for_next_episode_request(self) -> None:
        with self._condition:
            while rclpy.ok() and not bool(self.signal_values.get(SIGNAL_NEXT_EPISODE_REQUESTED, False)):
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
            self._logger.info(
                "Warmup complete; next episode will use online actor actor_version=%s required=%s learner_updates=%s/%s",
                self._safe_actor_version(),
                self._min_online_actor_version,
                self._learner_global_step(),
                self._learner_warmup_required_updates(),
            )
            return
        self._status = "warmup_wait_online"
        self._logger.info(
            "Warmup complete; waiting for online readiness actor_version=%s required=%s learner_updates=%s/%s",
            self._safe_actor_version(),
            self._min_online_actor_version,
            self._learner_global_step(),
            self._learner_warmup_required_updates(),
        )

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
        while True:
            actor_version = self._safe_actor_version()
            learner_status = self._safe_learner_status()
            learner_global_step = int(learner_status.get("global_step", 0))
            warmup_required_updates = int(learner_status.get("warmup_required_updates", 0))
            if actor_version >= self._min_online_actor_version and bool(learner_status.get("ready_for_online", False)):
                self._status = "online"
                self._logger.info(
                    "Online rollout ready actor_version=%s required=%s learner_updates=%s/%s",
                    actor_version,
                    self._min_online_actor_version,
                    learner_global_step,
                    warmup_required_updates,
                )
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
        if not isinstance(payload, dict):
            return {}
        return payload

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
        phase_controller: RolloutPhaseController,
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


def _bind_runtime_hook(callback: Any, runtime_context: RolloutRuntimeContext) -> list[Node]:
    bind = getattr(callback, "bind_runtime", None)
    if bind is None:
        return []
    bound = bind(runtime_context)
    if bound is None:
        return []
    if isinstance(bound, Node):
        return [bound]
    if isinstance(bound, (list, tuple)):
        return [node for node in bound if isinstance(node, Node)]
    raise TypeError(f"{type(callback).__name__}.bind_runtime() must return None, a Node, or a list/tuple of Node.")


def _missing_observation_fields(
    js: JointState | None,
    global_img: ROSImage | None,
    fisheye_img: ROSImage | None,
    depth_img: ROSImage | None,
) -> list[str]:
    missing = []
    if js is None:
        missing.append("joint")
    if global_img is None:
        missing.append("global_image")
    if fisheye_img is None:
        missing.append("fisheye_image")
    if depth_img is None:
        missing.append("depth_image")
    return missing


def _ros_stamp_to_sec(stamp: Any) -> float:
    sec = getattr(stamp, "sec", 0)
    nanosec = getattr(stamp, "nanosec", 0)
    return float(sec) + float(nanosec) * 1e-9


def _build_obs_subscription_qos(*, capture_like: bool, depth: int) -> QoSProfile:
    if not capture_like:
        return qos_profile_sensor_data
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=max(int(depth), 1),
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
    )


def _image_msg_to_rgb_u8_hwc(msg: ROSImage, resize_hw: tuple[int, int]) -> np.ndarray:
    if msg.height <= 0 or msg.width <= 0:
        raise ValueError(f"Invalid image shape from ROS message: {(msg.height, msg.width)}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if raw.size == msg.height * msg.width * 3:
        img = raw.reshape(msg.height, msg.width, 3)
    elif msg.step > 0 and raw.size >= msg.height * msg.step:
        row = raw[: msg.height * msg.step].reshape(msg.height, msg.step)
        channels = msg.step // msg.width
        if channels < 3:
            raise ValueError(f"Unsupported image step/channels: step={msg.step}, width={msg.width}")
        img = row[:, : msg.width * channels].reshape(msg.height, msg.width, channels)[..., :3]
    else:
        raise ValueError(
            f"Cannot decode ROS image buffer: size={raw.size}, " f"shape=({msg.height}, {msg.width}), step={msg.step}"
        )

    encoding = (msg.encoding or "").lower()
    if encoding == "bgr8":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif encoding != "rgb8":
        logging.warning("Unsupported image encoding %r; treating as BGR8.", msg.encoding)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = image_tools.resize_with_pad(img, resize_hw[0], resize_hw[1])
    return image_tools.convert_to_uint8(img)


class TaskState:
    def __init__(self, task: str):
        self._lock = threading.Lock()
        self._task = task

    def get(self) -> str:
        with self._lock:
            return self._task

    def set(self, task: str) -> None:
        with self._lock:
            self._task = task


class HumanInterventionState:
    """Thread-safe policy/human takeover state with resume cooldown semantics."""

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
            self._need_reset_on_resume = False
            self._resume_until = 0.0
            return self._mode_name_locked()

    def consume_reset_request(self) -> bool:
        with self._lock:
            need = self._need_reset_on_resume
            self._need_reset_on_resume = False
            return need


class TeleopTriggerNode(Node):
    """ROS Trigger service to toggle policy control and human takeover."""

    def __init__(
        self,
        intervention_state: HumanInterventionState,
        service_name: str,
        resume_delay_s: float,
        gripper_streamer: GripperCtrlStreamer | None = None,
    ):
        super().__init__("teleop_toggle_server")
        self._state = intervention_state
        self._resume_delay_s = float(resume_delay_s)
        self._gripper_streamer = gripper_streamer
        self._srv = self.create_service(Trigger, service_name, self._on_trigger)
        self._status_srv = self.create_service(Trigger, TELEOP_STATUS_SERVICE, self._on_status)
        self.get_logger().info(f"Human intervention service ready: {service_name}")
        self.get_logger().info(f"Teleop status service ready: {TELEOP_STATUS_SERVICE}")

    def _mode_name(self) -> str:
        return self._state.current_mode()

    def _on_trigger(self, _request, response):
        policy_enabled = self._state.toggle_policy(resume_delay_s=self._resume_delay_s)
        if policy_enabled is None:
            msg = "mode=reset Episode inactive/reset in progress; teleop toggle ignored."
            self.get_logger().warning(msg)
            response.success = False
            response.message = msg
            return response
        if not policy_enabled and self._gripper_streamer is not None:
            self._gripper_streamer.pause_stream()
        if policy_enabled:
            msg = f"mode=policy Policy ENABLED (teleop disabled, delay {self._resume_delay_s:.2f}s)"
        else:
            msg = "mode=teleop Teleop ENABLED (policy disabled)"
        self.get_logger().info(msg)
        response.success = True
        response.message = msg
        return response

    def _on_status(self, _request, response):
        response.success = True
        response.message = f"mode={self._mode_name()}"
        return response


class ROSObsBuffer(Node):
    """Subscribe ROS topics and cache latest observation."""

    def __init__(
        self,
        *,
        joint_topic: str,
        global_topic: str,
        fisheye_topic: str,
        depth_topic: str,
        sync_queue_size: int = 200,
        sub_qos: QoSProfile | None = None,
    ):
        super().__init__("rlt_ros_obs_buffer")
        self._lock = threading.Lock()
        self._joint_msg: JointState | None = None
        self._global_msg: ROSImage | None = None
        self._fisheye_msg: ROSImage | None = None
        self._depth_msg: ROSImage | None = None
        queue_size = max(int(sync_queue_size), 2)
        self._joint_queue: deque[JointState] = deque(maxlen=queue_size)
        self._global_queue: deque[ROSImage] = deque(maxlen=queue_size)
        self._fisheye_queue: deque[ROSImage] = deque(maxlen=queue_size)
        self._depth_queue: deque[ROSImage] = deque(maxlen=queue_size)
        self._last_wait_log_ts = 0.0
        qos = sub_qos or qos_profile_sensor_data

        self.create_subscription(JointState, joint_topic, self._on_joint, qos)
        self.create_subscription(ROSImage, global_topic, self._on_global, qos)
        self.create_subscription(ROSImage, fisheye_topic, self._on_fisheye, qos)
        self.create_subscription(ROSImage, depth_topic, self._on_depth, qos)

    def _on_joint(self, msg: JointState) -> None:
        with self._lock:
            self._joint_msg = msg
            self._joint_queue.append(msg)

    def _on_global(self, msg: ROSImage) -> None:
        with self._lock:
            self._global_msg = msg
            self._global_queue.append(msg)

    def _on_fisheye(self, msg: ROSImage) -> None:
        with self._lock:
            self._fisheye_msg = msg
            self._fisheye_queue.append(msg)

    def _on_depth(self, msg: ROSImage) -> None:
        with self._lock:
            self._depth_msg = msg
            self._depth_queue.append(msg)

    def snapshot(self) -> tuple[JointState | None, ROSImage | None, ROSImage | None, ROSImage | None]:
        with self._lock:
            return self._joint_msg, self._global_msg, self._fisheye_msg, self._depth_msg

    @staticmethod
    def _msg_stamp_sec(msg: JointState | ROSImage) -> float:
        return _ros_stamp_to_sec(getattr(msg.header, "stamp", None))

    def aligned_snapshot(self) -> tuple[JointState, ROSImage, ROSImage, ROSImage] | None:
        with self._lock:
            if not (self._joint_queue and self._global_queue and self._fisheye_queue and self._depth_queue):
                return None

            joint_latest = self._msg_stamp_sec(self._joint_queue[-1])
            global_latest = self._msg_stamp_sec(self._global_queue[-1])
            fisheye_latest = self._msg_stamp_sec(self._fisheye_queue[-1])
            depth_latest = self._msg_stamp_sec(self._depth_queue[-1])
            frame_time = min(joint_latest, global_latest, fisheye_latest, depth_latest)

            queues: list[deque[Any]] = [
                self._joint_queue,
                self._global_queue,
                self._fisheye_queue,
                self._depth_queue,
            ]
            aligned: list[Any] = []
            for q in queues:
                while len(q) > 1 and self._msg_stamp_sec(q[0]) < frame_time:
                    q.popleft()
                if self._msg_stamp_sec(q[0]) < frame_time:
                    return None
                aligned.append(q[0])
            return aligned[0], aligned[1], aligned[2], aligned[3]

    def wait_ready(self, timeout_s: float | None = None) -> None:
        start = time.time()
        while rclpy.ok():
            snap = self.snapshot()
            if not _missing_observation_fields(*snap):
                return
            now = time.time()
            if now - self._last_wait_log_ts >= 2.0:
                self._last_wait_log_ts = now
                self.get_logger().warning(
                    f"Waiting for ROS observations, missing: {_missing_observation_fields(*snap)}"
                )
            if timeout_s is not None and (now - start) > timeout_s:
                raise RuntimeError(f"Timeout waiting ROS topics, missing: {_missing_observation_fields(*snap)}")
            time.sleep(0.02)


class ROSCommandPublisher(Node):
    """Publish action to `/joint_states_gripper` (6 joints rad + gripper m)."""

    def __init__(self, cmd_topic: str):
        super().__init__("rlt_ros_cmd_publisher")
        self._pub = self.create_publisher(JointState, cmd_topic, 10)

    def publish_action(self, q6_rad: np.ndarray, g_m: float, grip_effort: float = 1.0) -> None:
        q6 = np.asarray(q6_rad, dtype=np.float64).reshape(-1)
        if q6.shape[0] < 6:
            raise ValueError(f"q6_rad must have at least 6 elements, got {q6.shape[0]}")

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(x) for x in q6[:6]] + [float(g_m)]
        msg.effort = [0.0] * 6 + [float(grip_effort)]
        self._pub.publish(msg)


class HumanActionRecorder(Node):
    """Subscribe the teleop command stream and expose the latest 7D action snapshot."""

    def __init__(self, cmd_topic: str):
        super().__init__("rlt_human_action_recorder")
        self._lock = threading.Lock()
        self._latest_action: np.ndarray | None = None
        self._latest_seq = -1
        self.create_subscription(JointState, cmd_topic, self._on_action, 50)

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


class GripperCtrlStreamer(Node):
    """Continuously publish `data_msgs/Gripper` unless teleop temporarily pauses it."""

    def __init__(self, ctrl_topic: str, ctrl_rate_hz: float):
        if GripperMsg is None:
            raise RuntimeError("data_msgs.msg.Gripper is unavailable")
        super().__init__("rlt_gripper_ctrl_streamer")
        self._pub = self.create_publisher(GripperMsg, ctrl_topic, 10)
        self._lock = threading.Lock()
        self._enabled = True
        self._publishing_active = True
        self._distance = 0.08
        self._velocity = 0.0
        period = 1.0 / max(float(ctrl_rate_hz), 1e-6)
        self._timer = self.create_timer(period, self._on_timer)

    def set_target(self, distance_m: float, velocity: float = 0.0, enable: bool = True) -> None:
        d = min(max(float(distance_m), 0.0), 0.097)
        with self._lock:
            self._distance = d
            self._velocity = float(velocity)
            self._enabled = bool(enable)

    def pause_stream(self) -> None:
        with self._lock:
            self._publishing_active = False

    def resume_stream(self) -> None:
        with self._lock:
            self._publishing_active = True

    def _on_timer(self) -> None:
        with self._lock:
            publishing_active = self._publishing_active
            enable = self._enabled
            distance = self._distance
            velocity = self._velocity

        if not publishing_active:
            return

        msg = GripperMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.enable = enable
        msg.set_zero = False
        msg.distance = distance
        msg.velocity = velocity
        self._pub.publish(msg)


class PikaRobotROSBridge:
    """Bridge from ROS topics to EnvDriver-style observation/action interfaces."""

    def __init__(
        self,
        args: argparse.Namespace,
        obs_node: ROSObsBuffer,
        cmd_node: ROSCommandPublisher,
        gripper_streamer: GripperCtrlStreamer | None,
    ):
        self._args = args
        self._obs_node = obs_node
        self._cmd_node = cmd_node
        self._gripper_streamer = gripper_streamer

    def shutdown(self) -> None:
        return

    def set_policy_control_active(self, enabled: bool) -> None:
        if self._gripper_streamer is None:
            return
        if enabled:
            self._gripper_streamer.resume_stream()
        else:
            self._gripper_streamer.pause_stream()

    def wait_for_observation_ready(self, timeout_s: float | None = None) -> None:
        self._obs_node.wait_ready(timeout_s=timeout_s)

    def get_observation(self, resize_hw: tuple[int, int], task: str) -> dict[str, Any]:
        retries = max(int(self._args.capture_retries), 1)
        snap = self._obs_node.snapshot() if self._args.disable_obs_stamp_align else self._obs_node.aligned_snapshot()
        for _ in range(retries):
            if snap is not None and not _missing_observation_fields(*snap):
                break
            time.sleep(self._args.capture_retry_sleep_s)
            snap = (
                self._obs_node.snapshot() if self._args.disable_obs_stamp_align else self._obs_node.aligned_snapshot()
            )

        if snap is None:
            latest = self._obs_node.snapshot()
            missing = _missing_observation_fields(*latest)
            if not missing:
                snap = latest
        else:
            missing = _missing_observation_fields(*snap)
        if missing:
            raise RuntimeError(f"Failed to collect observation after {retries} retries. Missing: {missing}")

        js_msg, global_msg, fisheye_msg, depth_msg = snap
        pos = np.asarray(js_msg.position, dtype=np.float32).reshape(-1)
        if pos.shape[0] < 7:
            raise RuntimeError(f"Expected JointState.position dim >= 7, got {pos.shape}")

        return {
            "state": pos[:7].astype(np.float32),
            "images": {
                "global_camera": _image_msg_to_rgb_u8_hwc(global_msg, resize_hw),
                "pikaGripperFisheyeCamera": _image_msg_to_rgb_u8_hwc(fisheye_msg, resize_hw),
                "pikaGripperDepthCamera": _image_msg_to_rgb_u8_hwc(depth_msg, resize_hw),
            },
            "prompt": task,
        }

    def send_action(self, action7: np.ndarray) -> None:
        action7 = np.asarray(action7, dtype=np.float32).reshape(-1)
        if action7.shape[0] < 7:
            raise ValueError(f"Expected action dim >= 7, got {action7.shape}")

        q_rad = action7[:6]
        gripper_m = float(np.clip(action7[6], 0.0, self._args.max_gripper_m))
        if gripper_m < 0.003:
            gripper_m = 0.0

        self._cmd_node.publish_action(q_rad, gripper_m, grip_effort=1.0)
        if self._gripper_streamer is not None:
            self._gripper_streamer.set_target(gripper_m, velocity=0.0, enable=True)
            self._gripper_streamer.resume_stream()


def _coerce_reward_output(reward: np.ndarray | list[float] | float, executed_steps: int) -> list[float]:
    if executed_steps <= 0:
        return []
    if np.isscalar(reward):
        return [float(reward)] * executed_steps
    reward_array = np.asarray(reward, dtype=np.float32).reshape(-1)
    if reward_array.shape[0] == 1:
        return [float(reward_array[0])] * executed_steps
    if reward_array.shape[0] != executed_steps:
        raise ValueError(
            f"Reward callback must return scalar or length {executed_steps}, got shape {reward_array.shape}."
        )
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


class PikaChunkEnvAdapter:
    """Robot-specific env adapter consumed by the generic EnvDriver."""

    def __init__(
        self,
        *,
        system: OnlineRLSystemConfig,
        robot: PikaRobotROSBridge,
        task_state: TaskState,
        intervention_state: HumanInterventionState,
        human_action_recorder: HumanActionRecorder,
        phase_controller: RolloutPhaseController,
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

    def reset(self) -> dict[str, Any]:
        self._episode_chunk_step = 0
        self._last_sent_action = None
        latest_human_action, latest_human_seq = self._human_action_recorder.snapshot_latest()
        self._last_human_seq = latest_human_seq
        self._last_human_action = (
            None if latest_human_action is None else latest_human_action.astype(np.float32, copy=False)
        )
        self._intervention_state.enter_episode_reset()
        self._runtime_context.reset_episode_state()
        self._robot.wait_for_observation_ready(timeout_s=self._obs_ready_timeout_s)
        self._reset_robot_to_mode_start()
        self._phase_controller.begin_episode()
        logger.info("Waiting for next episode request task_mode=%s", self._task_mode)
        self._runtime_context.wait_for_next_episode_request()
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
        control_hz: float,
        policy_planner: Callable[[dict[str, Any], int], PolicyPlan] | None = None,
    ) -> tuple[dict[str, Any], list[float], bool, dict[str, Any]]:
        self._phase_controller.observe_progress()
        phase = self._phase_controller.episode_phase
        critical_started = self._runtime_context.in_critical_phase()
        period = 1.0 / max(float(control_hz), 1e-6)
        horizon = int(self._system.env_driver.chunk_exec_horizon)

        observation = self._robot.get_observation(self._resize_hw, self._task_state.get())
        executed: list[np.ndarray] = []
        ref_actions: list[np.ndarray] = []
        human_controlled: list[bool] = []
        step_sources: list[int] = []
        actor_param_versions: list[int] = []
        step_observations: list[dict[str, Any]] = [observation]
        policy_anchor_offsets: list[int] = []
        policy_anchor_features: list[ChunkFeatures] = []
        chunk_start_features = None
        current_plan: PolicyPlan | None = None
        plan_cursor = 0

        for local_step in range(horizon):
            if self._manual_terminal_requested():
                break
            step_observation = step_observations[-1]
            tick_start = time.perf_counter()
            policy_enabled = bool(
                self._intervention_state.is_policy_enabled() and not self._intervention_state.in_resume_cooldown()
            )
            if not policy_enabled:
                current_plan = None
                plan_cursor = 0

            if policy_enabled and policy_planner is not None:
                if current_plan is None or plan_cursor >= current_plan.action_chunk.shape[0]:
                    current_plan = policy_planner(step_observation, local_step)
                    plan_cursor = 0
                    if current_plan.source != int(TransitionSource.HUMAN):
                        if local_step == 0:
                            chunk_start_features = current_plan.start_features
                        else:
                            policy_anchor_offsets.append(local_step)
                            policy_anchor_features.append(current_plan.start_features)

            if policy_enabled and current_plan is not None and plan_cursor < current_plan.action_chunk.shape[0]:
                raw_action = np.asarray(
                    current_plan.action_chunk[plan_cursor],
                    dtype=np.float32,
                )[: self._system.rl.action_dim]
                bounded = self._apply_action_limits(raw_action)
                self._robot.send_action(bounded)
                executed.append(bounded)
                ref_actions.append(
                    np.asarray(current_plan.ref_chunk[plan_cursor], dtype=np.float32)[: self._system.rl.action_dim]
                )
                human_controlled.append(bool(current_plan.source == int(TransitionSource.HUMAN)))
                step_sources.append(int(current_plan.source))
                actor_param_versions.append(int(current_plan.actor_param_version))
                plan_cursor += 1
            else:
                human_action = self._sample_latest_human_action(step_observation)
                executed.append(human_action)
                ref_actions.append(human_action.copy())
                human_controlled.append(True)
                step_sources.append(int(TransitionSource.HUMAN))
                actor_param_versions.append(-1)
            elapsed = time.perf_counter() - tick_start
            remaining = period - elapsed
            if remaining > 0:
                time.sleep(remaining)
            step_observations.append(self._robot.get_observation(self._resize_hw, self._task_state.get()))

        next_observation = (
            step_observations[-1]
            if step_observations
            else self._robot.get_observation(self._resize_hw, self._task_state.get())
        )
        signal_snapshot = self._runtime_context.snapshot_signals()
        context = {
            "episode_chunk_step": self._episode_chunk_step,
            "executed_steps": len(executed),
            "interrupted": bool(any(human_controlled)),
            "phase": phase,
            "task_mode": self._task_mode,
            "critical_started": critical_started,
            "runtime": self._runtime_context,
            "signals": signal_snapshot,
        }
        rewards = _coerce_reward_output(
            self._reward_fn(observation, np.asarray(executed, dtype=np.float32), next_observation, context),
            executed_steps=len(executed),
        )
        success = int(bool(self._success_fn(observation, next_observation, context)))
        if success and rewards and not any(float(reward) > 0.0 for reward in rewards):
            rewards[-1] = 1.0
            logger.warning(
                "Success recorded without a positive reward in the executed chunk; forcing terminal reward on the last executed step."
            )
        manual_done = bool(self._done_fn(observation, next_observation, context))
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
                self._intervention_state.enter_episode_reset()
                self._phase_controller.finish_episode()
            return (
                next_observation,
                rewards,
                terminal_requested,
                {
                    "drop_transition": True,
                    "intervention_flag": False,
                    "source": int(TransitionSource.HUMAN),
                    "success": success,
                    "step_trace": step_trace,
                    "policy_anchor_offsets": policy_anchor_offsets,
                    "policy_anchor_features": policy_anchor_features,
                    "chunk_start_features": chunk_start_features,
                },
            )

        self._episode_chunk_step += 1
        done = bool(terminal_requested or (self._episode_chunk_step >= self._max_chunk_steps_per_episode))
        if done:
            self._intervention_state.enter_episode_reset()
            self._phase_controller.finish_episode()
        if step_trace:
            step_trace[-1]["done"] = done
        if human_intervened:
            source = int(
                TransitionSource.MIXED if any(not flag for flag in human_controlled) else TransitionSource.HUMAN
            )
        else:
            source = int(step_sources[0])
        if not critical_started:
            return (
                next_observation,
                rewards,
                done,
                {
                    "drop_transition": True,
                    "intervention_flag": human_intervened,
                    "source": source,
                    "success": success,
                    "step_trace": step_trace,
                    "policy_anchor_offsets": policy_anchor_offsets,
                    "policy_anchor_features": policy_anchor_features,
                    "chunk_start_features": chunk_start_features,
                },
            )
        return (
            next_observation,
            rewards,
            done,
            {
                "success": success,
                "intervention_flag": human_intervened,
                "source": source,
                "step_trace": step_trace,
                "policy_anchor_offsets": policy_anchor_offsets,
                "policy_anchor_features": policy_anchor_features,
                "chunk_start_features": chunk_start_features,
            },
        )

    def _wait_until_policy_active(self) -> None:
        while rclpy.ok():
            if self._intervention_state.consume_reset_request():
                self._last_sent_action = None
                logging.info("Policy resumed: runtime state cleared.")
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
        logger.info(
            "Episode start control mode=%s resolved=%s",
            start_mode,
            "policy" if resolved_mode else "human",
        )
        if resolved_mode:
            self._wait_until_policy_active()

    def _consume_manual_terminal_events(self, signal_snapshot: dict[str, Any]) -> None:
        to_clear = [
            name
            for name in (
                SIGNAL_MANUAL_SUCCESS_PENDING,
                SIGNAL_MANUAL_FAILURE_PENDING,
                SIGNAL_MANUAL_DONE_PENDING,
            )
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
        target = self._reset_target_for_mode()
        if target is None:
            logger.info("No reset action configured for task_mode=%s; skipping commanded reset.", self._task_mode)
            return

        observation = self._robot.get_observation(self._resize_hw, self._task_state.get())
        start = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
        if start.shape[0] < target.shape[0]:
            raise RuntimeError(f"Reset observation dim {start.shape[0]} is smaller than target dim {target.shape[0]}.")
        start = start[: target.shape[0]]

        reset_timeout_s = 8.0
        reset_interp_duration_s = 2.0
        reset_control_hz = 20.0
        reset_sleep_s = 1.0 / reset_control_hz
        reset_steps = max(int(reset_interp_duration_s * reset_control_hz), 1)
        reset_joint_tol = 0.1
        reset_gripper_tol = 0.0001

        def reset_error(state: np.ndarray) -> tuple[float, float]:
            state7 = np.asarray(state, dtype=np.float32).reshape(-1)[: target.shape[0]]
            joint_err = float(np.max(np.abs(state7[:6] - target[:6])))
            gripper_err = float(abs(state7[6] - target[6]))
            return joint_err, gripper_err

        def reset_reached(state: np.ndarray) -> tuple[bool, float, float]:
            joint_err, gripper_err = reset_error(state)
            reached = joint_err <= reset_joint_tol and gripper_err <= reset_gripper_tol
            return reached, joint_err, gripper_err

        reached, joint_err, gripper_err = reset_reached(start)
        if reached:
            logger.info(
                "Reset already within tolerance for task_mode=%s joint_err=%.4f gripper_err=%.4f",
                self._task_mode,
                joint_err,
                gripper_err,
            )
            time.sleep(0.2)
            return

        logger.info(
            "Resetting task_mode=%s with linear interpolation duration=%.2fs steps=%s",
            self._task_mode,
            reset_interp_duration_s,
            reset_steps,
        )

        deadline = time.time() + reset_timeout_s
        for step in range(1, reset_steps + 1):
            if not rclpy.ok() or time.time() >= deadline:
                break
            alpha = float(step) / float(reset_steps)
            waypoint = (1.0 - alpha) * start + alpha * target
            self._robot.send_action(waypoint)
            time.sleep(reset_sleep_s)
            observation = self._robot.get_observation(self._resize_hw, self._task_state.get())
            state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            reached, joint_err, gripper_err = reset_reached(state)
            if reached:
                break

        while not reached and rclpy.ok() and time.time() < deadline:
            self._robot.send_action(target)
            time.sleep(reset_sleep_s)
            observation = self._robot.get_observation(self._resize_hw, self._task_state.get())
            state = np.asarray(observation["state"], dtype=np.float32).reshape(-1)
            reached, joint_err, gripper_err = reset_reached(state)
            if reached:
                break

        if not reached:
            logger.warning(
                "Reset target not reached within timeout for task_mode=%s joint_err=%.4f gripper_err=%.4f",
                self._task_mode,
                joint_err,
                gripper_err,
            )
        else:
            logger.info(
                "Reset reached target for task_mode=%s joint_err=%.4f gripper_err=%.4f",
                self._task_mode,
                joint_err,
                gripper_err,
            )
        time.sleep(0.5)

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
    default_config = REPO_ROOT / "configs" / "tasks" / "agilex_ethernet" / "online_rl.yaml"
    parser = argparse.ArgumentParser(description="ROS real-robot runner for OpenPI RLT online RL.")
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--task", type=str, default="insert the Ethernet cable into the port")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_chunk_steps_per_episode", type=int, default=200)
    parser.add_argument("--idle_sleep_sec", type=float, default=0.02)
    parser.add_argument("--machine_a_ws_url", type=str, default=None)
    parser.add_argument("--actor_service_url", type=str, default=None)
    parser.add_argument("--replay_service_url", type=str, default=None)
    parser.add_argument("--reward_factory", type=str, default=None)
    parser.add_argument("--success_factory", type=str, default=None)
    parser.add_argument("--done_factory", type=str, default=None)
    parser.add_argument("--safe_action_filter_factory", type=str, default=None)
    parser.add_argument(
        "--action_delta_limits",
        type=float,
        nargs=7,
        default=None,
        help="Optional per-dim delta clip for 6 joints + gripper.",
    )

    parser.add_argument("--image_h", type=int, default=224)
    parser.add_argument("--image_w", type=int, default=224)
    parser.add_argument("--capture_retries", type=int, default=30)
    parser.add_argument("--capture_retry_sleep_s", type=float, default=0.01)
    parser.add_argument("--capture_like_subscriber", action="store_true")
    parser.add_argument("--obs_sub_queue_depth", type=int, default=2000)
    parser.add_argument("--capture_like_no_align", action="store_true")
    parser.add_argument("--disable_obs_stamp_align", action="store_true")
    parser.add_argument("--obs_align_queue_size", type=int, default=200)
    parser.add_argument("--max_gripper_m", type=float, default=0.097)
    parser.add_argument("--joint_topic", type=str, default="/joint_states_single_gripper")
    parser.add_argument("--global_image_topic", type=str, default="/global_camera/camera/color/image_raw")
    parser.add_argument("--fisheye_image_topic", type=str, default="/gripper/camera_fisheye/color/image_raw")
    parser.add_argument("--depth_image_topic", type=str, default="/gripper/camera/color/image_raw")
    parser.add_argument("--cmd_topic", type=str, default="/joint_states_gripper")
    parser.add_argument("--disable_gripper_stream", action="store_true")
    parser.add_argument("--gripper_ctrl_topic", type=str, default="/gripper/gripper/ctrl")
    parser.add_argument("--gripper_ctrl_rate_hz", type=float, default=20.0)
    parser.add_argument("--teleop_trigger_service", type=str, default="/teleop_trigger_rl")
    parser.add_argument("--policy_resume_delay_s", type=float, default=1.0)
    parser.add_argument("--start_in_human_mode", action="store_true")
    parser.add_argument("--obs_ready_timeout_s", type=float, default=None)
    parser.add_argument(
        "--step_trace_stride",
        type=int,
        default=None,
        help="Optional replay stride override. 0 disables dense replay; 2/3/4/... build dense replay at episode end.",
    )
    parser.add_argument(
        "--eval_actor_only",
        action="store_true",
        help="Run rollout in eval mode: actor inference only, no replay/learner dependency.",
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
    log_path = setup_process_logging("pika_sync_ros", system, console_level=logging.INFO)

    reward_fn = _load_callable(args.reward_factory) or _default_reward_fn
    success_fn = _load_callable(args.success_factory) or _default_success_fn
    done_fn = _load_callable(args.done_factory) or _default_done_fn
    safe_action_filter = _load_callable(args.safe_action_filter_factory)

    task_state = TaskState(args.task)
    intervention_state = HumanInterventionState(policy_enabled=not args.start_in_human_mode)

    rclpy.init()
    if args.capture_like_no_align:
        args.disable_obs_stamp_align = True

    obs_sub_qos = _build_obs_subscription_qos(
        capture_like=args.capture_like_subscriber,
        depth=args.obs_sub_queue_depth,
    )
    obs_node = ROSObsBuffer(
        joint_topic=args.joint_topic,
        global_topic=args.global_image_topic,
        fisheye_topic=args.fisheye_image_topic,
        depth_topic=args.depth_image_topic,
        sync_queue_size=args.obs_align_queue_size,
        sub_qos=obs_sub_qos,
    )
    gripper_streamer = None
    if not args.disable_gripper_stream:
        if GripperMsg is None:
            raise RuntimeError(
                "data_msgs.msg.Gripper is unavailable, but gripper stream is enabled by default. "
                "Install the message package or pass --disable_gripper_stream."
            )
        gripper_streamer = GripperCtrlStreamer(
            ctrl_topic=args.gripper_ctrl_topic,
            ctrl_rate_hz=args.gripper_ctrl_rate_hz,
        )

    cmd_node = ROSCommandPublisher(cmd_topic=args.cmd_topic)
    human_action_recorder = HumanActionRecorder(cmd_topic=args.cmd_topic)
    teleop_node = TeleopTriggerNode(
        intervention_state=intervention_state,
        service_name=args.teleop_trigger_service,
        resume_delay_s=args.policy_resume_delay_s,
        gripper_streamer=gripper_streamer,
    )

    nodes: list[Node] = [obs_node, cmd_node, human_action_recorder, teleop_node]
    if gripper_streamer is not None:
        nodes.append(gripper_streamer)

    robot = PikaRobotROSBridge(args, obs_node, cmd_node, gripper_streamer)
    runtime_context = RolloutRuntimeContext(
        system=system,
        obs_node=obs_node,
        task_state=task_state,
        intervention_state=intervention_state,
        robot=robot,
    )
    manual_signal_bridge = ManualSignalBridge()
    nodes.extend(_bind_runtime_hook(manual_signal_bridge, runtime_context))
    nodes.extend(_bind_runtime_hook(reward_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(success_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(done_fn, runtime_context))
    nodes.extend(_bind_runtime_hook(safe_action_filter, runtime_context))

    executor = MultiThreadedExecutor()
    for node in nodes:
        executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
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
    actor_client = PhaseAwareActorClient(
        base_actor_client,
        phase_controller,
        runtime_context,
    )
    env = PikaChunkEnvAdapter(
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

    logger.info("Starting robot rollout log=%s config=%s", log_path, args.config)
    logger.info("Machine A ws: %s", system.env_driver.machine_a_ws_url)
    logger.info(
        "Machine A timeouts connect=%.2fs recv=%.2fs retry_interval=%.2fs",
        system.env_driver.machine_a_connect_timeout_sec,
        system.env_driver.machine_a_recv_timeout_sec,
        system.env_driver.machine_a_retry_interval_sec,
    )
    logger.info("Actor service: %s", system.env_driver.actor_service_url)
    logger.info("Replay service: %s", system.env_driver.replay_service_url)
    logger.info("Robot control hz: %.2f", system.env_driver.control_frequency_hz)
    logger.info("Task mode: %s", system.env_driver.task_mode)
    logger.info("Eval actor only: %s", args.eval_actor_only)
    logger.info("Actor deterministic: %s", system.env_driver.actor_deterministic)
    logger.info("Step trace stride: %s", system.env_driver.step_trace_stride)
    logger.info("Min online actor version: %s", min_online_actor_version)
    logger.info("Learner status path: %s", learner_status_path)
    logger.info(
        "Manual services next=%s success=%s failure=%s done=%s critical=%s toggle_critical=%s select_actor=%s select_base=%s",
        REQUEST_NEXT_EPISODE_SERVICE,
        RECORD_SUCCESS_SERVICE,
        RECORD_FAILURE_SERVICE,
        RECORD_DONE_SERVICE,
        ENTER_CRITICAL_PHASE_SERVICE,
        TOGGLE_CRITICAL_PHASE_SERVICE,
        SET_CRITICAL_POLICY_ACTOR_SERVICE,
        SET_CRITICAL_POLICY_BASE_SERVICE,
    )

    try:
        driver.run_forever(num_episodes=args.num_episodes)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down.")
    finally:
        robot.shutdown()
        executor.shutdown()
        for node in nodes:
            try:
                node.destroy_node()
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
