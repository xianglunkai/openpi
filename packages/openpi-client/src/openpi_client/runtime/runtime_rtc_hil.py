"""RTC runtime with keyboard / SpaceMouse human-in-the-loop and episode labeling.

Extends RuntimeRTC without modifying runtime_rtc.py.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Event, Lock
from typing import Callable, List, Optional

import numpy as np

from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
from openpi_client import base_policy as _base_policy
from openpi_client.rtc_config import RTCConfig
from openpi_client.runtime.runtime_rtc import RuntimeRTC
from openpi_client.teleop.eef_action import EefActionConverter, teleop_eef_to_actions
from openpi_client.teleop.kinematics import EefKinematics, EefKinematicsConfig
from openpi_client.teleop.spacemouse_teleop import SpacemouseTeleop, SpacemouseTeleopConfig
from openpi_client.teleop.teleoperator import Teleoperator
from openpi_client.time_utils import precise_sleep

from openpi_client.runtime.lerobot_remote import (
    LeRobotRemoteRecorderClient,
    build_lerobot_features,
    build_lerobot_frame,
)
from openpi_client.speech_utils import log_say

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = False
try:
    from pynput import keyboard

    PYNPUT_AVAILABLE = True
except Exception:
    keyboard = None  # type: ignore[assignment,misc]


class EpisodeEndStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    RERECORD = "rerecord_episode"
    NEXT_EPISODE = "next_episode"
    ABORT_ALL = "abort_all"
    END = "end"


@dataclass(frozen=True)
class EpisodeOutcome:
    status: EpisodeEndStatus
    reward: float


def _default_hil_repo_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"openpi_hil/hil_screw_sorting_{stamp}"


@dataclass(frozen=True)
class LeRobotStorageConfig:
    """LeRobot v2.1 recording config (``lerobot.common.datasets`` API).

    Matches the dataset layout used by ``examples/mobile_aloha_AgileX/convert_aloha_data_to_lerobot.py``
    at the pinned lerobot revision in ``pyproject.toml`` (``codebase_version``: ``v2.1``).

    Set ``enabled=True`` and customize ``repo_id`` / ``task`` to persist HIL rollouts.
    """

    enabled: bool = True
    repo_id: str = field(default_factory=_default_hil_repo_id)
    task: str = "Please sort and return the silver screws in the grey box to their proper places."
    root: Optional[str] = None
    robot_type: str = "openpi_hil"
    fps: Optional[int] = None
    use_videos: bool = False
    mode: str = "image"
    tolerance_s: float = 1e-4
    image_writer_processes: int = 0
    image_writer_threads: int = 4
    video_backend: Optional[str] = None
    clear_existing: bool = False
    push_to_hub: bool = False
    private: bool = False
    tags: Optional[List[str]] = None
    state_names: Optional[List[str]] = None
    action_names: Optional[List[str]] = None
    # Match lerobot HIL: skip interpolated control steps unless recording every sub-step.
    record_interpolated_actions: bool = False
    # When set (e.g. ``tcp://127.0.0.1:8765``), send frames to ``lerobot_recorder_service.py``
    # running in the openpi environment instead of importing lerobot in the robot client venv.
    remote_endpoint: Optional[str] = "tcp://127.0.0.1:8765"


def reward_for_status(status: EpisodeEndStatus) -> float:
    if status == EpisodeEndStatus.SUCCESS:
        return 1.0
    return 0.0


class EpisodeCommandState:
    """Thread-safe episode / intervention flags."""

    def __init__(self) -> None:
        self._lock = Lock()
        self.human_intervention = False
        self.abort_all = False
        self.end_status: Optional[EpisodeEndStatus] = None

    def reset_episode(self) -> None:
        """Reset per-episode flags; ``abort_all`` persists until ``run()`` ends."""
        with self._lock:
            self.human_intervention = False
            self.end_status = None

    def toggle_intervention(self) -> bool:
        with self._lock:
            self.human_intervention = not self.human_intervention
            return self.human_intervention

    def request_end(self, status: EpisodeEndStatus) -> None:
        with self._lock:
            if status == EpisodeEndStatus.ABORT_ALL:
                self.abort_all = True
            self.end_status = status

    def consume_end_status(self) -> Optional[EpisodeEndStatus]:
        with self._lock:
            status = self.end_status
            self.end_status = None
            return status

    def peek_end_status(self) -> Optional[EpisodeEndStatus]:
        with self._lock:
            return self.end_status

    def should_abort_all(self) -> bool:
        with self._lock:
            return self.abort_all

    def is_intervention_active(self) -> bool:
        with self._lock:
            return self.human_intervention


class KeyboardListener:
    """Keyboard listener for episode control and intervention toggle."""

    def __init__(self, state: EpisodeCommandState) -> None:
        self._state = state
        self._listener = None
        self._lock = Lock()

    def start(self) -> bool:
        if not PYNPUT_AVAILABLE:
            print("==================== pynput unavailable; keyboard episode control disabled.==========================")
            return False

        def on_press(key):
            with self._lock:
                self._handle_key(key)

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        self._print_help()
        print("[KeyboardListener] Start keyboard listener success!")
        return True

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @staticmethod
    def _print_help() -> None:
        print(
            "Keyboard controls:\n"
            "  Esc        abort all episodes\n"
            "  Right      next episode\n"
            "  Left / r   rerecord current episode\n"
            "  s          success (+reward 1.0)\n"
            "  f          failure\n"
            "  Space / i  toggle human intervention (policy <-> teleop)\n"
        )

    def _handle_key(self, key) -> None:
        try:
            if key == keyboard.Key.esc:
                self._state.request_end(EpisodeEndStatus.ABORT_ALL)
                print("Esc pressed: aborting all episodes.")
                return
            if key == keyboard.Key.right:
                self._state.request_end(EpisodeEndStatus.NEXT_EPISODE)
                return
            if key == keyboard.Key.left:
                self._state.request_end(EpisodeEndStatus.RERECORD)
                return
            if key == keyboard.Key.space:
                active = self._state.toggle_intervention()
                print("Human intervention %s", "ON" if active else "OFF")
                return
            if hasattr(key, "char") and key.char:
                ch = key.char.lower()
                if ch == "s":
                    self._state.request_end(EpisodeEndStatus.SUCCESS)
                elif ch == "f":
                    self._state.request_end(EpisodeEndStatus.FAILURE)
                elif ch == "r":
                    self._state.request_end(EpisodeEndStatus.RERECORD)
                elif ch == "i":
                    active = self._state.toggle_intervention()
                    print("Human intervention %s", "ON" if active else "OFF")
        except Exception:
            print("keyboard handler error")


TeleopActionFn = Callable[[dict], Optional[dict]]


def _copy_observation(obs: dict) -> dict:
    """Shallow copy with numpy payloads duplicated for cross-thread inference."""
    out = dict(obs)
    if "state" in out:
        out["state"] = np.asarray(out["state"]).copy()
    if "images" in out:
        out["images"] = {k: np.asarray(v).copy() for k, v in out["images"].items()}
    if "actions" in out:
        out["actions"] = np.asarray(out["actions"]).copy()
    return out


class RuntimeRTCHil(RuntimeRTC):
    """RuntimeRTC + keyboard episode control + optional SpaceMouse teleop.

    Lifecycle flags (see also ``EpisodeCommandState`` for keyboard/HIL input):

    - ``_shutdown_event`` (from ``ProcessSignalHandler``): entire ``run()`` session ends.
      Workers exit their loops when set. Only ``run()`` finally should set it.

    - ``_episode_active``: current episode rollout is in progress. Main thread sets/clears
      around ``_run_episode``; workers idle when clear (between episodes).

    - ``_policy_active``: RTC policy inference is allowed. Cleared during human
      intervention; set when resuming policy within an active episode. GetActionThread
      requires both ``_episode_active`` and ``_policy_active``.

    Intervention is derived at runtime via ``_is_intervention_active()`` (keyboard flag
    and/or SpaceMouse motion); it is not a separate Event.
    """

    def __init__(
        self,
        environment: _environment.Environment,
        policy: _base_policy.BasePolicy,
        subscribers: List[_subscriber.Subscriber],
        *,
        fps: float = 30.0,
        num_episodes: int = 1,
        max_episode_time_s: int = 0,
        use_action_interpolation: bool = False,
        multiplier: int = 1,
        rtc_config: Optional[RTCConfig] = None,
        action_queue_size_to_get_new_actions: int = 30,
        enable_keyboard: bool = True,
        enable_teleop: bool = True,
        teleop: Optional[Teleoperator] = None,
        teleop_action_fn: Optional[TeleopActionFn] = None,
        eef_kinematics: Optional[EefKinematics] = None,
        eef_kinematics_config: Optional[EefKinematicsConfig] = None,
        spacemouse_config: Optional[SpacemouseTeleopConfig] = None,
        get_joint_positions: Optional[Callable[[], np.ndarray]] = None,
        spacemouse_auto_intervene: bool = True,
        inference_start_delay_s: float = 0.1,
        monitor_interval_s: float = 0.1,
        lerobot_storage: Optional[LeRobotStorageConfig] = None,
        play_sounds: bool = True,
    ) -> None:
        super().__init__(
            environment=environment,
            policy=policy,
            subscribers=subscribers,
            fps=fps,
            num_episodes=num_episodes,
            max_episode_time_s=max_episode_time_s,
            use_action_interpolation=use_action_interpolation,
            multiplier=multiplier,
            rtc_config=rtc_config,
            action_queue_size_to_get_new_actions=action_queue_size_to_get_new_actions,
        )
        self._commands = EpisodeCommandState()
        self._keyboard = KeyboardListener(self._commands) if enable_keyboard else None
        self._spacemouse_auto_intervene = spacemouse_auto_intervene
        self._inference_start_delay_s = inference_start_delay_s
        self._monitor_interval_s = monitor_interval_s
        self._last_outcome: Optional[EpisodeOutcome] = None
        self._obs_lock = Lock()
        self._episode_active = Event()
        self._policy_active = Event()
        self._shared_observation: Optional[dict] = None
        self._cached_observation: Optional[dict] = None
        self._last_obs_poll_t = 0.0
        self._obs_poll_interval = 1.0 / max(self._fps, 1e-6)
        self._warned_intervention_no_teleop = False
        self._actor_thread: Optional[threading.Thread] = None
        self._get_action_thread: Optional[threading.Thread] = None
        self._lerobot_storage_cfg = lerobot_storage if lerobot_storage is not None else LeRobotStorageConfig()
        self._play_sounds = play_sounds
        self._lerobot_dataset = None
        self._lerobot_remote: Optional[LeRobotRemoteRecorderClient] = None
        if self._lerobot_storage_cfg.remote_endpoint:
            self._lerobot_remote = LeRobotRemoteRecorderClient(self._lerobot_storage_cfg.remote_endpoint)
        self._lerobot_frame_count = 0
        self._current_episode_idx = 0

        self._setup_hil_teleop_stack(
            enable_teleop=enable_teleop,
            teleop=teleop,
            teleop_action_fn=teleop_action_fn,
            eef_kinematics=eef_kinematics,
            eef_kinematics_config=eef_kinematics_config,
            spacemouse_config=spacemouse_config,
            get_joint_positions=get_joint_positions,
        )

    def _setup_hil_teleop_stack(
        self,
        *,
        enable_teleop: bool,
        teleop: Optional[Teleoperator],
        teleop_action_fn: Optional[TeleopActionFn],
        eef_kinematics: Optional[EefKinematics],
        eef_kinematics_config: Optional[EefKinematicsConfig],
        spacemouse_config: Optional[SpacemouseTeleopConfig],
        get_joint_positions: Optional[Callable[[], np.ndarray]],
    ) -> None:
        if teleop_action_fn is not None:
            self._teleop = teleop
            self._teleop_action_fn = teleop_action_fn
            return

        if not enable_teleop:
            self._teleop = teleop
            self._teleop_action_fn = None
            return

        self._teleop = teleop if teleop is not None else SpacemouseTeleop(spacemouse_config or SpacemouseTeleopConfig())

        kinematics = eef_kinematics
        if kinematics is None:
            try:
                kinematics = EefKinematics.from_config(eef_kinematics_config or EefKinematicsConfig())
            except Exception as exc:
                logger.warning("Default EefKinematics unavailable; teleop IK disabled: %s", exc)

        joint_fn = get_joint_positions or self._default_get_joint_positions
        if kinematics is not None:
            self._teleop_action_fn = EefActionConverter(
                kinematics,
                joint_fn,
                input_fps=self._fps,
            )
            print("_teleop_action_fn EefActionConverter set success!")
        else:
            self._teleop_action_fn = None
            print("_teleop_action_fn EefActionConverter set failed!")

    def _default_get_joint_positions(self) -> np.ndarray:
        with self._obs_lock:
            if self._cached_observation is not None and "state" in self._cached_observation:
                return np.asarray(self._cached_observation["state"], dtype=np.float64).reshape(-1)
        obs = self._environment.get_observation()
        return np.asarray(obs["state"], dtype=np.float64).reshape(-1)

    @property
    def last_episode_outcome(self) -> Optional[EpisodeOutcome]:
        return self._last_outcome

    def _say(self, text: str) -> None:
        log_say(text, play_sounds=self._play_sounds)

    def _lerobot_storage_enabled(self) -> bool:
        cfg = self._lerobot_storage_cfg
        return cfg is not None and cfg.enabled

    def _lerobot_storage_is_remote(self) -> bool:
        cfg = self._lerobot_storage_cfg
        return cfg is not None and bool(cfg.remote_endpoint)

    def _disable_lerobot_storage(self, reason: str) -> None:
        logger.warning("LeRobot storage disabled: %s", reason)
        if self._lerobot_storage_cfg is not None:
            self._lerobot_storage_cfg = replace(self._lerobot_storage_cfg, enabled=False)

    @staticmethod
    def _episode_end_message(status: EpisodeEndStatus, episode_idx: int) -> str:
        if status == EpisodeEndStatus.SUCCESS:
            return "Episode success."
        if status == EpisodeEndStatus.FAILURE:
            return "Episode failure."
        if status == EpisodeEndStatus.RERECORD:
            return "Rerecord requested."
        if status == EpisodeEndStatus.NEXT_EPISODE:
            return "Next episode."
        if status == EpisodeEndStatus.ABORT_ALL:
            return "Aborting all episodes."
        return f"Episode {episode_idx + 1} completed."

    def run(self) -> None:
        if self._keyboard is not None:
            if self._keyboard.start():
                self._say("Keyboard controls ready.")
        if self._teleop is not None:
            try:
                self._teleop.connect()
            except Exception as exc:
                print("Teleop connect failed; continuing without SpaceMouse teleop: %s", exc)
                self._teleop = None
                self._teleop_action_fn = None

        self._shutdown_event.clear()
        self._episode_active.clear()
        self._policy_active.clear()
        self._inference_warmup_steps = 0
        self._actor_thread = threading.Thread(
            target=self._actor_control_worker, daemon=True, name="ActorControlThread"
        )
        self._get_action_thread = threading.Thread(
            target=self._get_action_worker, daemon=True, name="GetActionThread"
        )
        self._actor_thread.start()
        time.sleep(self._inference_start_delay_s)
        self._get_action_thread.start()

        episode_idx = 0
        try:
            while episode_idx < self._num_episodes and not self._shutdown_event.is_set():
                outcome = self._run_episode(episode_idx)
                if outcome.status == EpisodeEndStatus.ABORT_ALL or self._commands.should_abort_all():
                    print("All episodes aborted.")
                    self._say("All episodes aborted.")
                    break
                if outcome.status == EpisodeEndStatus.RERECORD:
                    print(f"Rerecording episode {episode_idx + 1}...")
                    self._say(f"Rerecording episode {episode_idx + 1}.")
                    continue
                episode_idx += 1

            self._environment.reset()
            self._log_statistics()
        finally:
            self._episode_active.clear()
            self._policy_active.clear()
            self._shutdown_event.set()
            if self._actor_thread is not None:
                self._actor_thread.join()
            if self._get_action_thread is not None:
                self._get_action_thread.join()
            self._close_lerobot_dataset()
            if self._teleop is not None:
                self._teleop.disconnect()
            if self._keyboard is not None:
                self._keyboard.stop()

    def _reset_episode_state(self, episode_idx: int) -> None:
        """Reset per-episode state without restarting worker threads."""
        self._current_episode_idx = episode_idx
        self._commands.reset_episode()
        self._last_outcome = None

        self._environment.reset()
        self._policy.reset()
        self._action_queue = type(self._action_queue)(enabled=self._rtc_config.enabled)
        self._latency_tracker.reset()
        if self._interpolator:
            self._interpolator.reset()

        self._episode_steps = 0
        self._total_inference_time = 0
        self._total_control_time = 0
        self._inference_count = 0

        with self._obs_lock:
            self._shared_observation = None
        self._cached_observation = None
        self._last_obs_poll_t = 0.0
        self._warned_intervention_no_teleop = False

        for subscriber in self._subscribers:
            subscriber.on_episode_start()

    def _sync_policy_active(self) -> None:
        """Match ``_policy_active`` to whether human intervention is currently active."""
        if self._is_intervention_active():
            self._policy_active.clear()
        else:
            self._policy_active.set()

    def _run_episode(self, episode_idx: int) -> EpisodeOutcome:
        print(f"Starting episode {episode_idx + 1}/{self._num_episodes}...")
        self._say(f"Starting episode {episode_idx + 1} of {self._num_episodes}.")
        self._reset_episode_state(episode_idx)
        self._episode_active.set()
        self._sync_policy_active()

        start_time = time.perf_counter()
        terminal_status: Optional[EpisodeEndStatus] = None
        while self._episode_active.is_set() and not self._shutdown_event.is_set():
            precise_sleep(self._monitor_interval_s)
            if self._shutdown_event.is_set():
                terminal_status = EpisodeEndStatus.ABORT_ALL
                break
            if self._commands.should_abort_all():
                terminal_status = EpisodeEndStatus.ABORT_ALL
                break
            if self._environment.is_episode_complete():
                print("Environment marked episode as complete")
                terminal_status = EpisodeEndStatus.END
                break
            pending = self._commands.peek_end_status()
            if pending is not None:
                terminal_status = pending
                break
            if self._max_episode_time_s > 0 and (time.perf_counter() - start_time) > self._max_episode_time_s:
                terminal_status = EpisodeEndStatus.END
                break

        if terminal_status is None:
            terminal_status = self._commands.consume_end_status() or EpisodeEndStatus.END
        else:
            self._commands.consume_end_status()

        self._last_outcome = EpisodeOutcome(status=terminal_status, reward=reward_for_status(terminal_status))
        print(
            f"Episode {episode_idx + 1} ending: status={terminal_status.value}, reward={self._last_outcome.reward:.1f}"
        )
        self._say(self._episode_end_message(terminal_status, episode_idx))

        self._episode_active.clear()
        self._policy_active.clear()

        for subscriber in self._subscribers:
            subscriber.on_episode_end()

        self._finish_lerobot_episode(terminal_status)

        print(f"Episode {episode_idx + 1} completed. Steps: {self._episode_steps}")
        return self._last_outcome

    def _is_intervention_active(self) -> bool:
        if self._commands.is_intervention_active():
            return True
        if (
            self._spacemouse_auto_intervene
            and self._teleop is not None
            and self._teleop.is_connected
            and isinstance(self._teleop, SpacemouseTeleop)
        ):
            try:
                return self._teleop.should_intervene()
            except Exception:
                logger.debug("SpaceMouse should_intervene check failed", exc_info=True)
        return False

    def _should_record_lerobot_step(self, *, is_intervention: bool) -> bool:
        """Whether to call add_frame for this control step (lerobot HIL semantics).

        Must be evaluated before ``_resolve_control_action`` so that
        ``ActionInterpolator.needs_new_action()`` still reflects chunk boundaries.
        """
        if not self._lerobot_storage_enabled():
            return False
        cfg = self._lerobot_storage_cfg
        assert cfg is not None
        if cfg.record_interpolated_actions:
            return True
        if is_intervention or self._interpolator is None:
            return True
        # Policy + interpolation: one dataset frame per policy action, not per sub-step.
        return self._interpolator.needs_new_action()

    def _should_poll_observation(self, is_intervention: bool) -> bool:
        """Poll env observation at inference fps, or every control step during intervention."""
        if self._cached_observation is None:
            return True
        if is_intervention:
            return True
        return (time.perf_counter() - self._last_obs_poll_t) >= self._obs_poll_interval

    def _poll_observation_from_env(self) -> dict:
        """Blocking env read; only call from the actor control thread."""
        obs = self._environment.get_observation()
        self._cached_observation = obs
        with self._obs_lock:
            self._shared_observation = obs
        return obs

    def _get_action_worker(self) -> None:
        print("[GetActionThread] Starting...")
        time_per_step = 1.0 / self._fps
        threshold = self._action_queue_size_to_get_new_actions if self._rtc_config.enabled else 0

        while not self._shutdown_event.is_set():
            try:
                if not self._episode_active.is_set():
                    precise_sleep(0.01)
                    continue

                if not self._policy_active.is_set():
                    precise_sleep(0.01)
                    continue

                with self._obs_lock:
                    shared_obs = self._shared_observation
                if shared_obs is None:
                    precise_sleep(0.01)
                    continue

                if self._action_queue.qsize() <= threshold:
                    current_time = time.perf_counter()
                    action_index_before = self._action_queue.get_action_index()
                    prev_actions = self._action_queue.get_left_over()
                    obs = _copy_observation(shared_obs)
                    if prev_actions is not None:
                        obs["actions"] = prev_actions

                    result = self._policy.infer(obs=obs, use_rtc=self._rtc_config.enabled)

                    if self._inference_warmup_steps <= 3:
                        self._inference_warmup_steps += 1
                        continue
                    if "actions" not in result:
                        precise_sleep(0.01)
                        continue

                    actions = result["actions"]
                    if len(actions.shape) == 1:
                        actions = actions[np.newaxis, ...]

                    latency = time.perf_counter() - current_time
                    self._total_inference_time += latency
                    self._inference_count += 1
                    self._latency_tracker.add(latency)
                    self._action_queue.merge(
                        original_actions=actions,
                        processed_actions=actions,
                        real_delay=int(np.ceil(latency / time_per_step)),
                        action_index_before_inference=action_index_before,
                    )
                else:
                    precise_sleep(time_per_step)
            except Exception as e:
                logger.exception("[GetActionThread] Error: %s", e)
                if not self._shutdown_event.is_set():
                    precise_sleep(0.1)
        print("[GetActionThread] Exiting...")

    def _resolve_control_action(self, *, is_intervention: bool) -> tuple[Optional[dict], bool]:
        if is_intervention:
            if self._teleop is None or self._teleop_action_fn is None:
                if not self._warned_intervention_no_teleop:
                    logger.warning(
                        "Human intervention active but teleop is not configured; no actions will be sent."
                    )
                    self._warned_intervention_no_teleop = True
                return None, True
            action = self._teleop_action_fn(self._teleop.get_action())
            
            if action is None:
                return None, True
            if "actions" not in action:
                action = {"actions": action}
            return action, True

        if self._interpolator:
            if self._interpolator.needs_new_action():
                queued = self._action_queue.get()
                if queued is not None:
                    self._interpolator.add(queued)
            interpolated = self._interpolator.get()
            if interpolated is not None:
                return {"actions": interpolated}, False
            return None, False

        queued = self._action_queue.get()
        if queued is not None:
            return {"actions": queued}, False
        return None, False

    def _reset_policy_buffers_for_intervention(self, *, entering: bool) -> None:
        """Drop stale policy actions when switching between teleop and RTC policy."""
        self._action_queue.clear()
        if self._interpolator:
            self._interpolator.reset()
        if entering:
            self._policy_active.clear()
        else:
            self._policy.reset()
            if self._episode_active.is_set():
                self._policy_active.set()

    def _actor_control_worker(self) -> None:
        print("[ActorControlThread] Starting...")
        if self._interpolator:
            control_interval = self._interpolator.get_control_interval(self._fps)
        else:
            control_interval = 1.0 / self._fps

        step_count = 0
        prev_intervention = False

        while not self._shutdown_event.is_set():
            try:
                if not self._episode_active.is_set():
                    prev_intervention = False
                    precise_sleep(0.01)
                    continue

                start_time = time.perf_counter()

                is_intervention = self._is_intervention_active()
                if prev_intervention != is_intervention:
                    self._reset_policy_buffers_for_intervention(entering=is_intervention)
                    if is_intervention:
                        self._say("Human intervention active. Use SpaceMouse to teleoperate.")
                    else:
                        self._say("Resuming policy control.")

                record_lerobot = self._should_record_lerobot_step(is_intervention=is_intervention)
            
                action_to_apply, is_intervention = self._resolve_control_action(is_intervention=is_intervention)

                # Poll observation at inference fps, or every control step during intervention.
                if self._should_poll_observation(is_intervention):
                    observation = self._poll_observation_from_env()
                    self._last_obs_poll_t = time.perf_counter()
                else:
                    observation = self._cached_observation

                # Apply action to environment.
                if action_to_apply is not None:
                    self._environment.apply_action(action_to_apply)

                # Notify subscribers and record LeRobot step.
                if action_to_apply is not None and observation is not None:
                    pending = self._commands.peek_end_status()
                    hil_action = dict(action_to_apply)
                    hil_action["_hil"] = {
                        "is_intervention": is_intervention,
                        "reward": reward_for_status(pending) if pending is not None else 0.0,
                        "episode_end_status": pending.value if pending is not None else None,
                    }
                    for subscriber in self._subscribers:
                        subscriber.on_step(observation, hil_action)
                    if record_lerobot:
                        self._record_lerobot_step(observation, hil_action)
                    step_count += 1
                    self._episode_steps = step_count

                prev_intervention = is_intervention

                dt = time.perf_counter() - start_time
                self._total_control_time += dt
                sleep_time = control_interval - dt
                if sleep_time > 0:
                    precise_sleep(sleep_time)
                else:
                    # If sleep_time is negative or zero, we don't sleep to avoid over-compensating
                    print(f"[ActorControlThread] Warning: control loop is taking longer ({dt*1000:.1f}ms) than control interval ({control_interval*1000:.1f}ms). Skipping sleep.")
            except Exception as e:
                logger.exception("[ActorControlThread] Error: %s", e)
                if not self._shutdown_event.is_set():
                    precise_sleep(0.1)
        print(f"[ActorControlThread] Exiting. Executed {step_count} steps.")

    def _close_lerobot_dataset(self) -> None:
        if self._lerobot_remote is not None:
            try:
                self._lerobot_remote.close()
            except Exception as e:
                logger.exception("Failed to close remote LeRobot recorder: %s", e)
            finally:
                self._lerobot_remote = None
        if self._lerobot_dataset is None:
            return
        try:
            stop_image_writer = getattr(self._lerobot_dataset, "stop_image_writer", None)
            if callable(stop_image_writer):
                stop_image_writer()
            if self._lerobot_storage_cfg and self._lerobot_storage_cfg.push_to_hub:
                self._lerobot_dataset.push_to_hub(
                    tags=self._lerobot_storage_cfg.tags,
                    private=self._lerobot_storage_cfg.private,
                )
        except Exception as e:
            logger.exception("Failed to close LeRobot dataset: %s", e)
        finally:
            self._lerobot_dataset = None

    @staticmethod
    def _lerobot_vector_names(dim: int, names: Optional[List[str]]) -> List[str]:
        if names is not None and len(names) == dim:
            return list(names)
        return [f"dim_{i}" for i in range(dim)]

    @staticmethod
    def _lerobot_image_names(shape: tuple[int, ...]) -> List[str]:
        if len(shape) == 3:
            return ["channels", "height", "width"]
        return [f"dim_{i}" for i in range(len(shape))]

    def _build_lerobot_features(self, observation: dict, action: dict) -> dict:
        return build_lerobot_features(observation, action, self._lerobot_storage_cfg)

    def _lerobot_dataset_root(self) -> Path:
        cfg = self._lerobot_storage_cfg
        assert cfg is not None
        if cfg.root is not None:
            return Path(cfg.root)
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # type: ignore[import-not-found]

        return Path(HF_LEROBOT_HOME) / cfg.repo_id

    def _ensure_lerobot_remote(self) -> None:
        if self._lerobot_remote is None or self._lerobot_remote.initialized:
            return
        try:
            assert self._lerobot_storage_cfg is not None
            self._lerobot_remote.initialize(cfg=self._lerobot_storage_cfg, fps=self._fps)
            print(
                f"Remote LeRobot recorder connected: endpoint={self._lerobot_storage_cfg.remote_endpoint} "
                f"repo_id={self._lerobot_storage_cfg.repo_id}"
            )
        except Exception as e:
            logger.exception("Failed to connect remote LeRobot recorder: %s", e)
            self._disable_lerobot_storage(f"remote recorder unavailable: {e}")
            self._lerobot_remote = None

    def _ensure_lerobot_dataset(self, observation: dict, action: dict) -> None:
        if not self._lerobot_storage_enabled():
            return
        if self._lerobot_storage_is_remote():
            self._ensure_lerobot_remote()
            return
        if self._lerobot_dataset is not None:
            return
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
        except ImportError as e:
            self._disable_lerobot_storage(f"lerobot not installed: {e}")
            return

        try:
            features = self._build_lerobot_features(observation, action)
            cfg = self._lerobot_storage_cfg
            assert cfg is not None
            if cfg.clear_existing:
                candidate = self._lerobot_dataset_root()
                if candidate.exists():
                    shutil.rmtree(candidate)
            self._lerobot_dataset = LeRobotDataset.create(
                repo_id=cfg.repo_id,
                fps=cfg.fps if cfg.fps is not None else int(self._fps),
                root=cfg.root,
                robot_type=cfg.robot_type,
                features=features,
                use_videos=cfg.use_videos,
                tolerance_s=cfg.tolerance_s,
                image_writer_processes=cfg.image_writer_processes,
                image_writer_threads=cfg.image_writer_threads,
                video_backend=cfg.video_backend,
            )
            print(
                "LeRobot v2.1 recorder initialized: repo_id=%s root=%s",
                cfg.repo_id,
                self._lerobot_dataset.root,
            )
        except Exception as e:
            print("Failed to initialize LeRobot dataset %s", e)
            logger.exception("Failed to initialize LeRobot dataset: %s", e)
            self._disable_lerobot_storage(f"initialization failed: {e}")
            self._lerobot_dataset = None

    def _record_lerobot_step(self, observation: dict, action: dict) -> None:
        if not self._lerobot_storage_enabled():
            return
        self._ensure_lerobot_dataset(observation, action)
        cfg = self._lerobot_storage_cfg
        assert cfg is not None

        try:
            if self._lerobot_remote is not None:
                self._lerobot_remote.add_frame(
                    episode_idx=self._current_episode_idx,
                    observation=observation,
                    action=action,
                    task=cfg.task,
                )
                self._lerobot_frame_count += 1
                return
            if self._lerobot_dataset is None:
                return
            frame = build_lerobot_frame(observation, action, cfg.task)
            self._lerobot_dataset.add_frame(frame)
            self._lerobot_frame_count += 1
        except Exception as e:
            logger.exception("Failed to record LeRobot frame: %s", e)

    def _finish_lerobot_episode(self, terminal_status: EpisodeEndStatus) -> None:
        if self._lerobot_remote is not None:
            try:
                self._lerobot_remote.finish_episode(
                    episode_idx=self._current_episode_idx,
                    status=terminal_status.value,
                )
            except Exception as e:
                logger.exception("Failed to finalize remote LeRobot episode: %s", e)
            return
        if self._lerobot_dataset is None:
            return
        try:
            if terminal_status == EpisodeEndStatus.RERECORD:
                self._lerobot_dataset.clear_episode_buffer()
                logger.info("LeRobot episode buffer cleared (rerecord).")
                return

            if self._lerobot_dataset.episode_buffer is None or self._lerobot_dataset.episode_buffer.get("size", 0) == 0:
                logger.warning("LeRobot episode buffer is empty; skipping save_episode().")
                self._lerobot_dataset.clear_episode_buffer()
                return

            self._lerobot_dataset.save_episode()
            logger.info(
                "LeRobot episode saved (v2.1): index=%s status=%s success=%s",
                self._current_episode_idx,
                terminal_status.value,
                terminal_status == EpisodeEndStatus.SUCCESS,
            )
        except Exception as e:
            logger.exception("Failed to finalize LeRobot episode: %s", e)

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        arr = np.asarray(value)
        return arr.copy()


# Re-export teleop helpers for convenience at previous import path.
__all__ = [
    "EpisodeEndStatus",
    "EpisodeOutcome",
    "LeRobotRemoteRecorderClient",
    "LeRobotStorageConfig",
    "RuntimeRTCHil",
    "SpacemouseTeleop",
    "SpacemouseTeleopConfig",
    "teleop_eef_to_actions",
    "EefActionConverter",
    "EefKinematics",
    "EefKinematicsConfig",
]
