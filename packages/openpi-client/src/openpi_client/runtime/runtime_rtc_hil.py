"""RTC runtime with keyboard / SpaceMouse human-in-the-loop and episode labeling.

Extends RuntimeRTC without modifying runtime_rtc.py.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock
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

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = False
try:
    from pynput import keyboard

    PYNPUT_AVAILABLE = True
except Exception:
    pynput_keyboard = None  # type: ignore[assignment]


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


@dataclass(frozen=True)
class LeRobotStorageConfig:
    """Optional LeRobot-compatible recording config.

    This targets the workflow used in `lerobot/examples/rac/hil_data_collection.py`,
    while allowing simpler non-3.0-like feature schemas from this runtime.
    """

    repo_id: str
    task: str
    root: Optional[str] = None
    robot_type: str = "openpi_hil"
    fps: int = 30
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
            logger.warning("pynput unavailable; keyboard episode control disabled.")
            return False

        def on_press(key):
            with self._lock:
                self._handle_key(key)

        self._listener = keyboard.Listener(on_press=on_press)
        self._listener.start()
        self._print_help()
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
                logger.info("Esc pressed: aborting all episodes.")
                return
            if key == keyboard.Key.right:
                self._state.request_end(EpisodeEndStatus.NEXT_EPISODE)
                return
            if key == keyboard.Key.left:
                self._state.request_end(EpisodeEndStatus.RERECORD)
                return
            if key == keyboard.Key.space:
                active = self._state.toggle_intervention()
                logger.info("Human intervention %s", "ON" if active else "OFF")
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
                    logger.info("Human intervention %s", "ON" if active else "OFF")
        except Exception:
            logger.exception("keyboard handler error")


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
    """RuntimeRTC + keyboard episode control + optional SpaceMouse teleop."""

    def __init__(
        self,
        environment: _environment.Environment,
        policy: _base_policy.BasePolicy,
        subscribers: List[_subscriber.Subscriber],
        *,
        fps: float = 50.0,
        num_episodes: int = 1,
        max_episode_time_s: int = 0,
        use_action_interpolation: bool = False,
        multiplier: int = 1,
        rtc_config: Optional[RTCConfig] = None,
        action_queue_size_to_get_new_actions: int = 30,
        enable_keyboard: bool = True,
        teleop: Optional[Teleoperator] = None,
        teleop_action_fn: Optional[TeleopActionFn] = None,
        eef_kinematics: Optional[EefKinematics] = None,
        get_joint_positions: Optional[Callable[[], np.ndarray]] = None,
        spacemouse_auto_intervene: bool = True,
        inference_start_delay_s: float = 0.1,
        shared_obs_wait_timeout_s: float = 10.0,
        monitor_interval_s: float = 0.1,
        lerobot_storage: Optional[LeRobotStorageConfig] = None,
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
        self._teleop = teleop
        self._spacemouse_auto_intervene = spacemouse_auto_intervene
        self._inference_start_delay_s = inference_start_delay_s
        self._shared_obs_wait_timeout_s = shared_obs_wait_timeout_s
        self._monitor_interval_s = monitor_interval_s
        self._last_outcome: Optional[EpisodeOutcome] = None
        self._obs_lock = Lock()
        self._shared_observation: Optional[dict] = None
        self._lerobot_storage_cfg = lerobot_storage
        self._lerobot_dataset = None
        self._lerobot_feature_ready = False
        self._lerobot_frame_count = 0
        self._current_episode_idx = 0

        if teleop_action_fn is not None:
            self._teleop_action_fn = teleop_action_fn
        elif eef_kinematics is not None and get_joint_positions is not None:
            self._teleop_action_fn = EefActionConverter(eef_kinematics, get_joint_positions)
        else:
            self._teleop_action_fn = None

    @property
    def last_episode_outcome(self) -> Optional[EpisodeOutcome]:
        return self._last_outcome

    def run(self) -> None:
        if self._keyboard is not None:
            self._keyboard.start()
        if self._teleop is not None:
            self._teleop.connect()

        episode_idx = 0
        try:
            while episode_idx < self._num_episodes and not self._shutdown_event.is_set():
                outcome = self._run_episode(episode_idx)
                if outcome is None:
                    episode_idx += 1
                    continue
                if outcome.status == EpisodeEndStatus.ABORT_ALL or self._commands.should_abort_all():
                    print("All episodes aborted.")
                    break
                if outcome.status == EpisodeEndStatus.RERECORD:
                    print(f"Rerecording episode {episode_idx + 1}...")
                    continue
                episode_idx += 1

            self._environment.reset()
            self._log_statistics()
        finally:
            self._close_lerobot_dataset()
            if self._teleop is not None:
                self._teleop.disconnect()
            if self._keyboard is not None:
                self._keyboard.stop()
            self._shutdown_event.set()

    def _run_episode(self, episode_idx: int) -> Optional[EpisodeOutcome]:
        self._current_episode_idx = episode_idx
        self._commands.reset_episode()
        self._last_outcome = None
        print(f"Starting episode {episode_idx + 1}/{self._num_episodes}...")

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
        self._inference_warmup_steps = 0

        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        with self._obs_lock:
            self._shared_observation = None

        self._shutdown_event.clear()
        actor_control_thread = threading.Thread(target=self._actor_control_worker, daemon=True, name="ActorControlThread")
        get_action_thread = threading.Thread(target=self._get_action_worker, daemon=True, name="GetActionThread")
        # Actor owns observation I/O; inference reads shared cache after actor primes it.
        actor_control_thread.start()
        time.sleep(self._inference_start_delay_s)
        get_action_thread.start()

        start_time = time.perf_counter()
        terminal_status: Optional[EpisodeEndStatus] = None
        while not self._shutdown_event.is_set():
            precise_sleep(self._monitor_interval_s)
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

        self._shutdown_event.set()
        actor_control_thread.join()
        get_action_thread.join()

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
            and self._teleop.should_intervene()
        ):
            return True
        return False

    def _publish_shared_observation(self, obs: dict) -> None:
        with self._obs_lock:
            self._shared_observation = obs

    def _refresh_observation_from_env(self) -> dict:
        """Blocking env read; only call from the actor control thread."""
        obs = self._environment.get_observation()
        self._publish_shared_observation(obs)
        return obs

    def _get_shared_observation_for_infer(self) -> Optional[dict]:
        with self._obs_lock:
            if self._shared_observation is None:
                return None
            return _copy_observation(self._shared_observation)

    def _wait_for_shared_observation(self) -> dict:
        deadline = time.perf_counter() + self._shared_obs_wait_timeout_s
        while not self._shutdown_event.is_set():
            obs = self._get_shared_observation_for_infer()
            if obs is not None:
                return obs
            if time.perf_counter() >= deadline:
                break
            precise_sleep(0.01)
        raise TimeoutError(
            "Timed out waiting for shared observation from actor thread. "
            "Ensure the actor control thread is running."
        )

    def _get_action_worker(self) -> None:
        print("[GetActionThread] Starting...")
        time_per_step = 1.0 / self._fps
        threshold = self._action_queue_size_to_get_new_actions if self._rtc_config.enabled else 0

        while not self._shutdown_event.is_set():
            try:
                if self._is_intervention_active():
                    precise_sleep(time_per_step)
                    continue

                if self._action_queue.qsize() <= threshold:
                    current_time = time.perf_counter()
                    action_index_before = self._action_queue.get_action_index()
                    prev_actions = self._action_queue.get_left_over()
                    try:
                        obs = self._wait_for_shared_observation()
                    except TimeoutError:
                        precise_sleep(0.05)
                        continue
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
                logging.exception("[GetActionThread] Error: %s", e)
                if not self._shutdown_event.is_set():
                    precise_sleep(0.1)
        print("[GetActionThread] Exiting...")

    def _resolve_control_action(self) -> tuple[Optional[dict], bool]:
        if self._is_intervention_active():
            if self._teleop is None or self._teleop_action_fn is None:
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

    def _actor_control_worker(self) -> None:
        print("[ActorControlThread] Starting...")
        if self._interpolator:
            control_interval = self._interpolator.get_control_interval(self._fps)
        else:
            control_interval = 1.0 / self._fps

        step_count = 0
        prev_intervention = False
        # Prime shared observation before inference thread needs it.
        try:
            self._refresh_observation_from_env()
        except Exception as e:
            logging.exception("[ActorControlThread] Failed to prime observation: %s", e)

        while not self._shutdown_event.is_set():
            try:
                start_time = time.perf_counter()
                action_to_apply, is_intervention = self._resolve_control_action()

                if action_to_apply is not None:
                    self._environment.apply_action(action_to_apply)

                observation = self._refresh_observation_from_env()

                if action_to_apply is not None:
                    pending = self._commands.peek_end_status()
                    hil_action = dict(action_to_apply)
                    hil_action["_hil"] = {
                        "is_intervention": is_intervention,
                        "reward": reward_for_status(pending) if pending is not None else 0.0,
                        "episode_end_status": pending.value if pending is not None else None,
                    }
                    for subscriber in self._subscribers:
                        subscriber.on_step(observation, hil_action)
                    self._record_lerobot_step(observation, hil_action)
                    step_count += 1
                    self._episode_steps = step_count

                if prev_intervention and not is_intervention:
                    self._action_queue.clear()
                    if self._interpolator:
                        self._interpolator.reset()
                prev_intervention = is_intervention

                dt = time.perf_counter() - start_time
                self._total_control_time += dt
                sleep_time = control_interval - dt
                if sleep_time > 0:
                    precise_sleep(sleep_time)
            except Exception as e:
                logging.exception("[ActorControlThread] Error: %s", e)
                if not self._shutdown_event.is_set():
                    precise_sleep(0.1)
        print(f"[ActorControlThread] Exiting. Executed {step_count} steps.")

    def _close_lerobot_dataset(self) -> None:
        if self._lerobot_dataset is None:
            return
        try:
            finalize_fn = getattr(self._lerobot_dataset, "finalize", None)
            if callable(finalize_fn):
                finalize_fn()
            if self._lerobot_storage_cfg and self._lerobot_storage_cfg.push_to_hub:
                self._lerobot_dataset.push_to_hub(
                    tags=self._lerobot_storage_cfg.tags,
                    private=self._lerobot_storage_cfg.private,
                )
        except Exception as e:
            logger.exception("Failed to finalize LeRobot dataset: %s", e)
        finally:
            self._lerobot_dataset = None

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        arr = np.asarray(value)
        return arr.copy()

    def _build_lerobot_features(self, observation: dict, action: dict) -> dict:
        state = self._to_numpy(observation.get("state", []))
        action_vec = self._to_numpy(action.get("actions", []))
        state_dim = int(state.reshape(-1).shape[0])
        action_dim = int(action_vec.reshape(-1).shape[0])
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": [f"state_{i}" for i in range(state_dim)],
            },
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [f"action_{i}" for i in range(action_dim)],
            },
            "complementary_info.is_intervention": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["is_intervention"],
            },
        }
        images = observation.get("images") or {}
        image_dtype = "image"
        if self._lerobot_storage_cfg is not None:
            image_dtype = self._lerobot_storage_cfg.mode
        for cam_name, image in images.items():
            image_np = self._to_numpy(image)
            features[f"observation.images.{cam_name}"] = {
                "dtype": image_dtype,
                "shape": tuple(image_np.shape),
                "names": [f"dim_{i}" for i in range(len(image_np.shape))],
            }
        return features

    def _ensure_lerobot_dataset(self, observation: dict, action: dict) -> None:
        if self._lerobot_storage_cfg is None or self._lerobot_dataset is not None:
            return
        try:
            # Prefer convert_aloha_data_to_lerobot.py style API, fallback to newer path.
            from lerobot.common.datasets.lerobot_dataset import (  # type: ignore[import-not-found]
                HF_LEROBOT_HOME,
                LeRobotDataset,
            )
        except Exception as e:
            try:
                from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
                from lerobot.utils.constants import HF_LEROBOT_HOME  # type: ignore[import-not-found]
            except Exception:
                logger.warning("LeRobot storage disabled (import failed): %s", e)
                self._lerobot_storage_cfg = None
                return

        try:
            features = self._build_lerobot_features(observation, action)
            cfg = self._lerobot_storage_cfg
            if cfg.clear_existing:
                candidate = Path(HF_LEROBOT_HOME) / cfg.repo_id
                if candidate.exists():
                    shutil.rmtree(candidate)
            self._lerobot_dataset = LeRobotDataset.create(
                repo_id=cfg.repo_id,
                fps=cfg.fps,
                root=cfg.root,
                robot_type=cfg.robot_type,
                features=features,
                use_videos=cfg.use_videos,
                tolerance_s=cfg.tolerance_s,
                image_writer_processes=cfg.image_writer_processes,
                image_writer_threads=cfg.image_writer_threads,
                video_backend=cfg.video_backend,
            )
            self._lerobot_feature_ready = True
            logger.info("LeRobot recorder initialized: repo_id=%s", cfg.repo_id)
        except Exception as e:
            logger.exception("Failed to initialize LeRobot dataset: %s", e)
            self._lerobot_storage_cfg = None
            self._lerobot_dataset = None

    def _record_lerobot_step(self, observation: dict, action: dict) -> None:
        if self._lerobot_storage_cfg is None:
            return
        self._ensure_lerobot_dataset(observation, action)
        if self._lerobot_dataset is None:
            return

        try:
            frame = {
                "observation.state": self._to_numpy(observation.get("state", [])).astype(np.float32).reshape(-1),
                "action": self._to_numpy(action.get("actions", [])).astype(np.float32).reshape(-1),
                "task": self._lerobot_storage_cfg.task,
            }
            images = observation.get("images") or {}
            for cam_name, image in images.items():
                frame[f"observation.images.{cam_name}"] = self._to_numpy(image)
            hil_info = action.get("_hil", {})
            is_intervention = 1.0 if bool(hil_info.get("is_intervention", False)) else 0.0
            frame["complementary_info.is_intervention"] = np.array([is_intervention], dtype=np.float32)
            self._lerobot_dataset.add_frame(frame)
            self._lerobot_frame_count += 1
        except Exception as e:
            logger.exception("Failed to record LeRobot frame: %s", e)

    def _finish_lerobot_episode(self, terminal_status: EpisodeEndStatus) -> None:
        if self._lerobot_dataset is None:
            return
        try:
            should_save = terminal_status != EpisodeEndStatus.RERECORD
            if should_save:
                metadata = {
                    "episode_status": terminal_status.value,
                    "episode_success": terminal_status == EpisodeEndStatus.SUCCESS,
                    "episode_index": self._current_episode_idx,
                }
                save_episode_fn = getattr(self._lerobot_dataset, "save_episode")
                try:
                    save_episode_fn(extra_episode_metadata=metadata)
                except TypeError:
                    # Compatibility with older LeRobot APIs lacking extra metadata argument.
                    save_episode_fn()
            elif hasattr(self._lerobot_dataset, "clear_episode_buffer"):
                self._lerobot_dataset.clear_episode_buffer()
        except Exception as e:
            logger.exception("Failed to finalize LeRobot episode: %s", e)


# Re-export teleop helpers for convenience at previous import path.
__all__ = [
    "EpisodeEndStatus",
    "EpisodeOutcome",
    "RuntimeRTCHil",
    "SpacemouseTeleop",
    "SpacemouseTeleopConfig",
    "teleop_eef_to_actions",
    "EefActionConverter",
    "EefKinematics",
    "EefKinematicsConfig",
]
