"""SpaceMouse teleoperator (lerobot src/lerobot/teleoperators/spacemouse)."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from openpi_client.teleop.teleoperator import Teleoperator, TeleoperatorConfig

try:
    import pyspacemouse
except ImportError:
    pyspacemouse = None  # type: ignore[assignment]


class GripperAction(IntEnum):
    CLOSE = 0
    STAY = 1
    OPEN = 2


@dataclass
class SpacemouseTeleopConfig(TeleoperatorConfig):
    use_gripper: bool = True
    mock: bool = False
    device: str = ""
    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.01,
            "y": 0.01,
            "z": 0.01,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }
    )
    fps: float = 30.0
    eef_cutoff_freq: float = 3.0
    deadzone: float = 0.0005


class SpacemouseTeleop(Teleoperator):
    """Teleop class to use SpaceMouse inputs for control."""

    config_class = SpacemouseTeleopConfig
    name = "spacemouse"

    def __init__(self, config: SpacemouseTeleopConfig | None = None) -> None:
        super().__init__(config or SpacemouseTeleopConfig())
        self.config: SpacemouseTeleopConfig = self.config  # type: ignore[assignment]

        self._connected = False
        self._device = None
        self._gripper_state: int = GripperAction.STAY.value
        self._prev_button_state: int = 0
        self._latest_state = None
        self._reader_thread: threading.Thread | None = None
        self._stop_reader = False
        self._state_lock = threading.Lock()
        self._last_call_s: float | None = None
        self._prev_action: np.ndarray | None = None

    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {
                    "delta_x": 0,
                    "delta_y": 1,
                    "delta_z": 2,
                    "delta_roll": 3,
                    "delta_pitch": 4,
                    "delta_yaw": 5,
                    "gripper": 6,
                },
            }
        return {
            "dtype": "float32",
            "shape": (6,),
            "names": {
                "delta_x": 0,
                "delta_y": 1,
                "delta_z": 2,
                "delta_roll": 3,
                "delta_pitch": 4,
                "delta_yaw": 5,
            },
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if pyspacemouse is None:
            raise ImportError("pyspacemouse is required for SpacemouseTeleop")

        try:
            self._device = pyspacemouse.open()
        except Exception:
            self._device = None

        self._connected = self._device is not None
        if not self._connected:
            raise RuntimeError("Failed to open SpaceMouse device.")

        def _reader_loop():
            while not self._stop_reader:
                try:
                    if self._device is None:
                        break
                    with self._state_lock:
                        self._latest_state = self._device.read()
                except Exception:
                    break
                time.sleep(0.005)

        self._stop_reader = False
        self._reader_thread = threading.Thread(target=_reader_loop, daemon=True, name="SpacemouseReader")
        self._reader_thread.start()

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("SpaceMouse is not connected. Call connect() first.")

        with self._state_lock:
            if self._latest_state is not None:
                state = self._latest_state
            elif self._device is not None and hasattr(self._device, "read"):
                state = self._device.read()
            else:
                raise RuntimeError("SpaceMouse device not ready and no read() method available")

        deltas = [
            state.y,
            -state.x,
            state.z,
            state.roll,
            state.pitch,
            -state.yaw,
        ]
        spacemouse_action = np.array([max(-1.0, min(1.0, d)) for d in deltas], dtype=np.float32)

        rc = 1.0 / (2 * math.pi * self.config.eef_cutoff_freq)
        now = time.perf_counter()
        if self._prev_action is None:
            self._prev_action = np.zeros_like(spacemouse_action)
        if self._last_call_s is not None:
            take_time = now - self._last_call_s
            if take_time > 1.0:
                self._last_call_s = None
            else:
                dt_lp = min(max(take_time, 1e-3), 2.0 / self.config.fps)
                alpha = dt_lp / (dt_lp + rc)
                spacemouse_action = alpha * spacemouse_action.copy() + (1 - alpha) * self._prev_action.copy()

        self._prev_action = spacemouse_action.copy()

        axes = ["x", "y", "z", "roll", "pitch", "yaw"]
        for i, axis in enumerate(axes):
            step_size = self.config.end_effector_step_sizes.get(axis, 0.01)
            delta_scaled = spacemouse_action[i] * step_size
            if abs(delta_scaled) < self.config.deadzone:
                delta_scaled = 0.0
            spacemouse_action[i] = delta_scaled

        action_dict: dict[str, Any] = {
            "delta_x": float(spacemouse_action[0]),
            "delta_y": float(spacemouse_action[1]),
            "delta_z": float(spacemouse_action[2]),
            "delta_roll": float(spacemouse_action[3]),
            "delta_pitch": float(spacemouse_action[4]),
            "delta_yaw": float(spacemouse_action[5]),
            "home": False,
        }

        if self.config.use_gripper and hasattr(state, "buttons") and len(state.buttons) >= 2:
            if self._last_call_s is None:
                self._gripper_state = GripperAction.STAY.value
            else:
                try:
                    current_btn = bool(state.buttons[1])
                except Exception:
                    current_btn = False
                if current_btn and not bool(self._prev_button_state):
                    self._gripper_state = (
                        GripperAction.CLOSE.value
                        if self._gripper_state == GripperAction.OPEN.value
                        else GripperAction.OPEN.value
                    )
                self._prev_button_state = current_btn
            action_dict["gripper"] = float(self._gripper_state)

        if hasattr(state, "buttons") and len(state.buttons) >= 1:
            try:
                action_dict["home"] = bool(state.buttons[0])
            except Exception:
                action_dict["home"] = False

        self._last_call_s = now
        return action_dict

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    def disconnect(self) -> None:
        self._connected = False
        self._stop_reader = True
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.1)
            self._reader_thread = None
        if self._device is not None and hasattr(self._device, "close"):
            try:
                self._device.close()
            except Exception:
                pass
        self._device = None

    def should_intervene(self) -> bool:
        action = self.get_action()
        axes = [abs(action[k]) for k in ("delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw")]
        return max(axes) >= self.config.deadzone
