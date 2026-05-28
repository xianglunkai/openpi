# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from enum import IntEnum
import threading
import time
from typing import Any

import numpy as np

from openpi_client.teleop.teleoperator import Teleoperator
from openpi_client.teleop.teleoperator import TeleoperatorConfig

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
            "x": 0.02,
            "y": 0.02,
            "z": 0.02,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        }
    )
    deadzone: float = 0.05


class SpacemouseTeleop(Teleoperator):
    """
    Teleop class to use spacemouse inputs for control.
    """

    config_class = SpacemouseTeleopConfig

    name = "spacemouse"

    def __init__(self, config: SpacemouseTeleopConfig | None = None):
        super().__init__(config or SpacemouseTeleopConfig())
        self.config: SpacemouseTeleopConfig = self.config  # type: ignore[assignment]

        self._connected = False
        # Underlying pyspacemouse device object (returned by pyspacemouse.open())
        self._device = None
        # Gripper toggle state: assume starts OPEN
        self._gripper_state: int = GripperAction.STAY.value
        self._prev_button_state: int = 0

        # Background reader thread vars (used to keep only the latest state)
        self._latest_state = None  # will store the most recent raw state coming from the driver
        self._reader_thread = None
        self._stop_reader = False
        self.state_lock = threading.Lock()
        self._last_call_ms = None
    @property
    def action_features(self) -> dict:
        if self.config.use_gripper:
            return {
                "dtype": "float32",
                "shape": (7,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_roll": 3, "delta_pitch": 4, "delta_yaw": 5, "gripper": 6},
            }
        else:
            return {
                "dtype": "float32",
                "shape": (6,),
                "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "delta_roll": 3, "delta_pitch": 4, "delta_yaw": 5},
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
        """Connect to the SpaceMouse device (real or mock)."""
        if pyspacemouse is None:
            raise ImportError("pyspacemouse is required for SpacemouseTeleop")
        # pyspacemouse.open() returns a device object (which may also be used
        # as a context manager). Preserve the object so we can call its
        # read()/close() methods instead of relying on any module-level helpers.
        try:
            self._device = pyspacemouse.open()
        except Exception:
            # open failed; ensure device stays None
            self._device = None

        self._connected = self._device is not None

        # Start background reader to avoid piling up driver messages (reduces perceived latency)
        if self._connected:

            def _reader_loop():
                """Continuously poll the driver so its internal queue stays empty.

                We only keep the most recent state which `get_action` then consumes. This
                prevents buildup when the driver polls faster than the main control loop
                (e.g. teleoperate.py's ≈60 Hz loop vs. SpaceMouse ≈125 Hz updates).
                """
                while not self._stop_reader:
                    try:
                        # Prefer the device instance's read() method.
                        if self._device is None:
                            break
                        with self.state_lock:
                            self._latest_state = self._device.read()
                    except Exception:
                        # In case device is unplugged mid-run; exit thread gracefully
                        break
                    time.sleep(0.005)  # Sleep a little: driver typically updates ~100-200Hz; reduce CPU by a short sleep

            self._stop_reader = False
            self._reader_thread = threading.Thread(target=_reader_loop, daemon=True)
            self._reader_thread.start()
    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("SpaceMouse is not connected. Call connect() first.")

        # Prefer the state produced by the background reader (most recent),
        # fall back to direct read if thread hasn't produced anything yet.
        with self.state_lock:
            if self._latest_state is not None:
                state = self._latest_state
            else:
                # Fallback to device.read() when available. Avoid using module-level
                # read() which some pyspacemouse distributions do not expose.
                if self._device is not None and hasattr(self._device, "read"):
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
        
        # clamp deltas to [-1, 1] just in case (pyspacemouse docs say values are already normalized but we add this as a safety measure)
        deltas = np.array([max(-1.0, min(1.0, d)) for d in deltas])

        # convert deltas to spacemouse_action
        spacemouse_action = np.array(deltas, dtype=np.float32)
        
        now = time.perf_counter()
        # Intentionally avoid input-side low-pass filtering here.
        # We now smooth on the robot side (v*dt + IK + trajectory smoothing), so
        # filtering again at the teleop layer adds avoidable control lag.
        
        # apply deadzone on raw normalized axis, then map to velocity command
        for i, axis in enumerate(["x", "y", "z", "roll", "pitch", "yaw"]):
            # Keep backward compatibility: if a caller still uses legacy per-cycle
            # scaling, axis_max_speeds can be overridden in config.
            limit = self.config.end_effector_step_sizes.get(axis, 0.01)
        
            raw_axis = spacemouse_action[i]
            if abs(raw_axis) < self.config.deadzone:
                raw_axis = 0.0

            cmd = raw_axis * limit

            spacemouse_action[i] = cmd

        action_dict = {
            # Historical key names kept for compatibility with downstream robot
            # adapters. Values now represent velocity commands.
            "delta_x": spacemouse_action[0],
            "delta_y": spacemouse_action[1],
            "delta_z": spacemouse_action[2],
            "delta_roll": spacemouse_action[3],
            "delta_pitch": spacemouse_action[4],
            "delta_yaw": spacemouse_action[5],
            "home": False,
        }

        # Simple gripper control: right button (index 1) toggles open/close.
        # Assumption: the physical gripper starts in the OPEN state when the teleop script boots.
        # Each button press switches the command between OPEN and CLOSE accordingly.
       
        if self.config.use_gripper and hasattr(state, "buttons") and len(state.buttons) >= 2:

            if self._last_call_ms is None:
                self._gripper_state=  GripperAction.STAY.value
            else:
                # Normalize button value to boolean (some drivers return 0/1, others may return truthy values)
                try:
                    current_btn = bool(state.buttons[1])
                except Exception:
                    current_btn = False

                # Toggle on rising edge: pressed now but was not pressed previously
                if current_btn and not bool(self._prev_button_state):
                    self._gripper_state = (
                        GripperAction.CLOSE.value
                        if self._gripper_state == GripperAction.OPEN.value
                        else GripperAction.OPEN.value
                    )

                # Always update prev state to the normalized boolean
                self._prev_button_state = current_btn
                
            action_dict["gripper"] = float(self._gripper_state)

        # Home (left) button: independent of use_gripper so reset still works without gripper mapping.
        if hasattr(state, "buttons") and len(state.buttons) >= 2:
            try:
                home_btn = bool(state.buttons[0])
            except Exception:
                home_btn = False
            action_dict["home"] = home_btn

        self._last_call_ms = now
      
        return action_dict

    def disconnect(self) -> None:
        """Disconnect from the spacemouse."""
        # pyspacemouse does not expose an explicit close API but we reset connection flag
        self._connected = False

        # Stop reader thread if running
        self._stop_reader = True
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=0.1)
            self._reader_thread = None

    def calibrate(self) -> None:
        """Calibrate the spacemouse."""
        # No calibration needed for spacemouse
        pass

    def configure(self) -> None:
        """Configure the spacemouse."""
        # No additional configuration needed
        pass

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        """Send feedback to the spacemouse."""
        # Spacemouse doesn't support feedback
        del feedback
