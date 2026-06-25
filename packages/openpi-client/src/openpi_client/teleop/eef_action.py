"""Convert SpaceMouse EEF deltas to joint actions (agilex_cobot send_action_from_eef)."""

from __future__ import annotations

import time
from typing import Callable, Optional
import importlib
import importlib.util

import numpy as np

from openpi_client.teleop.kinematics import EefKinematics


def _euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    transforms3d_spec = importlib.util.find_spec("transforms3d")
    if transforms3d_spec is not None:
        t3d = importlib.import_module("transforms3d")
        return t3d.euler.euler2mat(roll, pitch, yaw)
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()


class EefActionConverter:
    """Map teleop delta dict -> environment action dict via IK (send_action_from_eef)."""

    def __init__(
        self,
        kinematics: EefKinematics,
        get_joint_positions: Callable[[], np.ndarray],
        *,
        gripper_open: float = 0.03,
        gripper_close: float = 0.0,
        input_fps: float = 30.0,
        velocity_filter_cutoff_hz: float = 5.0,
        dls_damping: float = 0.1,
        command_timeout_s: float = 0.5,
    ) -> None:
        self._kinematics = kinematics
        self._get_joint_positions = get_joint_positions
        self._gripper_open = gripper_open
        self._gripper_close = gripper_close
  
        self._dt = 1.0 / max(float(input_fps), 1.0)
        self._command_timeout_s = max(float(command_timeout_s), 0.0)

        self._last_call_eef_time: float | None = None
        self._servo_joint_state_initialized = False
        self._integrated_q = None
        self._filtered_vel = np.zeros(6, dtype=np.float64)
        self._velocity_filter_cutoff_hz = velocity_filter_cutoff_hz
        self._dls_damping = dls_damping
 

    def __call__(self, ee_command: dict) -> Optional[dict]:
        now = time.time()
        if self._last_call_eef_time is not None:
            if now - self._last_call_eef_time > self._command_timeout_s:
                self._servo_joint_state_initialized = False
                self._integrated_q = None
                self._filtered_vel.fill(0.0)
        self._last_call_eef_time = now

        dt = self._dt
        delta_x = ee_command.get("delta_x", 0.0)
        delta_y = ee_command.get("delta_y", 0.0)
        delta_z = ee_command.get("delta_z", 0.0)
        delta_roll = ee_command.get("delta_roll", 0.0)
        delta_pitch = ee_command.get("delta_pitch", 0.0)
        delta_yaw = ee_command.get("delta_yaw", 0.0)
        gripper = ee_command.get("gripper", 1.0)

        raw_vel = np.array([delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw],
                        dtype=np.float64)
        
        fc = self._velocity_filter_cutoff_hz
        alpha = 2.0 * np.pi * fc * dt
        alpha = np.clip(alpha, 0.0, 1.0)
        self._filtered_vel = (1.0 - alpha) * self._filtered_vel + alpha * raw_vel
        v_filt = self._filtered_vel

        
        q_raw = self._get_joint_positions()
        q = np.array(q_raw[:-1], dtype=np.float64)  
        if self._integrated_q is None or not self._servo_joint_state_initialized:
            if self._kinematics.use_rad:
                self._integrated_q = q.copy()
            else:
                self._integrated_q = np.deg2rad(q.copy())
            self._servo_joint_state_initialized = True

        lambda_dls = self._dls_damping
        J_pinv = self._kinematics.get_damped_pinv(self._integrated_q, damping=lambda_dls)
        q_dot = J_pinv @ v_filt
        self._integrated_q = self._integrated_q + q_dot * dt

        if self._kinematics.use_rad:
            desired_q = self._integrated_q
        else:
            desired_q = np.rad2deg(self._integrated_q)

        q_target = np.zeros(7)
        q_target[:-1] = desired_q

        gripper_mode = _decode_gripper_mode(gripper)
        if gripper_mode == 0:
            q_target[-1] = 0.0
        elif gripper_mode == 1:
            q_target[-1] = q_raw[-1]
        elif gripper_mode == 2:
            q_target[-1] = self._gripper_open
        else:
            q_target[-1] = q_raw[-1]

        return {"actions": q_target.astype(np.float32)}



def _decode_gripper_mode(value: float) -> int:
    # Robustly decode from float command (supports tiny numeric noise).
    if abs(value - 0.0) < 1e-6:
        return 0
    if abs(value - 1.0) < 1e-6:
        return 1
    if abs(value - 2.0) < 1e-6:
        return 2
    return 1


def teleop_eef_to_actions(
    ee_command: dict,
    *,
    kinematics: EefKinematics,
    get_joint_positions: Callable[[], np.ndarray],
    gripper_open: float = 0.03,
    gripper_close: float = 0.0,
) -> Optional[dict]:
    """Convert teleop EEF deltas to ``{"actions": joint_targets}`` (7-DoF arm + gripper).

    Mirrors ``AgilexCobot.send_action_from_eef`` in lerobot agilex_cobot.
    """
    delta_x = float(ee_command.get("delta_x", 0.0))
    delta_y = float(ee_command.get("delta_y", 0.0))
    delta_z = float(ee_command.get("delta_z", 0.0))
    delta_roll = float(ee_command.get("delta_roll", 0.0))
    delta_pitch = float(ee_command.get("delta_pitch", 0.0))
    delta_yaw = float(ee_command.get("delta_yaw", 0.0))
    gripper = float(ee_command.get("gripper", 1.0))


    q_raw = np.asarray(get_joint_positions(), dtype=np.float64).reshape(-1)
    if q_raw.size < 7:
        raise ValueError(f"Expected at least 7 joint values (6 arm + gripper), got shape {q_raw.shape}")

    q = q_raw[:-1]
    current_ee_pose = kinematics.forward_kinematics(q)

    ref = current_ee_pose.copy()
    delta_p = np.array([delta_x, delta_y, delta_z], dtype=np.float64)
    r_abs = _euler_to_matrix(delta_roll, delta_pitch, delta_yaw)
    desired = np.eye(4, dtype=np.float64)
    desired[:3, :3] = ref[:3, :3] @ r_abs
    desired[:3, 3] = ref[:3, 3] + delta_p

    q_target = np.zeros(7, dtype=np.float64)
    q_target[:-1] = kinematics.inverse_kinematics(q, desired)

    gripper_mode = _decode_gripper_mode(gripper)
    if gripper_mode == 0:
        q_target[-1] = gripper_close
    elif gripper_mode == 1:
        q_target[-1] = q_raw[-1]
    elif gripper_mode == 2:
        q_target[-1] = gripper_open
    else:
        q_target[-1] = q_raw[-1]

    return {"actions": q_target.astype(np.float32)}
