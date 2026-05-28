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
        min_delta: float = 1e-4,
        input_fps: float = 30.0,
        command_timeout_s: float = 0.5,
        joint_velocity_limits: np.ndarray | list[float] | None = None,
        joint_acceleration_limits: np.ndarray | list[float] | None = None,
        joint_jerk_limits: np.ndarray | list[float] | None = None,
    ) -> None:
        self._kinematics = kinematics
        self._get_joint_positions = get_joint_positions
        self._gripper_open = gripper_open
        self._gripper_close = gripper_close
        self._min_delta = min_delta
        self._dt = 1.0 / max(float(input_fps), 1.0)
        self._command_timeout_s = max(float(command_timeout_s), 0.0)

        # Fallback online smoother (velocity/acceleration/jerk constrained).
        default_v = np.array([0.785, 0.785, 0.785, 0.785, 0.785, 0.785], dtype=np.float64)
        default_a = np.array([1.57, 1.57, 1.57, 1.57, 1.57, 1.57], dtype=np.float64)
        default_j = np.array([6.28, 6.28, 6.28, 6.28, 6.28, 6.28], dtype=np.float64)
        self._joint_velocity_limits = _as_limit_array(joint_velocity_limits, default_v, "joint_velocity_limits")
        self._joint_acceleration_limits = _as_limit_array(
            joint_acceleration_limits,
            default_a,
            "joint_acceleration_limits",
        )
        self._joint_jerk_limits = _as_limit_array(joint_jerk_limits, default_j, "joint_jerk_limits")
        self._fallback_qd = np.zeros(6, dtype=np.float64)
        self._fallback_qdd = np.zeros(6, dtype=np.float64)
        self._last_call_s: float | None = None

    def __call__(self, ee_command: dict) -> Optional[dict]:
        now = time.time()
        if self._last_call_s is not None and (now - self._last_call_s) > self._command_timeout_s:
            self._fallback_qd[:] = 0.0
            self._fallback_qdd[:] = 0.0
        self._last_call_s = now

        converted = teleop_eef_to_actions(
            ee_command,
            kinematics=self._kinematics,
            get_joint_positions=self._get_joint_positions,
            gripper_open=self._gripper_open,
            gripper_close=self._gripper_close,
            min_delta=self._min_delta,
        )
        if converted is None:
            return None
        q_target = np.asarray(converted["actions"], dtype=np.float64).reshape(-1)
        if q_target.shape[0] < 7:
            return converted
        q_raw = np.asarray(self._get_joint_positions(), dtype=np.float64).reshape(-1)
        if q_raw.shape[0] < 7:
            return converted
        q_target[:6] = self._apply_internal_smoother(q_raw[:6], q_target[:6], self._dt)
        return {"actions": q_target.astype(np.float32)}

    def _apply_internal_smoother(self, q_current: np.ndarray, q_target: np.ndarray, dt: float) -> np.ndarray:
        """Rate-limit target joints using velocity/acceleration/jerk constraints."""
        dt = max(float(dt), 1e-3)
        v_lim = self._joint_velocity_limits
        a_lim = self._joint_acceleration_limits
        j_lim = self._joint_jerk_limits

        v_des = np.clip((q_target - q_current) / dt, -v_lim, v_lim)
        a_des = np.clip((v_des - self._fallback_qd) / dt, -a_lim, a_lim)

        da = a_des - self._fallback_qdd
        da_lim = j_lim * dt
        a_cmd = self._fallback_qdd + np.clip(da, -da_lim, da_lim)

        v_cmd = np.clip(self._fallback_qd + a_cmd * dt, -v_lim, v_lim)
        q_cmd = q_current + v_cmd * dt

        self._fallback_qd = v_cmd
        self._fallback_qdd = a_cmd
        return q_cmd


def _as_limit_array(
    value: np.ndarray | list[float] | None,
    default: np.ndarray,
    name: str,
) -> np.ndarray:
    if value is None:
        return default.copy()
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape[0] != default.shape[0]:
        raise ValueError(f"{name} must contain {default.shape[0]} entries, got {arr.shape[0]}")
    return np.maximum(arr, 1e-6)


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
    min_delta: float = 1e-4,
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

    # if the delta is less than the min_delta, return None
    if (
        max(abs(delta_x), abs(delta_y), abs(delta_z), abs(delta_roll), abs(delta_pitch), abs(delta_yaw))
        < min_delta
    ):
        return None

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
