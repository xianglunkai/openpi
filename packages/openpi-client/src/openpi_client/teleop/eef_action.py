"""Convert SpaceMouse EEF deltas to joint actions (agilex_cobot send_action_from_eef)."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from openpi_client.teleop.kinematics import EefKinematics


def _euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    try:
        import transforms3d as t3d

        return t3d.euler.euler2mat(roll, pitch, yaw)
    except ImportError:
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
    ) -> None:
        self._kinematics = kinematics
        self._get_joint_positions = get_joint_positions
        self._gripper_open = gripper_open
        self._gripper_close = gripper_close
        self._min_delta = min_delta

    def __call__(self, ee_command: dict) -> Optional[dict]:
        return teleop_eef_to_actions(
            ee_command,
            kinematics=self._kinematics,
            get_joint_positions=self._get_joint_positions,
            gripper_open=self._gripper_open,
            gripper_close=self._gripper_close,
            min_delta=self._min_delta,
        )


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

    if (
        max(abs(delta_x), abs(delta_y), abs(delta_z), abs(delta_roll), abs(delta_pitch), abs(delta_yaw))
        < min_delta
        and gripper == 1.0
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

    if gripper == 0.0:
        q_target[-1] = gripper_close
    elif gripper == 1.0:
        q_target[-1] = q_raw[-1]
    elif gripper == 2.0:
        q_target[-1] = gripper_open
    else:
        q_target[-1] = q_raw[-1]

    return {"actions": q_target.astype(np.float32)}
