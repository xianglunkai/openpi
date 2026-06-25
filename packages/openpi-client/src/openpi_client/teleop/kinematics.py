"""End-effector kinematics for teleop (lerobot RobotKinematics / agilex_cobot)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class EefKinematicsConfig:
    """Configuration matching agilex_cobot RobotKinematics setup."""

    urdf_path: str = "/home/xlk/work/lerobot/examples/hil-serl/aloha_new_description/urdf"
    target_frame_name: str = "gripper_frame_link"
    joint_names: List[str] = field(
        default_factory=lambda: [
            "right_joint0",
            "right_joint1",
            "right_joint2",
            "right_joint3",
            "right_joint4",
            "right_joint5",
            'gripper_frame_joint'
        ]
    )
    use_rad: bool = True
    ik_position_weight: float = 1.0
    ik_orientation_weight: float = 0.5


class EefKinematics:
    """Forward / inverse kinematics via placo (same as lerobot.model.kinematics.RobotKinematics)."""

    def __init__(
        self,
        urdf_path: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: Optional[List[str]] = None,
        *,
        use_rad: bool = True,
        ik_position_weight: float = 1.0,
        ik_orientation_weight: float = 0.5,
    ) -> None:
        try:
            import placo  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "placo is required for EefKinematics. Install lerobot kinematics extras or placo."
            ) from e

        self._placo = placo
        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)

        self.use_rad = use_rad
        self.target_frame_name = target_frame_name
        self.ik_position_weight = ik_position_weight
        self.ik_orientation_weight = ik_orientation_weight

        # If joint_names is None, use all joints from the URDF (including gripper); otherwise use provided list.
        self.with_gripper_joint_names = list(self.robot.joint_names()) if joint_names is None else joint_names
        
        # Remove gripper joint from joint_names if it's included, since FK/IK will only consider arm joints. Assumes gripper joint is last in the list.
        self.joint_names = self.with_gripper_joint_names[:-1]

        # Add tip frame as a task to the solver. This frame will be used for IK target pose.
        self.tip_frame = self.solver.add_frame_task(self.target_frame_name, np.eye(4))

    @classmethod
    def from_config(cls, config: EefKinematicsConfig) -> EefKinematics:
        return cls(
            config.urdf_path,
            config.target_frame_name,
            config.joint_names,
            use_rad=config.use_rad,
            ik_position_weight=config.ik_position_weight,
            ik_orientation_weight=config.ik_orientation_weight,
        )

    @classmethod
    def for_agilex_cobot(
        cls,
        urdf_path: str | None = None,
        *,
        target_frame_name: str = "gripper_frame_link",
        joint_names: Optional[List[str]] = None,
        use_rad: bool = True,
    ) -> EefKinematics:
        """Factory with agilex_cobot defaults."""
        cfg = EefKinematicsConfig(
            urdf_path= urdf_path or EefKinematicsConfig.urdf_path,
            target_frame_name=target_frame_name,
            joint_names=joint_names or EefKinematicsConfig().joint_names,
            use_rad=use_rad,
        )
        return cls.from_config(cfg)

    def forward_kinematics(self, joint_pos: np.ndarray) -> np.ndarray:
        """Compute FK; joint_pos is arm joints only (no gripper), radians if use_rad."""
        joint_pos = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        if self.use_rad:
            joint_pos_rad = joint_pos[: len(self.joint_names)]
        else:
            joint_pos_rad = np.deg2rad(joint_pos[: len(self.joint_names)])

        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, float(joint_pos_rad[i]))

        self.robot.update_kinematics()
        return self.robot.get_T_world_frame(self.target_frame_name)

    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float | None = None,
        orientation_weight: float | None = None,
    ) -> np.ndarray:
        """Compute IK; returns arm joint positions (same units as input, radians if use_rad)."""
        current_joint_pos = np.asarray(current_joint_pos, dtype=np.float64).reshape(-1)
        position_weight = self.ik_position_weight if position_weight is None else position_weight
        orientation_weight = self.ik_orientation_weight if orientation_weight is None else orientation_weight

        if self.use_rad:
            current_joint_rad = current_joint_pos[: len(self.joint_names)]
        else:
            current_joint_rad = np.deg2rad(current_joint_pos[: len(self.joint_names)])

        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, float(current_joint_rad[i]))

        self.tip_frame.T_world_frame = desired_ee_pose
        self.tip_frame.configure(self.target_frame_name, "soft", position_weight, orientation_weight)

        self.solver.solve(True)
        self.robot.update_kinematics()

        joint_pos_rad = [self.robot.get_joint(name) for name in self.joint_names]

        if self.use_rad:
            joint_pos_out = np.array(joint_pos_rad, dtype=np.float64)
        else:
            joint_pos_out = np.rad2deg(joint_pos_rad)

        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[: len(self.joint_names)] = joint_pos_out
            result[len(self.joint_names) :] = current_joint_pos[len(self.joint_names) :]
            return result
        return joint_pos_out

    def _arm_jacobian(self) -> np.ndarray:
        """6×n Jacobian for the controlled arm joints at the target frame."""
        try:
            j_full = self.robot.frame_jacobian(self.target_frame_name, "world")
            j_cols = [self.robot.get_joint_v_offset(joint_name) for joint_name in self.joint_names]
            return j_full[:, j_cols]
        except (TypeError, AttributeError, RuntimeError):
            # Older placo: joint_jacobian(joint) returns the chain Jacobian at that joint.
            return self.robot.joint_jacobian(self.joint_names[-1], "world")

    def get_damped_pinv(self, joint_pos_rad: np.ndarray, damping: float = 0.1) -> np.ndarray:
        """
        给定关节角(弧度)，返回当前构型的阻尼最小二乘伪逆 (6×n)。
        用于速度级遥操作。
        """
        # 更新 robot 到当前关节角
        if len(joint_pos_rad) != len(self.joint_names):
            raise ValueError(
                f"Expected {len(self.joint_names)} arm joint values, got {len(joint_pos_rad)}. "
                f"Arm joints: {self.joint_names}. "
                f"If you passed only arm names to joint_names, append the gripper "
                f"(e.g. fr_joint7) as the last element — RobotKinematics always excludes it."
            )
        for i, joint_name in enumerate(self.joint_names):
            self.robot.set_joint(joint_name, float(joint_pos_rad[i]))
        self.robot.update_kinematics()

        j = self._arm_jacobian()

        # 阻尼伪逆
        jj_t = j @ j.T
        damped_inv = j.T @ np.linalg.inv(jj_t + damping**2 * np.eye(6))
        return damped_inv