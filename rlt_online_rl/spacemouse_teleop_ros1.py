#!/usr/bin/env python3
"""ROS1 SpaceMouse teleop publisher for AgileX single-arm rollout."""

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from openpi_client.teleop import EefActionConverter
from openpi_client.teleop import EefKinematics
from openpi_client.teleop import SpacemouseTeleop
from openpi_client.teleop import SpacemouseTeleopConfig

TELEOP_TRIGGER_SERVICE = "/teleop_trigger_rl"
TELEOP_STATUS_SERVICE = "/teleop_status"


def _parse_mode_message(message: str | None) -> str | None:
    text = (message or "").strip().lower()
    if "mode=teleop" in text:
        return "teleop"
    if "mode=policy" in text:
        return "policy"
    if "mode=reset" in text:
        return "reset"
    return None


class JointStateCache:
    def __init__(self, topic: str):
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._sub = rospy.Subscriber(topic, JointState, self._on_joint_state, queue_size=100)

    def _on_joint_state(self, msg: JointState) -> None:
        q = np.asarray(msg.position, dtype=np.float64).reshape(-1)
        if q.shape[0] < 7:
            return
        with self._lock:
            self._latest = q[:7].copy()

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def wait_ready(self, timeout_s: float) -> np.ndarray:
        deadline = time.time() + timeout_s
        while not rospy.is_shutdown():
            latest = self.get()
            if latest is not None:
                return latest
            if time.time() > deadline:
                raise TimeoutError("Timeout waiting first JointState.")
            time.sleep(0.02)
        raise RuntimeError("ROS shutdown while waiting JointState.")

    def shutdown(self) -> None:
        self._sub.unregister()


class TeleopModeClient:
    def __init__(self, trigger_service: str, status_service: str):
        rospy.wait_for_service(trigger_service)
        rospy.wait_for_service(status_service)
        self._trigger = rospy.ServiceProxy(trigger_service, Trigger)
        self._status = rospy.ServiceProxy(status_service, Trigger)

    def get_mode(self) -> str:
        try:
            resp = self._status()
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed teleop status query: %s", exc)
            return "unknown"
        parsed = _parse_mode_message(resp.message)
        return parsed if parsed is not None else "unknown"

    def toggle(self) -> bool:
        try:
            resp = self._trigger()
        except rospy.ServiceException as exc:
            rospy.logwarn("Failed teleop toggle: %s", exc)
            return False
        rospy.loginfo(resp.message if resp.message else "teleop toggled")
        return bool(resp.success)

    def ensure_mode(self, target_mode: str) -> bool:
        mode = self.get_mode()
        if mode == target_mode:
            return True
        if mode == "reset":
            return False
        if not self.toggle():
            return False
        mode_after = self.get_mode()
        return mode_after == target_mode


def _joint_names_for_arm(arm: str) -> list[str]:
    if arm == "left":
        return [f"left_joint{i}" for i in range(6)]
    return [f"right_joint{i}" for i in range(6)]


def _is_intervening(action: dict, deadzone: float) -> bool:
    axes = [abs(float(action.get(k, 0.0))) for k in ("delta_x", "delta_y", "delta_z", "delta_roll", "delta_pitch", "delta_yaw")]
    return max(axes) >= float(deadzone)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpaceMouse teleop for ROS1 AgileX single-arm.")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--joint_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--deadzone", type=float, default=0.0005)
    parser.add_argument("--eef_cutoff_freq", type=float, default=3.0)
    parser.add_argument("--auto_toggle_teleop", action="store_true")
    parser.add_argument("--release_to_policy_delay_s", type=float, default=0.4)
    parser.add_argument("--teleop_trigger_service", type=str, default=TELEOP_TRIGGER_SERVICE)
    parser.add_argument("--teleop_status_service", type=str, default=TELEOP_STATUS_SERVICE)
    parser.add_argument("--joint_state_timeout_s", type=float, default=10.0)
    parser.add_argument("--urdf_path", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rospy.init_node("spacemouse_teleop_ros1", anonymous=True)

    pub = rospy.Publisher(args.cmd_topic, JointState, queue_size=50)
    joint_cache = JointStateCache(args.joint_topic)
    joint_cache.wait_ready(timeout_s=float(args.joint_state_timeout_s))

    kinematics = EefKinematics.for_agilex_cobot(
        urdf_path=args.urdf_path or None,
        joint_names=_joint_names_for_arm(args.arm),
        use_rad=True,
    )
    converter = EefActionConverter(
        kinematics=kinematics,
        get_joint_positions=lambda: joint_cache.get() if joint_cache.get() is not None else np.zeros((7,), dtype=np.float64),
    )
    teleop = SpacemouseTeleop(
        SpacemouseTeleopConfig(
            fps=float(args.fps),
            deadzone=float(args.deadzone),
            eef_cutoff_freq=float(args.eef_cutoff_freq),
            use_gripper=True,
        )
    )
    mode_client = None
    if args.auto_toggle_teleop:
        mode_client = TeleopModeClient(args.teleop_trigger_service, args.teleop_status_service)

    teleop.connect()
    rospy.loginfo(
        "SpaceMouse teleop started cmd_topic=%s joint_topic=%s arm=%s auto_toggle=%s",
        args.cmd_topic,
        args.joint_topic,
        args.arm,
        bool(args.auto_toggle_teleop),
    )
    rate = rospy.Rate(max(float(args.fps), 1.0))
    last_active_s = 0.0
    try:
        while not rospy.is_shutdown():
            action = teleop.get_action()
            active = _is_intervening(action, float(args.deadzone))
            if active:
                last_active_s = time.time()

            if mode_client is not None:
                if active:
                    mode_client.ensure_mode("teleop")
                elif time.time() - last_active_s >= float(args.release_to_policy_delay_s):
                    mode_client.ensure_mode("policy")

            converted = converter(action)
            if converted is not None and "actions" in converted:
                target = np.asarray(converted["actions"], dtype=np.float64).reshape(-1)
                if target.shape[0] >= 7:
                    msg = JointState()
                    msg.header = Header(stamp=rospy.Time.now())
                    msg.name = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
                    msg.position = target[:7].tolist()
                    pub.publish(msg)
            rate.sleep()
    finally:
        teleop.disconnect()
        joint_cache.shutdown()


if __name__ == "__main__":
    main()

