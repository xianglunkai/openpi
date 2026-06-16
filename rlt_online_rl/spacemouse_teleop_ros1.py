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
from speech_utils import log_say

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


class TeleopStatusClient:
    """Read rollout control mode; mode changes are done via keyboard (t / Space)."""

    def __init__(self, status_service: str, *, wait_timeout_s: float = 30.0):
        self._status_service = status_service
        self._status: rospy.ServiceProxy | None = None
        self._last_warn_s = 0.0
        self._warn_interval_s = 5.0
        self._connect(wait_timeout_s=wait_timeout_s)

    def _warn_throttled(self, message: str) -> None:
        now = time.time()
        if now - self._last_warn_s < self._warn_interval_s:
            return
        self._last_warn_s = now
        rospy.logwarn(message)

    def _connect(self, *, wait_timeout_s: float) -> bool:
        try:
            rospy.wait_for_service(self._status_service, timeout=wait_timeout_s)
        except rospy.ROSException:
            self._status = None
            self._warn_throttled(
                f"Rollout teleop status unavailable ({self._status_service}). "
                "Start rollout and use keyboard 't' or Space to toggle teleop."
            )
            return False
        self._status = rospy.ServiceProxy(self._status_service, Trigger)
        return True

    @property
    def available(self) -> bool:
        return self._status is not None

    def get_mode(self) -> str:
        if not self.available:
            return "unknown"
        try:
            resp = self._status()
        except rospy.ServiceException as exc:
            self._warn_throttled(f"Failed teleop status query: {exc}")
            return "unknown"
        parsed = _parse_mode_message(resp.message)
        return parsed if parsed is not None else "unknown"


def _joint_names_for_arm(arm: str) -> list[str]:
    if arm == "left":
        return [f"left_joint{i}" for i in range(6)]
    return [f"right_joint{i}" for i in range(6)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SpaceMouse teleop for ROS1 AgileX single-arm.")
    parser.add_argument("--arm", choices=("left", "right"), default="right")
    parser.add_argument("--joint_topic", type=str, default="/puppet/joint_right")
    parser.add_argument("--cmd_topic", type=str, default="/master/joint_right")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--space_mouse_deadzone", type=float, default=0.0)
    parser.add_argument("--teleop_status_service", type=str, default=TELEOP_STATUS_SERVICE)
    parser.add_argument("--joint_state_timeout_s", type=float, default=0.2)
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

    def _get_joint_positions() -> np.ndarray:
        latest = joint_cache.get()
        if latest is None:
            return np.zeros((7,), dtype=np.float64)
        return latest

    control_fps = max(float(args.fps), 1.0)
    converter = EefActionConverter(
        kinematics=kinematics,
        get_joint_positions=_get_joint_positions,
        input_fps=control_fps,
    )
    teleop = SpacemouseTeleop(
        SpacemouseTeleopConfig(
            deadzone=float(args.space_mouse_deadzone),
            use_gripper=True,
        )
    )
    status_client = TeleopStatusClient(args.teleop_status_service)

    teleop.connect()
    rospy.loginfo(
        "SpaceMouse teleop started cmd_topic=%s joint_topic=%s arm=%s",
        args.cmd_topic,
        args.joint_topic,
        args.arm,
    )
    log_say("SpaceMouse teleop started. Press t or Space on the keyboard to toggle teleop.")
    if not status_client.available:
        rospy.logwarn(
            "Teleop status unavailable: commands are published continuously; rollout gates execution. "
            "Press keyboard 't' or Space to switch to teleop mode once rollout is running."
        )
    rate = rospy.Rate(control_fps)
    last_announced_mode: str | None = None
    last_status_retry_s = 0.0
    status_retry_interval_s = 0.2
    last_cmd: np.ndarray | None = None
    try:
        while not rospy.is_shutdown():
            action = teleop.get_action()

            publish_allowed = False
            now = time.time()
            if not status_client.available and now - last_status_retry_s >= status_retry_interval_s:
                status_client._connect(wait_timeout_s=status_retry_interval_s)
                last_status_retry_s = now
            if status_client.available:
                mode = status_client.get_mode()
                if mode in {"teleop", "policy", "reset"} and mode != last_announced_mode:
                    log_say(f"Control mode {mode}.")
                    last_announced_mode = mode
                publish_allowed = mode == "teleop"

            converted = converter(action)
            target: np.ndarray | None = None
            if converted is not None and "actions" in converted:
                target = np.asarray(converted["actions"], dtype=np.float64).reshape(-1)
                if target.shape[0] >= 7:
                    last_cmd = target[:7].copy()
            elif publish_allowed and last_cmd is not None:
                target = last_cmd

            if publish_allowed and target is not None and target.shape[0] >= 7:
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
