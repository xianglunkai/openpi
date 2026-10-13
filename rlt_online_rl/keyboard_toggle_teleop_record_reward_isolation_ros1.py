#!/usr/bin/env python3
import select
import sys
import termios
import time
import tty

import rospy
from speech_utils import log_say
from std_srvs.srv import Trigger

from train_deploy_alignment.manual_signal_bridge_ros1 import ENTER_CRITICAL_PHASE_SERVICE
from train_deploy_alignment.manual_signal_bridge_ros1 import RECORD_FAILURE_SERVICE
from train_deploy_alignment.manual_signal_bridge_ros1 import RECORD_SUCCESS_SERVICE
from train_deploy_alignment.manual_signal_bridge_ros1 import REQUEST_NEXT_EPISODE_SERVICE
from train_deploy_alignment.manual_signal_bridge_ros1 import TOGGLE_CRITICAL_PHASE_SERVICE

RL_TELEOP_TRIGGER_SERVICE = "/teleop_trigger_rl"
HW_TELEOP_TRIGGER_SERVICE = "/teleop_trigger"
TELEOP_STATUS_SERVICE = "/teleop_status"
SHUTDOWN_ROLLOUT_SERVICE = "/shutdown_rollout"
HW_TELEOP_SETTLE_SEC = 0.5
# Ignore repeated Space/t within this window (keyboard auto-repeat / double-tap).
TELEOP_TOGGLE_DEBOUNCE_SEC = 0.5
# After a handled toggle, discard any extra Space/t already buffered by the OS.
TELEOP_TOGGLE_DRAIN_SEC = 0.15

def getch():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def getkey():
    ch = getch()
    if ch != "\x1b":
        return ch
    seq = ch
    for _ in range(2):
        ready, _, _ = select.select([sys.stdin], [], [], 0.01)
        if not ready:
            break
        seq += sys.stdin.read(1)
    if seq == "\x1b[C":
        return "RIGHT"
    return ch

def drain_pending_toggle_keys(*, window_sec: float = TELEOP_TOGGLE_DRAIN_SEC) -> None:
    """Drop buffered Space/t key repeats so one physical press does not toggle twice."""
    deadline = time.monotonic() + max(float(window_sec), 0.0)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while time.monotonic() < deadline:
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch not in {" ", "t"}:
                rospy.logdebug("Stopped toggle key drain after unexpected key %r.", ch)
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class KeyboardTeleopRecordRewardToggleRos1:
    def __init__(self):
        self._last_announced_mode = None
        self.control_mode = "unknown"
        self._last_teleop_toggle_at = 0.0
        self._hw_teleop_available = self._probe_hw_teleop_service()
        self.rl_teleop_cli = self._make_client(RL_TELEOP_TRIGGER_SERVICE)
        self.hw_teleop_cli = self._make_client(HW_TELEOP_TRIGGER_SERVICE)
        self.teleop_status_cli = self._make_client(TELEOP_STATUS_SERVICE)
        self.shutdown_rollout_cli = self._make_client(SHUTDOWN_ROLLOUT_SERVICE)
        self.next_episode_cli = self._make_client(REQUEST_NEXT_EPISODE_SERVICE)
        self.success_cli = self._make_client(RECORD_SUCCESS_SERVICE)
        self.failure_cli = self._make_client(RECORD_FAILURE_SERVICE)
        self.critical_phase_cli = self._make_client(ENTER_CRITICAL_PHASE_SERVICE)
        self.toggle_critical_phase_cli = self._make_client(TOGGLE_CRITICAL_PHASE_SERVICE)
        self.refresh_teleop_mode(retries=5, timeout_sec=1.5)
        rospy.loginfo(self._ready_message())
        log_say("Teleoperation keyboard ready.")

    @staticmethod
    def _make_client(name: str):
        rospy.loginfo("Waiting for service %s ...", name)
        # rospy.wait_for_service(name)
        return rospy.ServiceProxy(name, Trigger)

    @staticmethod
    def _probe_hw_teleop_service(*, timeout_sec: float = 2.0) -> bool:
        try:
            rospy.wait_for_service(HW_TELEOP_TRIGGER_SERVICE, timeout=timeout_sec)
            rospy.loginfo("Hardware teleop service %s is available.", HW_TELEOP_TRIGGER_SERVICE)
            return True
        except rospy.ROSException:
            rospy.logwarn(
                "Hardware teleop service %s unavailable; toggles use %s only (SpaceMouse / policy).",
                HW_TELEOP_TRIGGER_SERVICE,
                RL_TELEOP_TRIGGER_SERVICE,
            )
            return False

    def _ready_message(self) -> str:
        return (
            "Ready. Press Space (or 't') to toggle teleop. Press Right Arrow (or 'o') to start the next episode. "
            "Press 's' to end the episode with success. Press 'f' to end the episode with failure. "
            "Press 'c' to enter the critical phase. Press 'x' to toggle critical phase. "
            "Press 'q' to request rollout shutdown and quit."
        )

    @staticmethod
    def _parse_mode_message(message: str | None):
        text = (message or "").strip().lower()
        if "mode=teleop" in text:
            return "teleop"
        if "mode=policy" in text:
            return "policy"
        if "mode=reset" in text:
            return "reset"
        return None

    def log_teleop_mode(self):
        rospy.loginfo("Current control mode: %s", self.control_mode)
        if self.control_mode in {"teleop", "policy", "reset"} and self.control_mode != self._last_announced_mode:
            log_say(f"Control mode {self.control_mode}.")
            self._last_announced_mode = self.control_mode

    def refresh_teleop_mode(self, *, retries: int = 3, timeout_sec: float = 1.0):
        total_attempts = max(int(retries), 1)
        for attempt in range(1, total_attempts + 1):
            try:
                rospy.wait_for_service(TELEOP_STATUS_SERVICE, timeout=timeout_sec)
                resp = self.teleop_status_cli()
            except (rospy.ROSException, rospy.ServiceException) as exc:
                if attempt < total_attempts:
                    rospy.logwarn(
                        "Failed to query teleop status (attempt %s/%s): %s; retrying.",
                        attempt,
                        total_attempts,
                        exc,
                    )
                    continue
                self.control_mode = "unknown"
                rospy.logerr("Failed to query teleop status: %s", exc)
                return False

            parsed = self._parse_mode_message(resp.message)
            if parsed is None:
                self.control_mode = "unknown"
                rospy.logerr("Unexpected teleop status response: %r", resp.message)
                return False

            self.control_mode = parsed
            self.log_teleop_mode()
            return True
        return False

    def _call_trigger(self, client, failure_message: str):
        try:
            return client()
        except rospy.ServiceException as exc:
            rospy.logerr("%s (%s)", failure_message, exc)
            return None

    def _toggle_local_teleop(self) -> bool:
        resp = self._call_trigger(self.rl_teleop_cli, "Failed to toggle local teleop state.")
        if resp is None:
            return False
        if not resp.success:
            rospy.logwarn(resp.message if resp.message else "Local teleop toggle failed.")
            self.refresh_teleop_mode()
            return False

        parsed = self._parse_mode_message(resp.message)
        if parsed is None:
            rospy.logwarn("Could not parse teleop mode from response: %r", resp.message)
            self.refresh_teleop_mode()
        else:
            self.control_mode = parsed
            self.log_teleop_mode()
        rospy.loginfo(resp.message if resp.message else "Local teleop toggle succeeded.")
        return True

    def _toggle_hardware_teleop(self, *, reason: str) -> bool:
        if not self._hw_teleop_available:
            return True
        resp = self._call_trigger(self.hw_teleop_cli, f"Failed to toggle hardware teleop for {reason}.")
        if resp is None:
            return False
        message = resp.message if resp.message else f"Hardware teleop toggled for {reason}."
        rospy.loginfo(message)
        return True

    def _record_terminal(self, client, label: str) -> bool:
        if not self.refresh_teleop_mode():
            return False
        if self.control_mode == "reset":
            rospy.logwarn("Cannot record %s: episode inactive/reset in progress.", label)
            log_say(f"Cannot record {label}. Episode inactive.")
            return False
        if self.control_mode == "teleop":
            if not self._toggle_hardware_teleop(reason=f"{label} end"):
                return False
            time.sleep(HW_TELEOP_SETTLE_SEC)
            if not self._toggle_local_teleop():
                return False

        resp = self._call_trigger(client, f"Failed to record {label}.")
        if resp is None:
            return False
        if resp.success:
            rospy.loginfo(resp.message if resp.message else f"Recorded {label}.")
            log_say(f"Recorded {label}.")
        else:
            rospy.logwarn(resp.message if resp.message else f"Recording {label} failed.")
            log_say(f"Record {label} failed.")
        self.refresh_teleop_mode()
        return bool(resp.success)

    def toggle_teleop(self):
        now = time.monotonic()
        elapsed = now - self._last_teleop_toggle_at
        if elapsed < TELEOP_TOGGLE_DEBOUNCE_SEC:
            rospy.logwarn(
                "Teleop toggle ignored (debounce %.2fs remaining). Press Space/t once, then wait.",
                TELEOP_TOGGLE_DEBOUNCE_SEC - elapsed,
            )
            return
        self._last_teleop_toggle_at = now

        if not self.refresh_teleop_mode():
            return
        if self.control_mode == "reset":
            rospy.logwarn("Episode inactive/reset in progress; teleop toggle ignored.")
            log_say("Episode inactive. Toggle ignored.")
            return

        if self.control_mode == "teleop":
            if not self._toggle_hardware_teleop(reason="teleop exit"):
                return
            time.sleep(HW_TELEOP_SETTLE_SEC)
            if not self._toggle_local_teleop():
                return
        else:
            if not self._toggle_local_teleop():
                return
            if not self._toggle_hardware_teleop(reason="teleop entry"):
                return
        self.refresh_teleop_mode()

    def request_next_episode(self):
        resp = self._call_trigger(self.next_episode_cli, "Failed to request next episode start.")
        if resp is None:
            return
        if resp.success:
            rospy.loginfo(resp.message if resp.message else "Requested next episode start.")
            log_say("Next episode requested.")
            return
        rospy.logwarn(resp.message if resp.message else "Next episode request failed.")
        log_say("Next episode request failed.")

    def record_success(self):
        if self._record_terminal(self.success_cli, "success"):
            rospy.loginfo("Episode ended (success). Rollout will reset and wait for 'o' if still running.")

    def record_failure(self):
        if self._record_terminal(self.failure_cli, "failure"):
            rospy.loginfo(
                "Episode ended (failure). Rollout moves the arm to reset, then waits for 'o' for the next episode."
            )
            log_say("Failure recorded. Robot resetting. Press o for next episode.")

    def enter_critical_phase(self):
        resp = self._call_trigger(self.critical_phase_cli, "Failed to enter the critical phase.")
        if resp is None:
            return
        if resp.success:
            rospy.loginfo(resp.message if resp.message else "Entered the critical phase.")
            log_say("Entered critical phase.")
            return
        rospy.logwarn(resp.message if resp.message else "Entering the critical phase failed.")
        log_say("Enter critical phase failed.")
    
    def toggle_critical_phase(self):
        resp = self._call_trigger(self.toggle_critical_phase_cli, "Failed to toggle the critical phase.")
        if resp is None:
            return
        if resp.success:
            rospy.loginfo(resp.message if resp.message else "Toggled the critical phase.")
            log_say("Critical phase toggled.")
            return
        rospy.logwarn(resp.message if resp.message else "Toggling the critical phase failed.")
        log_say("Toggle critical phase failed.")

    def request_rollout_shutdown(self):
        resp = self._call_trigger(self.shutdown_rollout_cli, "Failed to request rollout shutdown.")
        if resp is None:
            return
        if resp.success:
            rospy.loginfo(resp.message if resp.message else "Requested rollout shutdown.")
            log_say("Rollout shutdown requested.")
            return
        rospy.logwarn(resp.message if resp.message else "Rollout shutdown request failed.")
        log_say("Rollout shutdown failed.")


def main():
    rospy.init_node("keyboard_teleop_record_reward_toggle_ros1", anonymous=True)
    node = KeyboardTeleopRecordRewardToggleRos1()
    try:
        while not rospy.is_shutdown():
            ch = getkey()
            if ch in {" ", "t"}:
                node.toggle_teleop()
                drain_pending_toggle_keys()
            elif ch in {"RIGHT", "o"}:
                node.request_next_episode()
            elif ch == "s":
                node.record_success()
            elif ch == "f":
                node.record_failure()
            elif ch == "c":
                node.enter_critical_phase()
            elif ch == "x":
                node.toggle_critical_phase()
            elif ch == "q":
                node.request_rollout_shutdown()
                break
    finally:
        rospy.loginfo("Shutting down keyboard teleop toggle node.")
        pass


if __name__ == "__main__":
    main()

