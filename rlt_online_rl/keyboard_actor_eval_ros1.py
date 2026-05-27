#!/usr/bin/env python3
import rospy
from std_srvs.srv import Trigger

from keyboard_toggle_teleop_record_reward_isolation_ros1 import getkey
from keyboard_toggle_teleop_record_reward_isolation_ros1 import KeyboardTeleopRecordRewardToggleRos1
from train_deploy_alignment.manual_signal_bridge_ros1 import RECORD_DONE_SERVICE
from train_deploy_alignment.manual_signal_bridge_ros1 import SET_CRITICAL_POLICY_ACTOR_SERVICE
from train_deploy_alignment.manual_signal_bridge_ros1 import SET_CRITICAL_POLICY_BASE_SERVICE
from train_deploy_alignment.manual_signal_bridge_ros1 import TOGGLE_CRITICAL_PHASE_SERVICE


class KeyboardActorEvalRos1(KeyboardTeleopRecordRewardToggleRos1):
    def _ready_message(self) -> str:
        return (
            "Eval ready. Press 'a' for actor refine, 'b' for Machine A only, Right Arrow (or 'o') to start the next episode, "
            "'c' to toggle critical on/off in full_task, 's' to end/reset the episode, "
            "Space (or 't') to toggle teleop, 'q' to quit."
        )

    def __init__(self):
        super().__init__()
        self.done_cli = self._make_client(RECORD_DONE_SERVICE)
        self.toggle_critical_cli = self._make_client(TOGGLE_CRITICAL_PHASE_SERVICE)
        self.select_actor_cli = self._make_client(SET_CRITICAL_POLICY_ACTOR_SERVICE)
        self.select_base_cli = self._make_client(SET_CRITICAL_POLICY_BASE_SERVICE)

    def reset_episode(self):
        self._record_terminal(self.done_cli, "done")

    def select_actor(self):
        resp = self._call_trigger(self.select_actor_cli, "Failed to select actor critical policy mode.")
        if resp is None:
            return
        rospy.loginfo(resp.message if resp.message else "Selected critical policy mode=actor.")

    def select_base(self):
        resp = self._call_trigger(self.select_base_cli, "Failed to select base critical policy mode.")
        if resp is None:
            return
        rospy.loginfo(resp.message if resp.message else "Selected critical policy mode=base.")

    def toggle_critical_phase(self):
        resp = self._call_trigger(self.toggle_critical_cli, "Failed to toggle the critical phase.")
        if resp is None:
            return
        rospy.loginfo(resp.message if resp.message else "Toggled critical phase.")


def main():
    rospy.init_node("keyboard_actor_eval_ros1", anonymous=True)
    node = KeyboardActorEvalRos1()

    try:
        while not rospy.is_shutdown():
            ch = getkey()
            if ch in {" ", "t"}:
                node.toggle_teleop()
            elif ch in {"RIGHT", "o"}:
                node.request_next_episode()
            elif ch == "s":
                node.reset_episode()
            elif ch == "c":
                node.toggle_critical_phase()
            elif ch == "a":
                node.select_actor()
            elif ch == "b":
                node.select_base()
            elif ch == "q":
                break
    finally:
        pass


if __name__ == "__main__":
    main()

