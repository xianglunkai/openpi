#!/usr/bin/env python3
from keyboard_toggle_teleop_record_reward_isolation import KeyboardTeleopRecordRewardToggle
from keyboard_toggle_teleop_record_reward_isolation import getch
import rclpy
from std_srvs.srv import Trigger
from train_deploy_alignment.manual_signal_bridge import RECORD_DONE_SERVICE
from train_deploy_alignment.manual_signal_bridge import SET_CRITICAL_POLICY_ACTOR_SERVICE
from train_deploy_alignment.manual_signal_bridge import SET_CRITICAL_POLICY_BASE_SERVICE
from train_deploy_alignment.manual_signal_bridge import TOGGLE_CRITICAL_PHASE_SERVICE


class KeyboardActorEval(KeyboardTeleopRecordRewardToggle):
    def _ready_message(self) -> str:
        return (
            "Eval ready. Press 'a' for actor refine, 'b' for Machine A only, 'o' to start the next episode, "
            "'c' to toggle critical on/off in full_task, 's' to end/reset the episode, "
            "'t' to toggle teleop, 'q' to quit."
        )

    def __init__(self):
        super().__init__()
        self.done_cli = self.create_client(Trigger, RECORD_DONE_SERVICE)
        self.toggle_critical_cli = self.create_client(Trigger, TOGGLE_CRITICAL_PHASE_SERVICE)
        self.select_actor_cli = self.create_client(Trigger, SET_CRITICAL_POLICY_ACTOR_SERVICE)
        self.select_base_cli = self.create_client(Trigger, SET_CRITICAL_POLICY_BASE_SERVICE)

        self.done_cli.wait_for_service()
        self.toggle_critical_cli.wait_for_service()
        self.select_actor_cli.wait_for_service()
        self.select_base_cli.wait_for_service()

    def reset_episode(self):
        self._record_terminal(self.done_cli, "done")

    def select_actor(self):
        resp = self._call_trigger(self.select_actor_cli, "Failed to select actor critical policy mode.")
        if resp is None:
            return
        self.get_logger().info(resp.message if resp.message else "Selected critical policy mode=actor.")

    def select_base(self):
        resp = self._call_trigger(self.select_base_cli, "Failed to select base critical policy mode.")
        if resp is None:
            return
        self.get_logger().info(resp.message if resp.message else "Selected critical policy mode=base.")

    def toggle_critical_phase(self):
        resp = self._call_trigger(self.toggle_critical_cli, "Failed to toggle the critical phase.")
        if resp is None:
            return
        self.get_logger().info(resp.message if resp.message else "Toggled critical phase.")


def main():
    rclpy.init()
    node = KeyboardActorEval()

    try:
        while rclpy.ok():
            ch = getch()
            if ch == "t":
                node.toggle_teleop()
            elif ch == "o":
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
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
