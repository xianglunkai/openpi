from __future__ import annotations

from typing import TYPE_CHECKING

from rclpy.node import Node
from std_srvs.srv import Trigger

if TYPE_CHECKING:
    from pika_sync_ros import RolloutRuntimeContext


REQUEST_NEXT_EPISODE_SERVICE = "/request_next_episode"
RECORD_SUCCESS_SERVICE = "/record_success"
RECORD_FAILURE_SERVICE = "/record_failure"
RECORD_DONE_SERVICE = "/record_done"
ENTER_CRITICAL_PHASE_SERVICE = "/enter_critical_phase"
TOGGLE_CRITICAL_PHASE_SERVICE = "/toggle_critical_phase"
SET_CRITICAL_POLICY_ACTOR_SERVICE = "/select_critical_policy_actor"
SET_CRITICAL_POLICY_BASE_SERVICE = "/select_critical_policy_base"

SIGNAL_NEXT_EPISODE_REQUESTED = "next_episode_requested"
SIGNAL_MANUAL_SUCCESS_PENDING = "manual_success_pending"
SIGNAL_MANUAL_FAILURE_PENDING = "manual_failure_pending"
SIGNAL_MANUAL_DONE_PENDING = "manual_done_pending"
SIGNAL_CRITICAL_STARTED = "critical_started"
SIGNAL_SELECTED_CRITICAL_POLICY = "selected_critical_policy"
SIGNAL_EPISODE_CRITICAL_POLICY = "episode_critical_policy"
SIGNAL_TASK_MODE = "task_mode"


class ManualSignalBridge:
    def bind_runtime(self, runtime_context: RolloutRuntimeContext) -> Node:
        return ManualSignalBridgeNode(runtime_context)


class ManualSignalBridgeNode(Node):
    def __init__(self, runtime_context: RolloutRuntimeContext):
        super().__init__("rlt_manual_signal_bridge")
        self._runtime_context = runtime_context
        self.create_service(Trigger, REQUEST_NEXT_EPISODE_SERVICE, self._on_request_next_episode)
        self.create_service(Trigger, RECORD_SUCCESS_SERVICE, self._on_record_success)
        self.create_service(Trigger, RECORD_FAILURE_SERVICE, self._on_record_failure)
        self.create_service(Trigger, RECORD_DONE_SERVICE, self._on_record_done)
        self.create_service(Trigger, ENTER_CRITICAL_PHASE_SERVICE, self._on_enter_critical_phase)
        self.create_service(Trigger, TOGGLE_CRITICAL_PHASE_SERVICE, self._on_toggle_critical_phase)
        self.create_service(Trigger, SET_CRITICAL_POLICY_ACTOR_SERVICE, self._on_select_critical_policy_actor)
        self.create_service(Trigger, SET_CRITICAL_POLICY_BASE_SERVICE, self._on_select_critical_policy_base)
        self.get_logger().info(
            "Manual signal services ready: "
            f"{REQUEST_NEXT_EPISODE_SERVICE} "
            f"{RECORD_SUCCESS_SERVICE} "
            f"{RECORD_FAILURE_SERVICE} "
            f"{RECORD_DONE_SERVICE} "
            f"{ENTER_CRITICAL_PHASE_SERVICE} "
            f"{TOGGLE_CRITICAL_PHASE_SERVICE} "
            f"{SET_CRITICAL_POLICY_ACTOR_SERVICE} "
            f"{SET_CRITICAL_POLICY_BASE_SERVICE}"
        )

    def _on_request_next_episode(self, _request, response):
        self._runtime_context.request_next_episode()
        response.success = True
        response.message = "Next episode requested."
        return response

    def _on_record_success(self, _request, response):
        self._runtime_context.mark_manual_success()
        response.success = True
        response.message = "Manual success recorded."
        return response

    def _on_record_failure(self, _request, response):
        self._runtime_context.mark_manual_failure()
        response.success = True
        response.message = "Manual failure recorded."
        return response

    def _on_record_done(self, _request, response):
        self._runtime_context.mark_manual_done()
        response.success = True
        response.message = "Manual done recorded."
        return response

    def _on_enter_critical_phase(self, _request, response):
        if self._runtime_context.task_mode() == "critical_phase":
            response.success = True
            response.message = "Critical phase mode is already active for this episode."
            return response
        if self._runtime_context.enter_critical_phase():
            response.success = True
            response.message = "Entered critical phase."
            return response
        response.success = True
        response.message = "Critical phase was already active."
        return response

    def _on_toggle_critical_phase(self, _request, response):
        if self._runtime_context.task_mode() == "critical_phase":
            response.success = True
            response.message = "Critical phase task mode is fixed for this episode; toggle is ignored."
            return response
        active = self._runtime_context.toggle_critical_phase()
        response.success = True
        response.message = "Entered critical phase." if active else "Exited critical phase."
        return response

    def _on_select_critical_policy_actor(self, _request, response):
        self._runtime_context.set_selected_critical_policy_mode("actor")
        response.success = True
        response.message = "Selected critical policy mode=actor for the next episode."
        return response

    def _on_select_critical_policy_base(self, _request, response):
        self._runtime_context.set_selected_critical_policy_mode("base")
        response.success = True
        response.message = "Selected critical policy mode=base for the next episode."
        return response
