from __future__ import annotations

from typing import TYPE_CHECKING

import rospy
from std_srvs.srv import Trigger
from std_srvs.srv import TriggerResponse

if TYPE_CHECKING:
    from pika_sync_ros1_agilex_single_arm import RolloutRuntimeContext


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
    def bind_runtime(self, runtime_context: RolloutRuntimeContext) -> "ManualSignalBridgeServer":
        return ManualSignalBridgeServer(runtime_context)


class ManualSignalBridgeServer:
    def __init__(self, runtime_context: RolloutRuntimeContext):
        self._runtime_context = runtime_context
        self._services = [
            rospy.Service(REQUEST_NEXT_EPISODE_SERVICE, Trigger, self._on_request_next_episode),
            rospy.Service(RECORD_SUCCESS_SERVICE, Trigger, self._on_record_success),
            rospy.Service(RECORD_FAILURE_SERVICE, Trigger, self._on_record_failure),
            rospy.Service(RECORD_DONE_SERVICE, Trigger, self._on_record_done),
            rospy.Service(ENTER_CRITICAL_PHASE_SERVICE, Trigger, self._on_enter_critical_phase),
            rospy.Service(TOGGLE_CRITICAL_PHASE_SERVICE, Trigger, self._on_toggle_critical_phase),
            rospy.Service(SET_CRITICAL_POLICY_ACTOR_SERVICE, Trigger, self._on_select_critical_policy_actor),
            rospy.Service(SET_CRITICAL_POLICY_BASE_SERVICE, Trigger, self._on_select_critical_policy_base),
        ]
        rospy.loginfo(
            "Manual signal services ready: %s %s %s %s %s %s %s %s",
            REQUEST_NEXT_EPISODE_SERVICE,
            RECORD_SUCCESS_SERVICE,
            RECORD_FAILURE_SERVICE,
            RECORD_DONE_SERVICE,
            ENTER_CRITICAL_PHASE_SERVICE,
            TOGGLE_CRITICAL_PHASE_SERVICE,
            SET_CRITICAL_POLICY_ACTOR_SERVICE,
            SET_CRITICAL_POLICY_BASE_SERVICE,
        )

    def _on_request_next_episode(self, _request: Trigger.Request) -> TriggerResponse:
        self._runtime_context.request_next_episode()
        return TriggerResponse(success=True, message="Next episode requested.")

    def _on_record_success(self, _request: Trigger.Request) -> TriggerResponse:
        self._runtime_context.mark_manual_success()
        return TriggerResponse(success=True, message="Manual success recorded.")

    def _on_record_failure(self, _request: Trigger.Request) -> TriggerResponse:
        self._runtime_context.mark_manual_failure()
        return TriggerResponse(success=True, message="Manual failure recorded.")

    def _on_record_done(self, _request: Trigger.Request) -> TriggerResponse:
        self._runtime_context.mark_manual_done()
        return TriggerResponse(success=True, message="Manual done recorded.")

    def _on_enter_critical_phase(self, _request: Trigger.Request) -> TriggerResponse:
        if self._runtime_context.task_mode() == "critical_phase":
            return TriggerResponse(success=True, message="Critical phase mode is already active for this episode.")
        if self._runtime_context.enter_critical_phase():
            return TriggerResponse(success=True, message="Entered critical phase.")
        return TriggerResponse(success=True, message="Critical phase was already active.")

    def _on_toggle_critical_phase(self, _request: Trigger.Request) -> TriggerResponse:
        if self._runtime_context.task_mode() == "critical_phase":
            return TriggerResponse(
                success=True,
                message="Critical phase task mode is fixed for this episode; toggle is ignored.",
            )
        active = self._runtime_context.toggle_critical_phase()
        return TriggerResponse(success=True, message="Entered critical phase." if active else "Exited critical phase.")

    def _on_select_critical_policy_actor(self, _request: Trigger.Request) -> TriggerResponse:
        self._runtime_context.set_selected_critical_policy_mode("actor")
        return TriggerResponse(success=True, message="Selected critical policy mode=actor for the next episode.")

    def _on_select_critical_policy_base(self, _request: Trigger.Request) -> TriggerResponse:
        self._runtime_context.set_selected_critical_policy_mode("base")
        return TriggerResponse(success=True, message="Selected critical policy mode=base for the next episode.")

    def shutdown(self) -> None:
        for srv in self._services:
            srv.shutdown("shutdown")

