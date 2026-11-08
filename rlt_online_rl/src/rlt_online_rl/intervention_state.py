from __future__ import annotations

from collections.abc import Callable
import threading
import time
from typing import Any


class HumanInterventionState:
    """Thread-safe policy/human takeover state with a control-epoch publish gate."""

    def __init__(self, *, policy_enabled: bool = True):
        self._lock = threading.Lock()
        self._policy_enabled = policy_enabled
        self._episode_active = False
        self._need_reset_on_resume = False
        self._resume_until = 0.0
        self._control_epoch = 0

    def _mode_name_locked(self) -> str:
        if not self._episode_active:
            return "reset"
        return "policy" if self._policy_enabled else "teleop"

    def current_mode(self) -> str:
        with self._lock:
            return self._mode_name_locked()

    def is_policy_enabled(self) -> bool:
        with self._lock:
            return self._policy_enabled

    def in_resume_cooldown(self) -> bool:
        with self._lock:
            return time.time() < self._resume_until

    def policy_epoch(self) -> int:
        with self._lock:
            return self._control_epoch

    def _bump_epoch_locked(self) -> None:
        self._control_epoch += 1

    def toggle_policy(self, *, resume_delay_s: float) -> bool | None:
        with self._lock:
            if not self._episode_active:
                return None
            self._policy_enabled = not self._policy_enabled
            self._bump_epoch_locked()
            if self._policy_enabled:
                self._need_reset_on_resume = True
                self._resume_until = time.time() + max(float(resume_delay_s), 0.0)
            else:
                self._need_reset_on_resume = False
                self._resume_until = 0.0
            return self._policy_enabled

    def set_policy_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self._episode_active = True
            self._policy_enabled = bool(enabled)
            self._need_reset_on_resume = False
            self._resume_until = 0.0
            self._bump_epoch_locked()
            return self._policy_enabled

    def enter_episode_reset(self) -> str:
        with self._lock:
            self._episode_active = False
            self._policy_enabled = True
            self._need_reset_on_resume = False
            self._resume_until = 0.0
            self._bump_epoch_locked()
            return self._mode_name_locked()

    def consume_reset_request(self) -> bool:
        with self._lock:
            need = self._need_reset_on_resume
            self._need_reset_on_resume = False
            return need

    def dispatch_policy_action(
        self,
        expected_epoch: int,
        command: Callable[[], Any],
    ) -> tuple[bool, Any]:
        """Publish only if this caller still owns the control epoch.

        The check and the local send share one lock, so a takeover cannot land
        between them. ``command`` must not re-enter this state.
        """
        with self._lock:
            if (
                not self._episode_active
                or not self._policy_enabled
                or self._control_epoch != int(expected_epoch)
                or time.time() < self._resume_until
            ):
                return False, None
            return True, command()
