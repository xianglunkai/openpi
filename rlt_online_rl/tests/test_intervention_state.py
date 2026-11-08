from __future__ import annotations

from rlt_online_rl.intervention_state import HumanInterventionState


def test_stale_epoch_cannot_publish_after_takeover() -> None:
    state = HumanInterventionState()
    state.set_policy_enabled(True)
    owned_epoch = state.policy_epoch()
    published: list[str] = []

    assert state.dispatch_policy_action(owned_epoch, lambda: published.append("first") or "sent") == (True, "sent")

    assert state.toggle_policy(resume_delay_s=0.0) is False
    assert state.toggle_policy(resume_delay_s=0.0) is True
    assert state.dispatch_policy_action(owned_epoch, lambda: published.append("stale")) == (False, None)
    assert published == ["first"]

    current_epoch = state.policy_epoch()
    assert current_epoch != owned_epoch
    assert state.dispatch_policy_action(current_epoch, lambda: published.append("fresh") or "sent") == (True, "sent")
    assert published == ["first", "fresh"]


def test_takeover_during_resume_cooldown_rejects_the_new_epoch() -> None:
    state = HumanInterventionState()
    state.set_policy_enabled(True)
    assert state.toggle_policy(resume_delay_s=0.0) is False
    assert state.toggle_policy(resume_delay_s=30.0) is True
    assert state.dispatch_policy_action(state.policy_epoch(), lambda: "sent") == (False, None)
