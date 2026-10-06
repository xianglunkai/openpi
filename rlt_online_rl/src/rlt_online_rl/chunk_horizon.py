from __future__ import annotations

from rlt_online_rl.config import EnvDriverConfig
from rlt_online_rl.config import OnlineRLSystemConfig
from rlt_online_rl.config import RLTOnlineRLConfig


def policy_uses_rl_actor(
    *,
    episode_phase: str,
    in_critical_phase: bool,
    critical_policy_mode: str,
) -> bool:
    """True when the rollout should execute RL-actor-refined chunks (Evo critical phase)."""
    return (
        str(episode_phase) != "warmup"
        and bool(in_critical_phase)
        and str(critical_policy_mode) == "actor"
    )


def resolve_chunk_exec_horizon(
    env_config: EnvDriverConfig,
    rl_config: RLTOnlineRLConfig,
    *,
    uses_rl_actor: bool,
) -> int:
    """Steps to execute per chunk: VLA phase (25) vs RL phase (chunk_len, e.g. 10/20)."""
    if uses_rl_actor:
        if env_config.rl_chunk_exec_horizon is not None:
            return int(env_config.rl_chunk_exec_horizon)
        return int(rl_config.chunk_len)
    return int(env_config.vla_chunk_exec_horizon)


def resolve_ref_chunk_horizon(
    env_config: EnvDriverConfig,
    rl_config: RLTOnlineRLConfig,
    *,
    uses_rl_actor: bool,
) -> int:
    """How many VLA ref steps to request from Machine A for the current chunk.

    With RTC enabled we always request the longer VLA horizon so Machine A can
    run leftover-guided VLA inference (Evo always calls full pi0.5 predict).
    RL still only *executes* ``chunk_len`` actor steps — see EnvDriver planner.
    """
    if bool(getattr(env_config, "use_rtc", False)):
        return int(env_config.vla_chunk_exec_horizon)
    if uses_rl_actor:
        return int(rl_config.chunk_len)
    return int(env_config.vla_chunk_exec_horizon)


class ChunkHorizonEnvMixin:
    """Expose per-chunk VLA/RL horizons to EnvDriver during execute_chunk."""

    _system: OnlineRLSystemConfig
    _chunk_uses_rl_actor: bool
    _chunk_exec_horizon: int
    _chunk_ref_horizon: int

    def _uses_rl_actor_for_chunk(self) -> bool:
        return policy_uses_rl_actor(
            episode_phase=self._phase_controller.episode_phase,
            in_critical_phase=self._runtime_context.in_critical_phase(),
            critical_policy_mode=self._runtime_context.episode_critical_policy_mode(),
        )

    def refresh_chunk_horizon_state(self) -> None:
        uses_rl = self._uses_rl_actor_for_chunk()
        self._chunk_uses_rl_actor = uses_rl
        self._chunk_exec_horizon = resolve_chunk_exec_horizon(
            self._system.env_driver,
            self._system.rl,
            uses_rl_actor=uses_rl,
        )
        self._chunk_ref_horizon = resolve_ref_chunk_horizon(
            self._system.env_driver,
            self._system.rl,
            uses_rl_actor=uses_rl,
        )

    def current_chunk_exec_horizon(self) -> int:
        if hasattr(self, "_chunk_exec_horizon"):
            return int(self._chunk_exec_horizon)
        return resolve_chunk_exec_horizon(
            self._system.env_driver,
            self._system.rl,
            uses_rl_actor=self._uses_rl_actor_for_chunk(),
        )

    def current_ref_chunk_horizon(self) -> int:
        if hasattr(self, "_chunk_ref_horizon"):
            return int(self._chunk_ref_horizon)
        uses_rl = self._uses_rl_actor_for_chunk()
        return resolve_ref_chunk_horizon(
            self._system.env_driver,
            self._system.rl,
            uses_rl_actor=uses_rl,
        )

    def policy_uses_rl_actor(self) -> bool:
        return bool(getattr(self, "_chunk_uses_rl_actor", False))
