from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.chunk_horizon import policy_uses_rl_actor
from rlt_online_rl.chunk_horizon import resolve_chunk_exec_horizon
from rlt_online_rl.chunk_horizon import resolve_ref_chunk_horizon
from rlt_online_rl.config import EnvDriverConfig
from rlt_online_rl.config import RLTOnlineRLConfig


def test_policy_uses_rl_actor_only_in_online_critical_actor_mode() -> None:
    assert not policy_uses_rl_actor(
        episode_phase="warmup",
        in_critical_phase=True,
        critical_policy_mode="actor",
    )
    assert policy_uses_rl_actor(
        episode_phase="online",
        in_critical_phase=True,
        critical_policy_mode="actor",
    )


def test_vla_25_rl_20_horizons() -> None:
    rl_config = RLTOnlineRLConfig(chunk_len=20)
    env_config = EnvDriverConfig(vla_chunk_exec_horizon=25)
    assert resolve_chunk_exec_horizon(env_config, rl_config, uses_rl_actor=False) == 25
    assert resolve_chunk_exec_horizon(env_config, rl_config, uses_rl_actor=True) == 20
    assert resolve_ref_chunk_horizon(env_config, rl_config, uses_rl_actor=False) == 25
    assert resolve_ref_chunk_horizon(env_config, rl_config, uses_rl_actor=True) == 20


def test_vla_25_rl_10_horizons() -> None:
    rl_config = RLTOnlineRLConfig(chunk_len=10)
    env_config = EnvDriverConfig(vla_chunk_exec_horizon=25)
    assert resolve_chunk_exec_horizon(env_config, rl_config, uses_rl_actor=False) == 25
    assert resolve_chunk_exec_horizon(env_config, rl_config, uses_rl_actor=True) == 10


def test_explicit_rl_exec_horizon_override() -> None:
    rl_config = RLTOnlineRLConfig(chunk_len=20)
    env_config = EnvDriverConfig(vla_chunk_exec_horizon=25, rl_chunk_exec_horizon=10)
    assert resolve_chunk_exec_horizon(env_config, rl_config, uses_rl_actor=True) == 10
