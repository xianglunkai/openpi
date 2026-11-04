from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.action_representation import ActionRepresentationAdapter
from rlt_online_rl.action_representation import jax_denormalize_to_abs_chunk
from rlt_online_rl.config import LearnerServiceConfig
from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.replay import TransitionSource
from rlt_online_rl.networks import clip_actor_output_straight_through
from rlt_online_rl.trainer import LearnerService
from rlt_online_rl.trainer import _behavior_cloning_terms
from rlt_online_rl.trainer import init_train_state
from rlt_online_rl.trainer import soft_update_targets
from rlt_online_rl.trainer import train_step


class FakeReplay:
    def __init__(self, batch: dict[str, np.ndarray]):
        self._batch = batch
        self._adds_total = batch["z_rl"].shape[0]

    def sample_batch(self, _batch_size: int, *, success_only: bool = False) -> dict[str, np.ndarray]:
        return self._batch

    def stats(self) -> dict[str, int]:
        return {"size": self._batch["z_rl"].shape[0], "adds_total": self._adds_total}


def _config() -> RLTOnlineRLConfig:
    return RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_num_layers=2,
        critic_num_layers=2,
        actor_update_period=2,
        warmup_min_size=1,
        grad_updates_per_cycle=2,
        bc_reduction="sum",
    )


def _batch(cfg: RLTOnlineRLConfig, batch_size: int = 8) -> dict[str, np.ndarray]:
    return {
        "z_rl": np.ones((batch_size, cfg.z_dim), dtype=np.float32),
        "proprio": np.ones((batch_size, cfg.proprio_dim), dtype=np.float32),
        "ref_chunk": np.ones((batch_size, cfg.chunk_len, cfg.action_dim), dtype=np.float32),
        "action_chunk": np.ones((batch_size, cfg.chunk_len, cfg.action_dim), dtype=np.float32),
        "rewards": np.ones((batch_size, cfg.chunk_len), dtype=np.float32),
        "done": np.zeros((batch_size,), dtype=np.float32),
        "next_z_rl": np.ones((batch_size, cfg.z_dim), dtype=np.float32),
        "next_proprio": np.ones((batch_size, cfg.proprio_dim), dtype=np.float32),
        "next_ref_chunk": np.ones((batch_size, cfg.chunk_len, cfg.action_dim), dtype=np.float32),
        "source": np.zeros((batch_size,), dtype=np.uint8),
        "source_chunk": np.zeros((batch_size, cfg.chunk_len), dtype=np.uint8),
        "success": np.zeros((batch_size,), dtype=np.int8),
        "intervention_flag": np.zeros((batch_size,), dtype=np.bool_),
        "episode_id": np.zeros((batch_size,), dtype=np.int32),
        "step_id": np.arange(batch_size, dtype=np.int32),
        "actual_steps": np.full((batch_size,), cfg.chunk_len, dtype=np.int32),
    }


def test_train_step_runs_and_actor_update_period_works() -> None:
    cfg = _config()
    state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(0))
    batch = {k: jax.numpy.asarray(v) for k, v in _batch(cfg).items()}
    state, metrics1 = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
    assert int(metrics1["did_actor_update"]) == 0
    state, metrics2 = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
    assert int(metrics2["did_actor_update"]) == 1


def test_train_step_bc_target_switches_to_human_actions() -> None:
    cfg = dataclasses.replace(_config(), bc_imitate_human=True)
    state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(0))
    batch_np = _batch(cfg)
    batch_np["ref_chunk"] = np.zeros_like(batch_np["ref_chunk"])
    batch_np["action_chunk"] = np.ones_like(batch_np["action_chunk"])
    batch_np["source_chunk"] = np.full_like(batch_np["source_chunk"], int(TransitionSource.HUMAN), dtype=np.uint8)
    batch = {k: jax.numpy.asarray(v) for k, v in batch_np.items()}

    state, _ = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
    _state, metrics = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)

    assert int(metrics["did_actor_update"]) == 1
    assert np.isclose(float(metrics["human_mask_ratio"]), 1.0)
    assert np.isclose(float(metrics["policy_mask_ratio"]), 0.0)
    assert np.isclose(float(metrics["bc_penalty"]), float(metrics["bc_human_penalty"]))
    assert np.isclose(float(metrics["bc_ref_penalty"]), 0.0)


def test_bc_stops_pushing_when_label_exceeds_clip() -> None:
    cfg = _config()
    label = jnp.ones((1, cfg.chunk_len, cfg.action_dim), dtype=jnp.float32) * 1.4
    batch = {
        "source_chunk": jnp.zeros((1, cfg.chunk_len), dtype=jnp.int32),
        "ref_chunk": label,
        "action_chunk": label,
    }
    raw = jnp.ones_like(label) * 50.0

    def loss(raw_action: jax.Array) -> jax.Array:
        bc_action = clip_actor_output_straight_through(
            raw_action,
            action_min=cfg.action_clip_min,
            action_max=cfg.action_clip_max,
        )
        return _behavior_cloning_terms(bc_action, batch, cfg)["bc_penalty"]

    assert float(loss(raw)) == 0.0
    assert np.allclose(jax.grad(loss)(raw), 0.0)


def test_train_step_bc_stays_on_ref_when_human_imitate_disabled() -> None:
    cfg = dataclasses.replace(_config(), bc_imitate_human=False)
    state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(0))
    batch_np = _batch(cfg)
    batch_np["ref_chunk"] = np.zeros_like(batch_np["ref_chunk"])
    batch_np["action_chunk"] = np.ones_like(batch_np["action_chunk"])
    batch_np["source_chunk"] = np.full_like(batch_np["source_chunk"], int(TransitionSource.HUMAN), dtype=np.uint8)
    batch = {k: jax.numpy.asarray(v) for k, v in batch_np.items()}

    state, _ = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
    _state, metrics = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)

    assert int(metrics["did_actor_update"]) == 1
    assert np.isclose(float(metrics["human_mask_ratio"]), 1.0)
    # Policy-only metric is undefined on an all-human batch (mask sum=0 → 0).
    # Disabled path BCs to ref (zeros), not executed (ones), so the two penalties differ.
    assert not np.isclose(float(metrics["bc_penalty"]), float(metrics["bc_human_penalty"]))


def _params_allclose(left, right) -> bool:
    return all(
        np.allclose(np.asarray(old), np.asarray(new))
        for old, new in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_success_batch_filters_bc_but_q_uses_full_replay() -> None:
    cfg = dataclasses.replace(
        _config(),
        actor_update_period=1,
        reference_dropout_prob=0.0,
        delta_acc_weight=0.0,
        delta_jerk_weight=0.0,
        l2c2_policy_weight=0.0,
    )
    full_np = _batch(cfg)
    success_np = _batch(cfg)
    full_np["z_rl"] = np.ones_like(full_np["z_rl"])
    success_np["z_rl"] = np.zeros_like(success_np["z_rl"])
    full = {k: jax.numpy.asarray(v) for k, v in full_np.items()}
    success = {k: jax.numpy.asarray(v) for k, v in success_np.items()}

    def _step(full_batch, success_batch, *, bc_weight: float, q_weight: float):
        state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(0))
        return train_step(
            state,
            full_batch,
            actor=actor,
            critic=critic,
            rl_config=cfg,
            actor_batch=success_batch,
            bc_weight=bc_weight,
            q_weight=q_weight,
        )

    full_shift = dict(full)
    full_shift["z_rl"] = jax.numpy.full_like(full["z_rl"], 0.5)
    success_shift = dict(success)
    success_shift["z_rl"] = jax.numpy.full_like(success["z_rl"], -0.5)

    state_bc_ref, bc_base = _step(full, success, bc_weight=1.0, q_weight=0.0)
    state_bc_full, bc_full = _step(full_shift, success, bc_weight=1.0, q_weight=0.0)
    state_bc_success, bc_success = _step(full, success_shift, bc_weight=1.0, q_weight=0.0)
    assert _params_allclose(state_bc_ref.actor_params, state_bc_full.actor_params)
    assert np.isclose(float(bc_base["bc_penalty"]), float(bc_full["bc_penalty"]))
    assert not _params_allclose(state_bc_ref.actor_params, state_bc_success.actor_params)
    assert not np.isclose(float(bc_base["bc_penalty"]), float(bc_success["bc_penalty"]))

    state_q_ref, q_base = _step(full, success, bc_weight=0.0, q_weight=1.0)
    state_q_full, q_full = _step(full_shift, success, bc_weight=0.0, q_weight=1.0)
    state_q_success, q_success = _step(full, success_shift, bc_weight=0.0, q_weight=1.0)
    assert np.isclose(float(q_base["actor_q"]), float(q_success["actor_q"]))
    assert _params_allclose(state_q_ref.actor_params, state_q_success.actor_params)
    assert not np.isclose(float(q_base["actor_q"]), float(q_full["actor_q"]))
    assert not _params_allclose(state_q_ref.actor_params, state_q_full.actor_params)


def test_soft_update_changes_target_params() -> None:
    cfg = _config()
    state, _, _ = init_train_state(cfg, rng=jax.random.PRNGKey(1))
    updated = soft_update_targets(state.target_actor_params, jax.tree.map(lambda x: x + 1.0, state.actor_params), 0.5)
    leaves_old = jax.tree.leaves(state.target_actor_params)
    leaves_new = jax.tree.leaves(updated)
    assert any(not np.allclose(np.asarray(o), np.asarray(n)) for o, n in zip(leaves_old, leaves_new, strict=True))


def test_td_bootstrap_actor_modes_run() -> None:
    batch_np = None
    for mode in ("online", "target"):
        cfg = dataclasses.replace(_config(), td_bootstrap_actor=mode)
        state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(3))
        if batch_np is None:
            batch_np = _batch(cfg)
        batch = {k: jax.numpy.asarray(v) for k, v in batch_np.items()}
        state, metrics = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
        assert np.isfinite(float(metrics["critic_loss"]))
        state, metrics = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
        assert int(metrics["did_actor_update"]) == 1
        assert np.isfinite(float(metrics["actor_loss"]))


def test_target_critic_updates_without_actor_update() -> None:
    cfg = _config()
    state, actor, critic = init_train_state(cfg, rng=jax.random.PRNGKey(2))
    batch = {k: jax.numpy.asarray(v) for k, v in _batch(cfg).items()}
    target_critic_before = jax.tree.map(np.asarray, state.target_critic_params)
    target_actor_before = jax.tree.map(np.asarray, state.target_actor_params)
    state, metrics = train_step(state, batch, actor=actor, critic=critic, rl_config=cfg)
    assert int(metrics["did_actor_update"]) == 0
    target_critic_after = jax.tree.map(np.asarray, state.target_critic_params)
    target_actor_after = jax.tree.map(np.asarray, state.target_actor_params)
    assert any(
        not np.allclose(np.asarray(old), np.asarray(new))
        for old, new in zip(
            jax.tree.leaves(target_critic_before),
            jax.tree.leaves(target_critic_after),
            strict=True,
        )
    )
    assert all(
        np.allclose(np.asarray(old), np.asarray(new))
        for old, new in zip(
            jax.tree.leaves(target_actor_before),
            jax.tree.leaves(target_actor_after),
            strict=True,
        )
    )


def test_learner_service_exports_actor_snapshot(tmp_path) -> None:
    cfg = _config()
    service_cfg = LearnerServiceConfig(
        sample_batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        actor_snapshot_path=str(tmp_path / "actor.pkl"),
        push_actor_interval_steps=1,
        checkpoint_interval_steps=100,
    )
    learner = LearnerService(cfg, service_cfg, FakeReplay(_batch(cfg)))
    metrics = learner.train_once()
    assert metrics is not None
    metrics = learner.train_once()
    assert metrics is not None
    snapshot = learner.get_actor_snapshot()
    assert snapshot["version"] == 1
    assert (tmp_path / "actor.pkl").exists()
    history_path = tmp_path / "history" / "actor_v000001.pkl"
    assert history_path.exists()
    with history_path.open("rb") as f:
        history_payload = pickle.load(f)
    assert history_payload["version"] == 1
    assert history_payload["global_step"] == 2


def test_learner_service_stops_when_update_budget_is_consumed(tmp_path) -> None:
    cfg = _config()
    service_cfg = LearnerServiceConfig(
        sample_batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        actor_snapshot_path=str(tmp_path / "actor.pkl"),
        push_actor_interval_steps=100,
        checkpoint_interval_steps=100,
    )
    learner = LearnerService(cfg, service_cfg, FakeReplay(_batch(cfg, batch_size=3)))
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is None


def test_learner_service_uses_fixed_warmup_budget_then_online_ratio(tmp_path) -> None:
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_num_layers=2,
        critic_num_layers=2,
        actor_update_period=2,
        warmup_min_size=1,
        warmup_post_collect_updates=3,
        grad_updates_per_cycle=5,
    )
    replay = FakeReplay(_batch(cfg, batch_size=3))
    service_cfg = LearnerServiceConfig(
        sample_batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        actor_snapshot_path=str(tmp_path / "actor.pkl"),
        push_actor_interval_steps=100,
        checkpoint_interval_steps=100,
    )
    learner = LearnerService(cfg, service_cfg, replay)

    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is None

    replay._adds_total = 4
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is None


def test_learner_service_freezes_after_warmup_when_configured(tmp_path) -> None:
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        z_dim=5,
        proprio_dim=2,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        actor_num_layers=2,
        critic_num_layers=2,
        actor_update_period=2,
        warmup_min_size=1,
        warmup_post_collect_updates=3,
        freeze_after_warmup=True,
        grad_updates_per_cycle=5,
    )
    replay = FakeReplay(_batch(cfg, batch_size=3))
    service_cfg = LearnerServiceConfig(
        sample_batch_size=8,
        checkpoint_dir=str(tmp_path / "ckpts"),
        actor_snapshot_path=str(tmp_path / "actor.pkl"),
        push_actor_interval_steps=100,
        checkpoint_interval_steps=100,
    )
    learner = LearnerService(cfg, service_cfg, replay)

    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is not None
    assert learner.train_once() is None

    replay._adds_total = 4
    assert learner.train_once() is None


def test_delta_chunk_normalization_broadcasts_batched_state0(tmp_path) -> None:
    stats_path = tmp_path / "norm_stats_delta.json"
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "actions": {
                        "q01": [-1.0, -1.0, -1.0],
                        "q99": [1.0, 1.0, 1.0],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    cfg = RLTOnlineRLConfig(
        action_dim=3,
        chunk_len=4,
        proprio_dim=3,
        action_representation="delta_chunk",
        action_norm_stats_path=str(stats_path),
    )
    adapter = ActionRepresentationAdapter.from_config(cfg)
    assert adapter is not None
    chunk_abs = np.ones((8, cfg.chunk_len, cfg.action_dim), dtype=np.float32)
    state0 = np.full((8, cfg.proprio_dim), 0.5, dtype=np.float32)
    normalized = adapter.normalize_chunk(chunk_abs, state0)
    assert normalized.shape == chunk_abs.shape
    restored = np.asarray(
        jax.device_get(
            jax_denormalize_to_abs_chunk(
                jax.numpy.asarray(normalized),
                jax.numpy.asarray(state0),
                jax.numpy.asarray(adapter.stats.q01),
                jax.numpy.asarray(adapter.stats.q99),
                action_representation=cfg.action_representation,
            )
        ),
        dtype=np.float32,
    )
    assert np.allclose(restored, chunk_abs, atol=1e-5)



