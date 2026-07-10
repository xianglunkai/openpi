from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp

from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import PyTree
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.networks import build_td_target


def _q_values_on_action_candidates(
    critic: TwinCritic,
    params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    action_candidates: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Evaluate Q1/Q2 on action candidates shaped (batch, num_candidates, chunk_len, action_dim)."""
    batch_size, num_candidates = action_candidates.shape[:2]
    flat_actions = action_candidates.reshape(batch_size * num_candidates, *action_candidates.shape[2:])
    z_flat = jnp.repeat(z_rl, num_candidates, axis=0)
    proprio_flat = jnp.repeat(proprio, num_candidates, axis=0)
    q1_flat, q2_flat = critic.q_values(params, z_flat, proprio_flat, flat_actions)
    return q1_flat.reshape(batch_size, num_candidates), q2_flat.reshape(batch_size, num_candidates)


def _sample_random_action_chunks(
    rng: jax.Array,
    shape_prefix: tuple[int, int],
    chunk_len: int,
    action_dim: int,
    method: Literal["normal", "uniform"],
    action_min: float,
    action_max: float,
) -> jax.Array:
    batch_size, n_actions = shape_prefix
    shape = (batch_size, n_actions, chunk_len, action_dim)
    if method == "uniform":
        return jax.random.uniform(rng, shape, minval=action_min, maxval=action_max)
    return jax.random.normal(rng, shape)


def _sample_policy_action_chunks(
    actor: ChunkActor,
    params: PyTree,
    rng: jax.Array,
    z_rl: jax.Array,
    proprio: jax.Array,
    ref_chunk: jax.Array,
    n_actions: int,
) -> jax.Array:
    keys = jax.random.split(rng, n_actions)

    def _one_sample(sample_rng: jax.Array) -> jax.Array:
        return actor.sample_action(
            params,
            sample_rng,
            z_rl,
            proprio,
            ref_chunk,
            deterministic=False,
        )

    sampled = jax.vmap(_one_sample)(keys)
    return jnp.swapaxes(sampled, 0, 1)


def _cql_conservative_q(
    q_ood_samples: jax.Array,
    q_behavior: jax.Array,
    temp: float,
) -> jax.Array:
    """Soft maximum over OOD Q values plus the dataset action Q value."""
    all_q = jnp.concatenate([q_ood_samples, q_behavior[:, None]], axis=-1)
    num_actions = all_q.shape[-1]
    all_q = all_q - jnp.log(num_actions) * temp
    return jax.scipy.special.logsumexp(all_q / temp, axis=-1) * temp


def _compute_td_critic_loss(
    critic: TwinCritic,
    critic_params: PyTree,
    actor: ChunkActor,
    target_actor_params: PyTree,
    target_critic_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    action_chunk: jax.Array,
    rewards: jax.Array,
    done: jax.Array,
    next_z_rl: jax.Array,
    next_proprio: jax.Array,
    next_ref_chunk: jax.Array,
    gamma: float,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, dict[str, jax.Array]]:
    q1, q2 = critic.q_values(critic_params, z_rl, proprio, action_chunk)
    target_q = build_td_target(
        actor,
        target_actor_params,
        critic,
        target_critic_params,
        next_z_rl,
        next_proprio,
        next_ref_chunk,
        rewards,
        done,
        gamma,
        rng,
    )
    td_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
    metrics = {
        "td_loss": td_loss,
        "q1_mean": jnp.mean(q1),
        "q2_mean": jnp.mean(q2),
        "target_q_mean": jnp.mean(target_q),
    }
    return td_loss, q1, q2, target_q, metrics


def _compute_cql_penalty(
    critic: TwinCritic,
    critic_params: PyTree,
    actor: ChunkActor,
    actor_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    ref_chunk: jax.Array,
    next_z_rl: jax.Array,
    next_proprio: jax.Array,
    next_ref_chunk: jax.Array,
    q1_behavior: jax.Array,
    q2_behavior: jax.Array,
    rl_config: RLTOnlineRLConfig,
    rng: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    n_actions = rl_config.cql_n_actions
    chunk_len = ref_chunk.shape[1]
    action_dim = ref_chunk.shape[2]
    batch_size = z_rl.shape[0]

    rng, random_rng, current_rng, next_rng = jax.random.split(rng, 4)
    random_actions = _sample_random_action_chunks(
        random_rng,
        (batch_size, n_actions),
        chunk_len,
        action_dim,
        rl_config.cql_action_sample_method,
        rl_config.cql_action_min,
        rl_config.cql_action_max,
    )
    current_actions = _sample_policy_action_chunks(
        actor, actor_params, current_rng, z_rl, proprio, ref_chunk, n_actions
    )
    next_actions = _sample_policy_action_chunks(
        actor, actor_params, next_rng, next_z_rl, next_proprio, next_ref_chunk, n_actions
    )
    ood_actions = jnp.concatenate([random_actions, current_actions, next_actions], axis=1)

    q1_ood, q2_ood = _q_values_on_action_candidates(critic, critic_params, z_rl, proprio, ood_actions)
    cql_ood_q1 = _cql_conservative_q(q1_ood, q1_behavior, rl_config.cql_temp)
    cql_ood_q2 = _cql_conservative_q(q2_ood, q2_behavior, rl_config.cql_temp)

    cql_diff_1 = jnp.clip(
        cql_ood_q1 - q1_behavior,
        rl_config.cql_clip_diff_min,
        rl_config.cql_clip_diff_max,
    )
    cql_diff_2 = jnp.clip(
        cql_ood_q2 - q2_behavior,
        rl_config.cql_clip_diff_min,
        rl_config.cql_clip_diff_max,
    )
    cql_loss = jnp.mean(cql_diff_1) + jnp.mean(cql_diff_2)

    n_total = n_actions * 3
    return cql_loss, {
        "cql_loss": cql_loss,
        "cql_diff": 0.5 * (jnp.mean(cql_diff_1) + jnp.mean(cql_diff_2)),
        "cql_ood_q1": jnp.mean(cql_ood_q1),
        "cql_ood_q2": jnp.mean(cql_ood_q2),
        "cql_random_q1": jnp.mean(q1_ood[:, :n_actions]),
        "cql_current_q1": jnp.mean(q1_ood[:, n_actions : 2 * n_actions]),
        "cql_next_q1": jnp.mean(q1_ood[:, 2 * n_actions : n_total]),
    }


def compute_critic_loss(
    critic: TwinCritic,
    critic_params: PyTree,
    actor: ChunkActor,
    actor_params: PyTree,
    target_actor_params: PyTree,
    target_critic_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    action_chunk: jax.Array,
    ref_chunk: jax.Array,
    rewards: jax.Array,
    done: jax.Array,
    next_z_rl: jax.Array,
    next_proprio: jax.Array,
    next_ref_chunk: jax.Array,
    rl_config: RLTOnlineRLConfig,
    rng: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Chunk n-step TD critic loss, optionally with CQL regularization (no MC lower bound)."""
    td_rng, cql_rng = jax.random.split(rng)
    td_loss, q1, q2, _, metrics = _compute_td_critic_loss(
        critic,
        critic_params,
        actor,
        target_actor_params,
        target_critic_params,
        z_rl,
        proprio,
        action_chunk,
        rewards,
        done,
        next_z_rl,
        next_proprio,
        next_ref_chunk,
        rl_config.gamma,
        td_rng,
    )

    if rl_config.critic_loss_mode == "td":
        metrics["critic_loss"] = td_loss
        return td_loss, metrics

    cql_loss, cql_metrics = _compute_cql_penalty(
        critic,
        critic_params,
        actor,
        actor_params,
        z_rl,
        proprio,
        ref_chunk,
        next_z_rl,
        next_proprio,
        next_ref_chunk,
        q1,
        q2,
        rl_config,
        cql_rng,
    )
    critic_loss = td_loss + rl_config.cql_alpha * cql_loss
    metrics = {
        **metrics,
        **cql_metrics,
        "critic_loss": critic_loss,
        "cql_alpha": jnp.asarray(rl_config.cql_alpha, dtype=jnp.float32),
    }
    return critic_loss, metrics
