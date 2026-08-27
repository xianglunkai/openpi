from __future__ import annotations

import jax
import jax.numpy as jnp

from rlt_online_rl.config import RLTOnlineRLConfig
from rlt_online_rl.networks import ChunkActor
from rlt_online_rl.networks import PyTree
from rlt_online_rl.networks import TwinCritic
from rlt_online_rl.networks import build_td_target
from rlt_online_rl.networks import l2c2_mix_alpha
from rlt_online_rl.networks import mean_squared_l2
from rlt_online_rl.networks import mix_between


def compute_critic_loss(
    critic: TwinCritic,
    critic_params: PyTree,
    actor: ChunkActor,
    actor_params: PyTree,
    target_critic_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    action_chunk: jax.Array,
    rewards: jax.Array,
    done: jax.Array,
    next_z_rl: jax.Array,
    next_proprio: jax.Array,
    next_ref_chunk: jax.Array,
    actual_steps: jax.Array,
    rl_config: RLTOnlineRLConfig,
    *,
    rng: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Chunk n-step TD critic loss (twin Q), optional L2C2 smoothness on Q."""
    q1, q2 = critic.q_values(critic_params, z_rl, proprio, action_chunk)
    target_q = build_td_target(
        actor,
        actor_params,
        critic,
        target_critic_params,
        next_z_rl,
        next_proprio,
        next_ref_chunk,
        rewards,
        done,
        actual_steps,
        gamma=rl_config.gamma,
        action_clip_min=rl_config.action_clip_min,
        action_clip_max=rl_config.action_clip_max,
        target_q_clip=rl_config.target_q_clip,
    )
    td_loss = jnp.mean(jnp.square(q1 - target_q)) + jnp.mean(jnp.square(q2 - target_q))
    l2c2_penalty = jnp.asarray(0.0, dtype=jnp.float32)
    if rl_config.l2c2_critic_weight != 0.0 and rng is not None:
        # HoST: D(Q(s, a), Q(s̄, a)) with s̄ = s + α(s' - s), α ∈ [-1, 1].
        alpha = l2c2_mix_alpha(rng, done)
        mix_z = mix_between(z_rl, next_z_rl, alpha)
        mix_proprio = mix_between(proprio, next_proprio, alpha)
        mix_q1, mix_q2 = critic.q_values(critic_params, mix_z, mix_proprio, action_chunk)
        l2c2_penalty = mean_squared_l2(q1, mix_q1) + mean_squared_l2(q2, mix_q2)
    weighted_l2c2 = jnp.asarray(rl_config.l2c2_critic_weight, dtype=jnp.float32) * l2c2_penalty
    critic_loss = td_loss + weighted_l2c2
    metrics = {
        "td_loss": td_loss,
        "critic_loss": critic_loss,
        "l2c2_critic_penalty": l2c2_penalty,
        "weighted_l2c2_critic": weighted_l2c2,
        "q1_mean": jnp.mean(q1),
        "q2_mean": jnp.mean(q2),
        "target_q_mean": jnp.mean(target_q),
    }
    return critic_loss, metrics
