from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp

PyTree = Any


def _build_hidden_dims(hidden_dim: int, num_layers: int) -> tuple[int, ...]:
    return tuple(hidden_dim for _ in range(num_layers))


def _init_linear_params(rng: jax.Array, in_dim: int, out_dim: int) -> dict[str, jax.Array]:
    limit = jnp.sqrt(6.0 / float(in_dim + out_dim))
    w_key, _ = jax.random.split(rng)
    return {
        "w": jax.random.uniform(w_key, (in_dim, out_dim), minval=-limit, maxval=limit),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _layer_norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(variance + eps)


def _init_mlp_params(
    rng: jax.Array,
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
) -> dict[str, tuple[dict[str, jax.Array], ...]]:
    dims = (input_dim, *hidden_dims, output_dim)
    keys = jax.random.split(rng, len(dims) - 1)
    layers = tuple(_init_linear_params(k, dims[i], dims[i + 1]) for i, k in enumerate(keys))
    return {"layers": layers}


def _mlp_forward(params: PyTree, x: jax.Array) -> jax.Array:
    hidden = x
    for layer in params["layers"][:-1]:
        hidden = hidden @ layer["w"] + layer["b"]
        hidden = _layer_norm(hidden)
        hidden = jax.nn.gelu(hidden)
    last = params["layers"][-1]
    return hidden @ last["w"] + last["b"]


@dataclasses.dataclass(frozen=True)
class ChunkActor:
    z_dim: int
    proprio_dim: int
    chunk_len: int
    action_dim: int
    hidden_dim: int
    num_layers: int
    fixed_std: float

    def init_params(self, rng: jax.Array) -> PyTree:
        z_key, proprio_key, ref_key, trunk_key = jax.random.split(rng, 4)
        input_dim = 256 + 64 + 256
        output_dim = self.chunk_len * self.action_dim
        return {
            "z_proj": _init_linear_params(z_key, self.z_dim, 256),
            "proprio_proj": _init_linear_params(proprio_key, self.proprio_dim, 64),
            "ref_proj": _init_linear_params(ref_key, self.chunk_len * self.action_dim, 256),
            "trunk": _init_mlp_params(
                trunk_key, input_dim, _build_hidden_dims(self.hidden_dim, self.num_layers), output_dim
            ),
        }

    def _encode_inputs(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
    ) -> jax.Array:
        batch_size = z_rl.shape[0]
        ref_flat = ref_chunk.reshape(batch_size, self.chunk_len * self.action_dim)
        z_feat = _layer_norm(z_rl @ params["z_proj"]["w"] + params["z_proj"]["b"])
        proprio_feat = jnp.tanh(_layer_norm(proprio @ params["proprio_proj"]["w"] + params["proprio_proj"]["b"]))
        ref_feat = jnp.tanh(_layer_norm(ref_flat @ params["ref_proj"]["w"] + params["ref_proj"]["b"]))
        return jnp.concatenate([z_feat, proprio_feat, ref_feat], axis=-1)

    def actor_mean(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
    ) -> jax.Array:
        batch_size = z_rl.shape[0]
        features = self._encode_inputs(params, z_rl, proprio, ref_chunk)
        mu = _mlp_forward(params["trunk"], features)
        return mu.reshape(batch_size, self.chunk_len, self.action_dim)

    def actor_dist(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        mu = self.actor_mean(params, z_rl, proprio, ref_chunk)
        std = jnp.full_like(mu, self.fixed_std)
        return mu, std

    def sample_action(
        self,
        params: PyTree,
        rng: jax.Array,
        z_rl: jax.Array,
        proprio: jax.Array,
        ref_chunk: jax.Array,
        *,
        deterministic: bool = False,
    ) -> jax.Array:
        mu, std = self.actor_dist(params, z_rl, proprio, ref_chunk)
        if deterministic:
            return mu
        noise = jax.random.normal(rng, mu.shape, dtype=mu.dtype)
        return mu + std * noise


@dataclasses.dataclass(frozen=True)
class QNetwork:
    z_dim: int
    proprio_dim: int
    chunk_len: int
    action_dim: int
    hidden_dim: int
    num_layers: int

    def init_params(self, rng: jax.Array) -> PyTree:
        z_key, proprio_key, action_key, trunk_key = jax.random.split(rng, 4)
        input_dim = 256 + 64 + 256
        return {
            "z_proj": _init_linear_params(z_key, self.z_dim, 256),
            "proprio_proj": _init_linear_params(proprio_key, self.proprio_dim, 64),
            "action_proj": _init_linear_params(action_key, self.chunk_len * self.action_dim, 256),
            "trunk": _init_mlp_params(trunk_key, input_dim, _build_hidden_dims(self.hidden_dim, self.num_layers), 1),
        }

    def apply(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        action_chunk: jax.Array,
    ) -> jax.Array:
        batch_size = z_rl.shape[0]
        action_flat = action_chunk.reshape(batch_size, self.chunk_len * self.action_dim)
        z_feat = _layer_norm(z_rl @ params["z_proj"]["w"] + params["z_proj"]["b"])
        proprio_feat = jnp.tanh(_layer_norm(proprio @ params["proprio_proj"]["w"] + params["proprio_proj"]["b"]))
        action_feat = jnp.tanh(_layer_norm(action_flat @ params["action_proj"]["w"] + params["action_proj"]["b"]))
        features = jnp.concatenate([z_feat, proprio_feat, action_feat], axis=-1)
        q_value = _mlp_forward(params["trunk"], features)
        return q_value.squeeze(-1)


@dataclasses.dataclass(frozen=True)
class TwinCritic:
    z_dim: int
    proprio_dim: int
    chunk_len: int
    action_dim: int
    hidden_dim: int
    num_layers: int

    def init_params(self, rng: jax.Array) -> PyTree:
        q1_key, q2_key = jax.random.split(rng)
        q_network = QNetwork(
            z_dim=self.z_dim,
            proprio_dim=self.proprio_dim,
            chunk_len=self.chunk_len,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        return {
            "q1": q_network.init_params(q1_key),
            "q2": q_network.init_params(q2_key),
        }

    def q_values(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        action_chunk: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        q_network = QNetwork(
            z_dim=self.z_dim,
            proprio_dim=self.proprio_dim,
            chunk_len=self.chunk_len,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        q1 = q_network.apply(params["q1"], z_rl, proprio, action_chunk)
        q2 = q_network.apply(params["q2"], z_rl, proprio, action_chunk)
        return q1, q2

    def min_q(
        self,
        params: PyTree,
        z_rl: jax.Array,
        proprio: jax.Array,
        action_chunk: jax.Array,
    ) -> jax.Array:
        """TD3 / Evo-style conservative Q: min(Q1, Q2)."""
        q1, q2 = self.q_values(params, z_rl, proprio, action_chunk)
        return jnp.minimum(q1, q2)


def apply_reference_dropout(
    rng: jax.Array,
    ref_chunk: jax.Array,
    dropout_prob: float,
) -> jax.Array:
    if dropout_prob <= 0.0:
        return ref_chunk
    keep_mask = jax.random.bernoulli(rng, 1.0 - dropout_prob, (ref_chunk.shape[0], 1, 1))
    return ref_chunk * keep_mask.astype(ref_chunk.dtype)


def _discounted_chunk_rewards(rewards: jax.Array, gamma: float) -> jax.Array:
    discounts = jnp.power(gamma, jnp.arange(rewards.shape[-1], dtype=rewards.dtype))
    return jnp.sum(rewards * discounts[None, :], axis=-1)


def clamp_action_chunk(
    action_chunk: jax.Array,
    *,
    action_min: float,
    action_max: float,
) -> jax.Array:
    return jnp.clip(action_chunk, action_min, action_max)


def l2c2_mix_alpha(rng: jax.Array, done: jax.Array) -> jax.Array:
    """Sample α ∈ [-1, 1] for L2C2 state mixing; zero when the transition is terminal."""
    done_f = jnp.asarray(done, dtype=jnp.float32).reshape(-1)
    cont = 1.0 - done_f
    return cont * (jax.random.uniform(rng, done_f.shape) * 2.0 - 1.0)


def mix_between(x: jax.Array, next_x: jax.Array, alpha: jax.Array) -> jax.Array:
    """Interpolate ``x̄ = x + α (x' - x)`` with α broadcast over trailing dims."""
    alpha_b = alpha.reshape((alpha.shape[0],) + (1,) * (x.ndim - 1)).astype(x.dtype)
    return x + alpha_b * (next_x - x)


def mean_squared_l2(a: jax.Array, b: jax.Array) -> jax.Array:
    """Batch mean of ||a - b||_2^2 over all non-batch dimensions (HoST L2C2 distance)."""
    diff = (a - b).reshape((a.shape[0], -1))
    return jnp.mean(jnp.sum(jnp.square(diff), axis=-1))


def compute_delta_ref_match_penalty(
    pred_abs_chunk: jax.Array,
    ref_abs_chunk: jax.Array,
    *,
    pose_dim: int = 6,
) -> jax.Array:
    """Mean squared error between consecutive pose deltas of μ and ref.

    Both chunks must be in absolute pose space (after denormalize if needed).
    Only the first ``pose_dim`` dims (xyz + rot) are used; gripper is ignored.
    """
    pred_pose = pred_abs_chunk[..., :pose_dim]
    ref_pose = ref_abs_chunk[..., :pose_dim]
    pred_delta = pred_pose[:, 1:, :] - pred_pose[:, :-1, :]
    ref_delta = ref_pose[:, 1:, :] - ref_pose[:, :-1, :]
    return jnp.mean(jnp.square(pred_delta - ref_delta))


def compute_abs_chunk_acc_jerk_penalty(
    abs_chunk: jax.Array,
    *,
    pose_dim: int = 6,
    acc_weight: float = 1.0,
    jerk_weight: float = 1.0,
    dt: float = 1.0 / 30.0,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Physical acc/jerk penalties aligned with OpenPI ``OptimizeActionsQP``.

    Stencils match ``src/openpi/transforms.py`` (``D2=[1,-2,1]``, ``D3=[1,-3,3,-1]``).
    Hard acc limits there use ``|D2 x| <= a_max * dt^2``; we expose physical units:

      a_phys = D2(x) / dt^2
      j_phys = D3(x) / dt^3

    Metrics return ``mean(a_phys^2)`` and ``mean(j_phys^2)``.

    OpenPI penalizes ``w_acc * ||D2 x||^2 + w_jerk * ||D3 x||^2`` (no dt in the
    objective). With ``acc_weight=jerk_weight=100`` and ``dt=1/f_hz``, the combined
    training term is equivalent:

      acc_weight * dt^4 * mean(a_phys^2) + jerk_weight * dt^6 * mean(j_phys^2)

    Returns ``(combined, acc_penalty, jerk_penalty)``.
    """
    dt_f = max(float(dt), 1e-8)
    dt2 = dt_f * dt_f
    dt4 = dt2 * dt2
    dt6 = dt4 * dt2
    inv_dt2 = 1.0 / dt2
    inv_dt3 = inv_dt2 / dt_f
    pose = abs_chunk[..., :pose_dim]
    d1 = pose[:, 1:, :] - pose[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    d3 = d2[:, 1:, :] - d2[:, :-1, :]
    acc_penalty = jnp.mean(jnp.square(d2 * inv_dt2))
    jerk_penalty = jnp.mean(jnp.square(d3 * inv_dt3))
    combined = (
        jnp.asarray(acc_weight, dtype=jnp.float32) * dt4 * acc_penalty
        + jnp.asarray(jerk_weight, dtype=jnp.float32) * dt6 * jerk_penalty
    )
    return combined, acc_penalty, jerk_penalty


def compute_bc_penalty(
    action_chunk: jax.Array,
    bc_target: jax.Array,
    *,
    reduction: str,
) -> jax.Array:
    """BC penalty with Evo-RLT-compatible scaling."""
    error_squared = jnp.square(action_chunk - bc_target)
    per_sample = jnp.sum(error_squared, axis=(-2, -1))
    if reduction == "sum":
        return jnp.mean(per_sample)
    chunk_len = error_squared.shape[-2]
    action_dim = error_squared.shape[-1]
    return jnp.mean(per_sample) / jnp.maximum(chunk_len * action_dim, 1.0)


def build_td_target(
    actor: ChunkActor,
    actor_params: PyTree,
    target_critic: TwinCritic,
    target_critic_params: PyTree,
    next_z_rl: jax.Array,
    next_proprio: jax.Array,
    next_ref_chunk: jax.Array,
    rewards: jax.Array,
    done: jax.Array,
    actual_steps: jax.Array,
    *,
    gamma: float,
    action_clip_min: float,
    action_clip_max: float,
    target_q_clip: float,
) -> jax.Array:
    """TD3-style chunk TD target with deterministic next action and actual-step bootstrap."""
    next_action = actor.actor_mean(actor_params, next_z_rl, next_proprio, next_ref_chunk)
    next_action = clamp_action_chunk(
        next_action,
        action_min=action_clip_min,
        action_max=action_clip_max,
    )
    next_q1, next_q2 = target_critic.q_values(target_critic_params, next_z_rl, next_proprio, next_action)
    next_q = jnp.minimum(next_q1, next_q2)
    if target_q_clip > 0.0:
        next_q = jnp.clip(next_q, -target_q_clip, target_q_clip)
    bootstrap_exp = actual_steps.astype(rewards.dtype)
    bootstrap = (1.0 - done.astype(rewards.dtype)) * jnp.power(gamma, bootstrap_exp) * next_q
    return _discounted_chunk_rewards(rewards, gamma) + bootstrap


def compute_actor_loss(
    actor: ChunkActor,
    actor_params: PyTree,
    critic: TwinCritic,
    critic_params: PyTree,
    z_rl: jax.Array,
    proprio: jax.Array,
    ref_chunk: jax.Array,
    beta: float,
    reference_dropout_prob: float,
    bc_reduction: str,
    rng: jax.Array,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    dropout_rng, _ = jax.random.split(rng)
    dropped_ref = apply_reference_dropout(dropout_rng, ref_chunk, reference_dropout_prob)
    action_chunk = actor.actor_mean(actor_params, z_rl, proprio, dropped_ref)
    q = critic.min_q(critic_params, z_rl, proprio, action_chunk)
    bc_penalty = compute_bc_penalty(action_chunk, ref_chunk, reduction=bc_reduction)
    actor_loss = -jnp.mean(q) + beta * bc_penalty
    metrics = {
        "actor_loss": actor_loss,
        "actor_q": jnp.mean(q),
        "bc_penalty": bc_penalty,
    }
    return actor_loss, metrics
