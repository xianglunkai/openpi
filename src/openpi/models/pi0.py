import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override
from typing import Any

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.policies import rtc_processor

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs, rtc_config: rtc_processor.RTCConfig = None):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True
        
        self.init_rtc_processor(rtc_config)
        
    def init_rtc_processor(self, rtc_config: rtc_processor.RTCConfig = None):
        rtc_config = rtc_processor.RTCConfig(
            enabled=True,
            prefix_attention_schedule="EXP",
            max_guidance_weight=10.0,
            execution_horizon=25,
        )

        if rtc_config is None:
            self.rtc_processor = None
        else:
            self.rtc_processor = rtc_processor.RTCProcessor(rtc_config)

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)
    
    @override
    def guided_inference(
        self,
        rng: at.KeyArrayLike,
        prev_action: _model.Actions,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,       
        s: int = 25,
        d: int = 10,
        beta: float = 10.0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        
        if prev_action is not None:
            # get prev_action from s-th step to the end, and then pad s steps with zeros
            prev_action_slice = prev_action[:, s:, :]  # get prev_action from s-th step to the end
            # jax.debug.print("prev_action_slice shape: {prev_action_slice_shape}", prev_action_slice_shape=prev_action_slice.shape)
            # create s steps with zeros
            zero_actions = jnp.zeros((batch_size, s, self.action_dim))
            # concatenate prev_action_slice and zero_actions
            prev_action_slice = jnp.concatenate([prev_action_slice, zero_actions], axis=1)

        def make_W(d: int, s: int) -> jnp.ndarray:
            """
            generate the weight vector W ∈ ℝ^H
            parameters
            ----
            H : int  # sequence length
            d : int  # "deterministic region" threshold
            s : int  # "truncated" window length
            return
            ----
            W : jnp.ndarray, shape (H,)
            """
            H = self.action_horizon
            i = jnp.arange(H)           # 0,1,2,...,H-1

            # three-segment condition
            cond_1 = i < d
            cond_2 = (i >= d) & (i < H - s)
            cond_3 = i >= H - s         # actually can be else

            # segment (1): all 1
            w1 = jnp.ones_like(i, dtype=float)

            # segment (2): exponential decay
            c_i = (H - s - i) / (H - s - d + 1)
            w2  = jnp.exp(c_i) - 1
            w2  = c_i * w2 / (jnp.e - 1)      # (e^{c_i} - 1) / (e - 1)
         

            # segment (3): all 0
            w3 = jnp.zeros_like(i, dtype=float)

            # concatenate three segments
            W = jnp.where(cond_1, w1,
                jnp.where(cond_2, w2, w3)
            )

            D = jnp.diag(W)

            D_batch = jnp.stack([D] * 1, axis=0)
            return D_batch

        # create W
        diag_W = make_W(d, s)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def func_a_1_prime(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=kv_cache, adarms_cond=[None, adarms_cond]
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t - time * v_t, v_t

        def step_rtc(carry):
            x_t, time = carry
            (a_1_prime, v_t), f_vjp = jax.vjp(func_a_1_prime, x_t, time)

            e = prev_action_slice - a_1_prime
            e = jnp.matmul(diag_W, e)
            #Compute vector-Jacobian product
            grad_a_1_prime_x_t = f_vjp((e, jnp.zeros_like(v_t)))
            
        
            inv_r2 = (time**2 + (1 - time) ** 2) / (time ** 2)
            c = jnp.nan_to_num(time / (1 - time), posinf=beta)
            guidance_weight = jnp.nan_to_num(c * inv_r2, posinf = beta)
            guidance_weight = jnp.minimum(guidance_weight, beta)
            a_2_prime = x_t + dt * (v_t - guidance_weight * grad_a_1_prime_x_t[0])
           
            # r_t = time * time / (time * time + (1 - time) * (1 - time))
            # a_2_prime = x_t + dt * (v_t - jax.lax.min(beta, time / ((1 - time) * r_t * r_t + 1e-6)) * grad_a_1_prime_x_t[0])
            
            return a_2_prime, time + dt
        
        
        def step_normal(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt
        
        
        def step(carry):
            if prev_action is not None:
                return step_rtc(carry)
            else:
                return step_normal(carry)
        

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0



    @override
    def sample_actions_with_rtc(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        **kwargs: Any,
    ) -> _model.Actions | tuple[_model.Actions, dict]:

        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        inference_delay = kwargs.get("inference_delay")
        prev_chunk_left_over = kwargs.get("prev_chunk_left_over")

        rtc_is_enabled = self.rtc_processor is not None and self.rtc_processor.rtc_enabled()
        logger.info(f"RTC check before scan: rtc_processor={self.rtc_processor is not None}, enabled={rtc_is_enabled}")

        # For JAX compilation, we need to determine RTC path outside the compiled function
        use_rtc = rtc_is_enabled and prev_chunk_left_over is not None

        #  Make padding for prev_chunk_left_over to match the action_horizon and action_dim
        # prev_chunk_left_over shape: (batch, time, action_dim)
        # Pad the time dimension (axis 1) to match action_horizon
        # Pad the action dimension (axis 2) to match self.action_dim
        time_pad = self.action_horizon - prev_chunk_left_over.shape[1]
        action_dim_pad = self.action_dim - prev_chunk_left_over.shape[2]
        prev_chunk_left_over = jnp.pad(prev_chunk_left_over, ((0, 0), (0, time_pad), (0, action_dim_pad)))

        # Debug prints before entering JAX-compiled loop
        logger.info(f"RTC Config enabled: {self.rtc_processor is not None}")
        if self.rtc_processor is not None:
            logger.info(f"RTC processor details: enabled={self.rtc_processor.rtc_enabled()}, config={self.rtc_processor.rtc_config}")
            logger.info(f"RTC execution_horizon from config: {self.rtc_processor.rtc_config.execution_horizon}")
            logger.info(f"RTC prefix_attention_schedule: {self.rtc_processor.rtc_config.prefix_attention_schedule}")
            logger.info(f"RTC max_guidance_weight: {self.rtc_processor.rtc_config.max_guidance_weight}")
        logger.info(f"inference_delay: {inference_delay}")
        logger.info(f"prev_chunk_left_over shape: {prev_chunk_left_over.shape if prev_chunk_left_over is not None else None}")
        logger.info(f"action_horizon: {self.action_horizon}")
        logger.info(f"action_dim: {self.action_dim}")

        logger.info(f"use_rtc: {use_rtc}")  
        logger.info(f"batch_size: {batch_size}")
        def original_step_scan(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        def step_scan(carry, step_idx):
            x_t, time = carry
            if use_rtc:
                # Use jax.debug.print for runtime logging (not just tracing)
                jax.debug.print("=== Step {} - USING RTC PATH === at time={}", step_idx, time)

                def pinv_corrected_velocity(x_t, y, time):
                    jax.debug.print("  Step {}: time={}, x_t_norm={}", step_idx, time, jnp.linalg.norm(x_t))

                    def denoiser(x_t):
                        v_t = original_step_scan((x_t, time))
                        # Remove batch dimension from outputs
                        return (x_t - v_t * (1 - time)), v_t

                    x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)

                    weights = self.rtc_processor.get_prefix_weights(
                        inference_delay, self.rtc_processor.rtc_config.execution_horizon, self.action_horizon, self.rtc_processor.rtc_config.prefix_attention_schedule
                    )

                    weights = einops.repeat(weights, "c -> b c a", b=batch_size, a=self.action_dim)

                    error = (y - x_1) * weights

                    pinv_correction = vjp_fun(error)[0]
                    # constants from paper
                    # Handle numerical stability: at time=1.0, we get (1-time)=0 which causes 0*inf=NaN
                    # The correct limit as t→1 is guidance_weight→0, so we replace NaN with 0
                    inv_r2 = (time**2 + (1 - time) ** 2) / ((1 - time) ** 2)
                    c = jnp.nan_to_num((1 - time) / time, posinf=self.rtc_processor.rtc_config.max_guidance_weight)
                    guidance_weight = jnp.minimum(c * inv_r2, self.rtc_processor.rtc_config.max_guidance_weight)
                    # Replace NaN with 0 (occurs at t=1 where guidance should be 0 anyway)
                    guidance_weight = jnp.nan_to_num(guidance_weight, nan=0.0)

                    jax.debug.print("Error {}: pinv_correction={}", error, pinv_correction)
                    v_t_corrected = v_t - guidance_weight * pinv_correction

                    jax.debug.print("  Guidance: weight={}, error_norm={}", guidance_weight, jnp.linalg.norm(error))

                    # Return both velocity and tracking data
                    return v_t_corrected, {
                        "x_1": x_1,
                        "v_t": v_t_corrected,
                        "error": error,
                        "weights": weights,
                        "guidance_weight": guidance_weight,
                        "pinv_correction": pinv_correction,
                    }

                v_t, step_tracking = pinv_corrected_velocity(x_t, prev_chunk_left_over, time)

            else:
                jax.debug.print("=== Step {} - USING NON-RTC PATH === at time={}", step_idx, time)

                v_t = original_step_scan((x_t, time))

                step_tracking = {
                    "x_1": jnp.zeros_like(x_t),
                    "v_t": v_t,
                    "error": jnp.zeros_like(x_t),
                    "weights": jnp.zeros((batch_size, self.action_horizon, self.action_dim)),
                    "guidance_weight": jnp.zeros(()),
                    "pinv_correction": jnp.zeros_like(x_t),
                }

            x_t = x_t - dt * v_t

            # Add x_t, time, and step_idx to tracking
            step_tracking["x_t"] = x_t
            step_tracking["time"] = time
            step_tracking["step_idx"] = step_idx

            # Return updated carry and scan output
            return (x_t, time + dt), step_tracking

        # Create step indices array for scan
        step_indices = jnp.arange(num_steps)
        final_carry, tracking_history = jax.lax.scan(step_scan, (noise, 1.0), step_indices)

        # Extract final x_t from carry
        x_0 = final_carry[0]

        # tracking_history now contains ALL steps (shape: (num_steps, ...))
        # Each field in tracking_history dict has shape (num_steps, batch_size, ...)
        logger.info(f"Collected tracking history for {num_steps} steps")
        logger.info(f"tracking_history keys: {tracking_history.keys()}")
        logger.info(f"Tracking history shapes:")
        for key, value in tracking_history.items():
            logger.info(f"  {key}: {value.shape}")

        # Log summary of collected data
        logger.info(f"Total denoise steps collected: {tracking_history['step_idx'].shape[0]}")
        logger.info(f"Step indices range: {tracking_history['step_idx'][0]} to {tracking_history['step_idx'][-1]}")
        logger.info(f"Time values: start={tracking_history['time'][0]:.4f}, end={tracking_history['time'][-1]:.4f}")

        # Store tracking history in the tracker if available
        if self.rtc_processor is not None and self.rtc_processor.tracker is not None:
            self.rtc_processor.tracker.set_tracking_history(tracking_history)

        # Always return both to avoid JAX tracer issues
        # The caller can decide whether to use the tracking data
        # tracking_history contains data from ALL denoising steps
        return x_0, tracking_history