"""Serve RLT policy via WebSocket.

Supports both single and batch inference.
- Single: client sends one observation dict → returns one result dict
- Batch:  client sends {"batch": [obs1, obs2, ...]} → returns {"batch_results": [r1, r2, ...]}

Usage:
    python scripts/serve_rlt_policy.py \
        --config rlt_pi05_agilexbag_image \
        --checkpoint-dir checkpoints/rlt_pi05_agilexbag_image/exp_A_frozen_finetuned/1000 \
        --port 8000
"""

from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import socket
import time
from typing import Any, ClassVar

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override
import tyro

import openpi.models.model as _model
from openpi.models.rl_token import RLTokenConfig
from openpi.models.rl_token import RLTokenModel
from openpi.serving import websocket_policy_server
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.transforms as _transforms


def _create_rlt_config(config: _config.TrainConfig) -> RLTokenConfig:
    kwargs = {}
    if config.rlt_num_tokens is not None:
        kwargs["num_rl_tokens"] = config.rlt_num_tokens
    if config.rlt_num_layers is not None:
        kwargs["num_layers"] = config.rlt_num_layers
    if config.rlt_embed_dim is not None:
        kwargs["embed_dim"] = config.rlt_embed_dim
    if config.rlt_input_dim is not None:
        kwargs["input_dim"] = config.rlt_input_dim
    return RLTokenConfig(**kwargs)


def _infer_prefix_seq_len(model_config: _model.BaseModelConfig) -> int:
    def _get_len(rng):
        model = model_config.create(rng)
        obs = model_config.fake_obs(batch_size=1)
        embs, _ = model.extract_prefix_embeddings(rng, obs, image_only=True)
        return embs

    embs_shape = jax.eval_shape(_get_len, jax.random.key(0))
    return embs_shape.shape[1]


class RLTInferenceModel(nnx.Module):
    """Combined VLA + RLT model for inference. Outputs both actions and RL tokens."""

    def __init__(
        self,
        vla_model: _model.BaseModel,
        rlt_config: RLTokenConfig,
        rngs: nnx.Rngs,
        prefix_seq_len: int = 768,
        *,
        shared_prefix_inference: bool = False,
    ):
        self.vla = vla_model
        self.shared_prefix_inference = shared_prefix_inference
        linen_rlt = RLTokenModel(config=rlt_config)
        self.rlt_module = nnx_bridge.ToNNX(linen_rlt)
        dummy_prefix = jnp.zeros((1, prefix_seq_len, rlt_config.input_dim))
        dummy_mask = jnp.ones((1, prefix_seq_len), dtype=jnp.bool_)
        self.rlt_module.lazy_init(dummy_prefix, dummy_mask, rngs=rngs)
        self.deterministic = True

    def infer(self, rng: at.KeyArrayLike, observation: _model.Observation) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Run inference. Supports any batch size.

        Returns:
            actions: [batch, action_horizon, action_dim] VLA action chunks
            rl_token: [batch, num_rl_tokens, embed_dim] compressed RL token
        """
        if self.shared_prefix_inference:
            return self._infer_shared_prefix(rng, observation)
        return self._infer_legacy(rng, observation)

    def infer_guided(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        prev_action: jnp.ndarray | None = None,
        s: int = 25,
        d: int = 7,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """RTC path: guided VLA chunk + RL token from the same observation prefix.

        ``prev_action=None`` matches openpi ``Policy``: still calls ``guided_inference``,
        which falls back to the unguided ``step_normal`` branch internally.
        """
        if self.shared_prefix_inference:
            prefix_cache = self.vla.prepare_prefix_for_inference(observation)
            prefix_f32 = prefix_cache.image_prefix_out.astype(jnp.float32)
            image_prefix_mask = prefix_cache.prefix_mask[:, : prefix_f32.shape[1]]
            rl_token = self.rlt_module(prefix_f32, image_prefix_mask, method="encode", train=False)
        else:
            prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(
                rng, observation, train=False, image_only=True
            )
            prefix_f32 = prefix_embs.astype(jnp.float32)
            rl_token = self.rlt_module(prefix_f32, prefix_mask, method="encode", train=False)
        actions = self.vla.guided_inference(
            rng,
            observation,
            prev_action=prev_action,
            s=int(s),
            d=int(d),
        )
        return actions, rl_token

    def _infer_legacy(self, rng: at.KeyArrayLike, observation: _model.Observation) -> tuple[jnp.ndarray, jnp.ndarray]:
        prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(
            rng, observation, train=False, image_only=True
        )
        prefix_f32 = prefix_embs.astype(jnp.float32)
        rl_token = self.rlt_module(prefix_f32, prefix_mask, method="encode", train=False)
        actions = self.vla.sample_actions(rng, observation)
        return actions, rl_token

    def _infer_shared_prefix(
        self, rng: at.KeyArrayLike, observation: _model.Observation
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        prefix_cache = self.vla.prepare_prefix_for_inference(observation)
        prefix_f32 = prefix_cache.image_prefix_out.astype(jnp.float32)
        # Cache stores full prefix_mask (images + language); RLT uses image tokens only.
        image_prefix_mask = prefix_cache.prefix_mask[:, : prefix_f32.shape[1]]
        rl_token = self.rlt_module(prefix_f32, image_prefix_mask, method="encode", train=False)
        actions = self.vla.sample_actions_from_prefix_cache(rng, prefix_cache)
        return actions, rl_token

# Constants for output format
PROPRIO_DIM = 7
CHUNK_LEN = 50
ACTION_DIM = 7


def _pad_prev_action_time(prev_action: jnp.ndarray, action_horizon: int) -> jnp.ndarray:
    """Pad/truncate leftover actions to ``(B, action_horizon, D)`` for stable JIT shapes."""
    prev = jnp.asarray(prev_action, dtype=jnp.float32)
    if prev.ndim != 3:
        raise ValueError(f"prev_action must be rank-3 (B, T, D), got {prev.shape}")
    t = int(prev.shape[1])
    if t == action_horizon:
        return prev
    if t > action_horizon:
        return prev[:, :action_horizon, :]
    pad = jnp.zeros((prev.shape[0], action_horizon - t, prev.shape[2]), dtype=prev.dtype)
    return jnp.concatenate([prev, pad], axis=1)


class RLTPolicy(_base_policy.BasePolicy):
    """Policy that returns both VLA actions and RL tokens via WebSocket.

    Supports single and batch inference:
    - Single: obs dict → result dict
    - Batch:  {"batch": [obs1, obs2, ...]} → {"batch_results": [r1, r2, ...]}
    """

    def __init__(
        self,
        model: RLTInferenceModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        metadata: dict[str, Any] | None = None,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._metadata = metadata or {}
        self._rng = rng or jax.random.key(0)
        self._action_horizon = int(getattr(model.vla, "action_horizon", CHUNK_LEN))
        self._infer_fn = nnx_utils.module_jit(model.infer)
        # s/d must be static: guided_inference uses them as concrete Python ints, and
        # deploy warmup intentionally compiles a separate graph per (s, d) pair.
        self._infer_guided_fn = nnx_utils.module_jit(
            model.infer_guided, static_argnames=("s", "d")
        )

    @override
    def infer(
        self,
        obs: dict,
        use_rtc: bool = False,
        noise: np.ndarray | None = None,
    ) -> dict:
        del noise  # RLT feature server does not accept custom noise yet.
        if "batch" in obs:
            if use_rtc:
                logging.warning("RLT batch inference ignores use_rtc; falling back to sample_actions.")
            return self._infer_batch(obs["batch"])
        return self._infer_single(obs, use_rtc=use_rtc)

    @staticmethod
    def _as_action_chunk(actions: np.ndarray) -> np.ndarray:
        """Normalize model output to (T, action_dim)."""
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim == 3:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got {arr.shape[0]}")
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected action chunk shape (T, D) or (1, T, D), got {arr.shape}")
        return arr

    def _infer_single(self, obs: dict, *, use_rtc: bool = False) -> dict:
        """Single observation inference (original behavior)."""
        # RTC control knobs are client-side metadata; keep them out of model transforms.
        inputs = dict(obs)
        inference_delay = int(inputs.pop("inference_delay", 0) or 7)
        execution_horizon = int(inputs.pop("rtc_execution_horizon", 25) or 25)

        inputs = self._input_transform(inputs)
        # Match openpi.policies.policy.Policy: leftover actions pass through DeltaActions etc.
        prev_action = inputs.pop("actions", None) if use_rtc else None
        if not use_rtc:
            inputs.pop("actions", None)

        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if use_rtc:
            # Align with openpi.policies.policy.Policy: use_rtc alone selects guided_inference.
            # prev_action may be None on the first chunk; pi0.guided_inference then uses step_normal.
            guided_prev = None
            if prev_action is not None:
                guided_prev = jnp.asarray(prev_action, dtype=jnp.float32)
                if guided_prev.ndim == 2:
                    guided_prev = guided_prev[np.newaxis, ...]
                # Client hold-pads leftover to ``s``; remaining ``H - s`` is W=0, so
                # zero-pad here is only for a stable JIT time dimension.
                guided_prev = _pad_prev_action_time(guided_prev, self._action_horizon)
            s = max(1, int(execution_horizon))
            # LeRobot get_prefix_weights: start = min(start, end). When d >= s the soft
            # band collapses and the first s steps are hard-locked (weight=1).
            d = max(0, min(int(inference_delay), s))
            if d != int(inference_delay):
                logging.warning(
                    "Clamping RTC inference_delay %d -> %d for execution_horizon s=%d",
                    int(inference_delay),
                    d,
                    s,
                )
            actions, rl_token = self._infer_guided_fn(
                sample_rng,
                observation,
                prev_action=guided_prev,
                s=s,
                d=d,
            )
        else:
            actions, rl_token = self._infer_fn(sample_rng, observation)
        infer_time = time.monotonic() - start_time

        return self._build_single_result(
            actions=self._as_action_chunk(actions),
            rl_token=np.asarray(rl_token[0]),
            state=np.asarray(inputs["state"])[0],
            infer_time=infer_time,
        )

    # Pre-compiled batch sizes used for padding and startup warmup.
    COMPILED_BATCH_SIZES: ClassVar[list[int]] = [1, 2, 4, 6, 8, 10, 12, 16]

    @staticmethod
    def _pad_to_compiled_size(n: int) -> int:
        """Find the smallest pre-compiled batch size >= n."""
        for size in RLTPolicy.COMPILED_BATCH_SIZES:
            if size >= n:
                return size
        # If larger than all pre-compiled sizes, use as-is (will trigger JIT)
        return n

    def _infer_batch(self, obs_list: list[dict]) -> dict:
        """Batch inference: process multiple observations in one model forward pass.

        Pads to nearest pre-compiled batch size to avoid JIT recompilation.
        """
        real_batch_size = len(obs_list)
        if real_batch_size == 0:
            return {"batch_results": []}

        start_time = time.monotonic()

        # Step 1: Apply input_transform to each observation individually
        transformed_list = []
        for obs in obs_list:
            inp = jax.tree.map(lambda x: x, obs)
            inp = self._input_transform(inp)
            transformed_list.append(inp)

        # Step 2: Pad to nearest pre-compiled batch size by repeating last observation
        padded_size = self._pad_to_compiled_size(real_batch_size)
        while len(transformed_list) < padded_size:
            transformed_list.append(transformed_list[-1])  # duplicate last

        # Step 3: Stack into batch tensors
        batch_inputs = jax.tree.map(
            lambda *vals: jnp.stack([jnp.asarray(v) for v in vals], axis=0),
            *transformed_list,
        )

        # Step 4: One model forward pass (padded batch)
        self._rng, sample_rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(batch_inputs)
        actions, rl_token = self._infer_fn(sample_rng, observation)

        infer_time = time.monotonic() - start_time

        # Step 5: Take only the real results (discard padding)
        actions_np = np.asarray(actions)[:real_batch_size]
        rl_token_np = np.asarray(rl_token)[:real_batch_size]
        states_np = np.asarray(batch_inputs["state"])[:real_batch_size]

        results = []
        for i in range(real_batch_size):
            result = self._build_single_result(
                actions=self._as_action_chunk(actions_np[i]),
                rl_token=rl_token_np[i],
                state=states_np[i],
                infer_time=infer_time / real_batch_size,
            )
            results.append(result)

        return {
            "batch_results": results,
            "batch_size": real_batch_size,
            "padded_size": padded_size,
            "total_infer_ms": infer_time * 1000,
            "per_sample_infer_ms": infer_time / real_batch_size * 1000,
        }

    def _build_single_result(
        self,
        actions: np.ndarray,
        rl_token: np.ndarray,
        state: np.ndarray,
        infer_time: float,
    ) -> dict:
        """Build output dict for one sample."""
        # Save rl_token before output_transform
        rl_token_flat = rl_token.reshape(-1).astype(np.float32)
        raw_state = state
        if raw_state.ndim > 1:
            raw_state = raw_state[0]

        outputs = {
            "state": np.array(raw_state, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
        }
        outputs = self._output_transform(outputs)

        # z_rl
        z_rl = rl_token_flat

        # proprio
        proprio = np.zeros(PROPRIO_DIM, dtype=np.float32)
        n = min(PROPRIO_DIM, raw_state.shape[0])
        proprio[:n] = raw_state[:n].astype(np.float32)

        # ref_chunk: VLA action chunk for RLT rollout and robot clients.
        vla_actions = np.asarray(outputs["actions"], dtype=np.float32)
        # actions shape : (50, 7)
        if vla_actions.ndim == 1:
            vla_actions = vla_actions[np.newaxis, :]
        ref_chunk = vla_actions[:CHUNK_LEN, :ACTION_DIM].astype(np.float32)

        return {
            "z_rl": z_rl,
            "proprio": proprio,
            "ref_chunk": ref_chunk,
            "policy_timing": {"infer_ms": infer_time * 1000},
            "actions": ref_chunk,
            "_raw_rl_token": rl_token,
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def load_rlt_model(
    config: _config.TrainConfig,
    checkpoint_dir: str,
    *,
    shared_prefix_inference: bool = False,
) -> RLTInferenceModel:
    """Load RLT model from checkpoint."""
    checkpoint_path = pathlib.Path(checkpoint_dir)
    params_path = checkpoint_path / "params"

    rlt_config = _create_rlt_config(config)
    prefix_seq_len = _infer_prefix_seq_len(config.model)

    logging.info(f"RLT config: {rlt_config}, prefix_seq_len: {prefix_seq_len}")

    vla_model = nnx.eval_shape(config.model.create, jax.random.key(0))
    model = RLTInferenceModel(
        vla_model,
        rlt_config,
        rngs=nnx.Rngs(jax.random.key(0)),
        prefix_seq_len=prefix_seq_len,
        shared_prefix_inference=shared_prefix_inference,
    )

    logging.info(f"Loading params from {params_path}")
    loaded_params = _model.restore_params(params_path, dtype=jnp.bfloat16)

    graphdef, state = nnx.split(model)
    import orbax.checkpoint as ocp

    loaded_params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), loaded_params)
    state.replace_by_pure_dict(loaded_params)
    model = nnx.merge(graphdef, state)

    logging.info("RLT model loaded successfully")
    return model


@dataclasses.dataclass
class Args:
    config: str = "rlt_pi05_agilexbag_image"
    checkpoint_dir: str = ""
    port: int = 8000
    default_prompt: str | None = None
    shared_prefix_inference: bool = False
    # Pre-compile guided_inference graphs before serving.
    # ``s`` is phase-stable after client hold-pad (VLA 25 / RL 10), not leftover-clamped.
    # ``d`` is a separate graph only if inference_delay is not a single fixed value.
    rtc_warmup_s: tuple[int, ...] = (10, 25)
    rtc_warmup_d: tuple[int, ...] = (7,)
    skip_rtc_warmup: bool = False


def warmup_rlt_policy_graphs(
    policy: RLTPolicy,
    model_config: _model.BaseModelConfig,
    *,
    rtc_s_values: Sequence[int],
    rtc_d_values: Sequence[int],
) -> None:
    """JIT-warm sample_actions + guided_inference for each deploy (s, d).

    ``s``/``d`` are static JIT args (``make_W``). After client hold-pad, leftover
    length no longer changes ``s``; warm one graph per phase ``s`` and each
    expected ``d``. ``prev_action=None`` is a second trace (first chunk).
    """
    observation = model_config.fake_obs(batch_size=1)
    policy._rng, sample_rng = jax.random.split(policy._rng)
    logging.info("Warming up sample_actions / infer ...")
    t0 = time.monotonic()
    policy._infer_fn(sample_rng, observation)
    logging.info("sample_actions warmup done in %.1fs", time.monotonic() - t0)

    action_dim = int(getattr(policy._model.vla, "action_dim", ACTION_DIM))
    prev = jnp.zeros((1, policy._action_horizon, action_dim), dtype=jnp.float32)
    pairs: list[tuple[int, int]] = []
    for s_raw in rtc_s_values:
        s = max(1, int(s_raw))
        for d_raw in rtc_d_values:
            d_eff = max(0, min(int(d_raw), s))
            pairs.append((s, d_eff))
    # Deduplicate while preserving order.
    seen: set[tuple[int, int]] = set()
    unique_pairs: list[tuple[int, int]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    logging.info(
        "Warming up %d guided_inference (s,d) graphs (+ first-chunk prev=None): %s",
        len(unique_pairs),
        ", ".join(f"({s},{d_eff})" for s, d_eff in unique_pairs),
    )
    for s, d_eff in unique_pairs:
        policy._rng, sample_rng = jax.random.split(policy._rng)
        logging.info("Warming up guided_inference s=%d d=%d (with prev) ...", s, d_eff)
        t0 = time.monotonic()
        policy._infer_guided_fn(
            sample_rng,
            observation,
            prev_action=prev,
            s=s,
            d=d_eff,
        )
        logging.info("guided_inference s=%d d=%d warmup done in %.1fs", s, d_eff, time.monotonic() - t0)

    # First RTC chunk has no leftover: still calls guided_inference (step_normal). Warm that
    # specialization for the largest s (VLA preferred) to avoid a cold compile on episode start.
    s0, d0 = max(unique_pairs, key=lambda sd: sd[0])
    policy._rng, sample_rng = jax.random.split(policy._rng)
    logging.info("Warming up guided_inference s=%d d=%d (prev=None / first chunk) ...", s0, d0)
    t0 = time.monotonic()
    policy._infer_guided_fn(
        sample_rng,
        observation,
        prev_action=None,
        s=s0,
        d=d0,
    )
    logging.info(
        "guided_inference s=%d d=%d prev=None warmup done in %.1fs",
        s0,
        d0,
        time.monotonic() - t0,
    )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    config = _config.get_config(args.config)
    if config.rlt_num_tokens is None:
        raise ValueError("Config must have RLT fields set.")

    model = load_rlt_model(
        config,
        args.checkpoint_dir,
        shared_prefix_inference=args.shared_prefix_inference,
    )

    data_config = config.data.create(config.assets_dirs, config.model)
    checkpoint_path = pathlib.Path(args.checkpoint_dir)
    norm_stats = None
    if data_config.asset_id is not None:
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_path / "assets", data_config.asset_id)
        except Exception as e:
            logging.warning(f"Could not load norm stats: {e}")

    transforms_list = [
        _transforms.InjectDefaultPrompt(args.default_prompt),
        *data_config.data_transforms.inputs,
    ]
    if norm_stats:
        transforms_list.append(_transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    transforms_list.extend(data_config.model_transforms.inputs)

    output_transforms = list(data_config.model_transforms.outputs)
    if norm_stats:
        output_transforms.append(_transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm))
    output_transforms.extend(data_config.data_transforms.outputs)
    output_transforms.append(_transforms.OptimizeActionsBSpline(k=3, n_ctrl=10, verbose=False))

    policy = RLTPolicy(
        model,
        transforms=transforms_list,
        output_transforms=output_transforms,
        metadata={
            **(config.policy_metadata or {}),
            "has_rl_token": True,
            "z_dim": 2048,
            "proprio_dim": PROPRIO_DIM,
            "chunk_len": CHUNK_LEN,
            "action_dim": ACTION_DIM,
            "supports_batch": True,
            "shared_prefix_inference": args.shared_prefix_inference,
            "rtc_warmup_s": list(args.rtc_warmup_s),
            "rtc_warmup_d": list(args.rtc_warmup_d),
        },
    )

    if not args.skip_rtc_warmup:
        warmup_rlt_policy_graphs(
            policy,
            config.model,
            rtc_s_values=args.rtc_warmup_s,
            rtc_d_values=args.rtc_warmup_d,
        )

    # Serve
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Creating RLT server (host: {hostname}, ip: {local_ip}, port: {args.port})")

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
