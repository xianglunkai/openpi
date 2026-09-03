from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.recap.config import RecapConfig
from openpi.recap.tags import build_acp_tagged_task
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

import enum
from openpi.policies.aloha_policy import make_aloha_example
from openpi.policies.droid_policy import make_droid_example
from openpi.policies.libero_policy import make_libero_example
from openpi.policies.cobot_policy import make_cobot_example


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    COBOT = "cobot"
    COBOT_REALTIME = "cobot_realtime"
    
class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        recap: RecapConfig | None = None,
        default_prompt: str | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            recap: Loaded from ``TrainConfig.recap``. Serving uses this automatically.
            default_prompt: Fallback language prompt used when the observation has none.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._recap = recap if recap is not None and recap.active_at_inference else None
        self._default_prompt = default_prompt

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._guided_inference = model.guided_inference
            self._sample_actions_cfg = getattr(model, "sample_actions_cfg", None) if self._recap else None
        else:
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._guided_inference = nnx_utils.module_jit(model.guided_inference)
            self._rng = rng or jax.random.key(0)
            self._sample_actions_cfg = None
            if self._recap is not None and hasattr(model, "sample_actions_cfg"):
                self._sample_actions_cfg = nnx_utils.module_jit(model.sample_actions_cfg)

    def _prompt_of(self, obs: dict) -> str:
        prompt = obs.get("prompt")
        if prompt is None:
            return self._default_prompt or ""
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
        return prompt

    def _encode(self, obs: dict, prompt: str | None = None):
        """Apply input transforms and add a batch dimension."""
        inputs = jax.tree.map(lambda x: x, obs)
        if prompt is not None:
            inputs["prompt"] = prompt
        inputs = self._input_transform(inputs)
        if self._is_pytorch_model:
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs
            )
        else:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return inputs, _model.Observation.from_dict(inputs)

    def _predict(self, obs: dict, rng, sample_kwargs: dict[str, Any]):
        """Sample actions. Recap CFG is selected from TrainConfig at construction."""
        recap = self._recap
        if recap is None:
            inputs, observation = self._encode(obs)
            return inputs, self._sample_actions(rng, observation, **sample_kwargs)

        base_prompt = self._prompt_of(obs)
        if recap.guidance_type == "no_guide":
            inputs, observation = self._encode(obs, base_prompt)
            return inputs, self._sample_actions(rng, observation, **sample_kwargs)

        inputs, observation_cond = self._encode(obs, build_acp_tagged_task(base_prompt, True))
        if recap.cfg_guidance_scale == 1.0 or self._sample_actions_cfg is None:
            return inputs, self._sample_actions(rng, observation_cond, **sample_kwargs)

        _, observation_uncond = self._encode(obs, base_prompt)
        cfg_kwargs = {key: value for key, value in sample_kwargs.items() if key in {"num_steps", "noise"}}
        cfg_kwargs["guidance_scale"] = recap.cfg_guidance_scale
        return inputs, self._sample_actions_cfg(rng, observation_uncond, observation_cond, **cfg_kwargs)

    @override
    def infer(
        self,
        obs: dict,
        use_rtc: bool = False,
        noise: np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        if not self._is_pytorch_model:
            self._rng, rng = jax.random.split(self._rng)
        else:
            rng = self._pytorch_device

        start_time = time.monotonic()
        if use_rtc:
            inputs, observation = self._encode(obs)
            prev_action = inputs.get("actions")
            if prev_action is not None:
                prev_action = (
                    torch.as_tensor(prev_action).to(self._pytorch_device)
                    if self._is_pytorch_model
                    else jnp.asarray(prev_action)
                )
            origin_actions = self._guided_inference(
                rng,
                prev_action=prev_action,
                observation=observation,
                **sample_kwargs,
            )
        else:
            inputs, origin_actions = self._predict(obs, rng, sample_kwargs)

        outputs = {
            "state": inputs["state"],
            "actions": origin_actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
    def make_example(self) -> dict:
        assert "env" in self._metadata, "Environment not set in metadata"
        env = EnvMode(self._metadata["env"])
        if env == EnvMode.ALOHA:
            return make_aloha_example()
        if env == EnvMode.DROID:
            return make_droid_example()
        if env in [EnvMode.COBOT, EnvMode.COBOT_REALTIME]:
            return make_cobot_example()

        raise ValueError(f"Unknown environment: {env}")

class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
