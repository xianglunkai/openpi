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
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
            self._guided_inference = model.guided_inference
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._guided_inference = nnx_utils.module_jit(model.guided_inference)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self, 
        obs: dict, 
        *, 
        prev_action: np.ndarray | None = None,
        use_rtc: bool = False,
        noise: np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        
        if use_rtc:
            if prev_action is None:
                # First RTC call: fall back to normal sampling (but with RTC parameters).
                origin_actions = self._sample_actions(
                    sample_rng_or_pytorch_device,
                    observation,
                    **sample_kwargs,
                )
            else:
                # Subsequent RTC call: use guided_inference. This API returns only actions,
                # so we construct a simple timing dict here.
                if self._is_pytorch_model:
                    prev_action = torch.from_numpy(prev_action.copy()).unsqueeze(0).to(self._pytorch_device)
                else:
                    prev_action = jnp.asarray(prev_action)[np.newaxis, ...]
                    
                origin_actions = self._guided_inference(
                    sample_rng_or_pytorch_device,
                    prev_action = prev_action,
                    observation = observation,
                    **sample_kwargs,
                )
                
        else:
            # Non-RTC path: standard sampling with full sample_kwargs (including s/d/noise).
            origin_actions = self._sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                **sample_kwargs,
            )
          
            
        outputs = {
            "state": inputs["state"],
            "actions": origin_actions,
            "origin_actions": origin_actions,
        }
        
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        origin_actions_backup = outputs.get("origin_actions")
        
        outputs = self._output_transform(outputs)
        if "origin_actions" not in outputs and origin_actions_backup is not None:
            outputs["origin_actions"] = origin_actions_backup
        
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
       
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
