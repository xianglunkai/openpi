import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
from openpi.models.rl_token import RLTokenConfig
from openpi.models.rl_token import RLTokenModel
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_vla_weights(loader: _weight_loaders.WeightLoader, full_params_shape: at.Params) -> at.Params:
    """Load pretrained VLA weights into the RLT composite model.

    The checkpoint contains VLA-only params (e.g. PaliGemma/...), but our model
    wraps them under a 'vla' prefix (vla/PaliGemma/...) alongside rlt_module/.
    We extract the vla/ subtree shape, load weights into it, then re-prefix.
    """
    flat_shape = traverse_util.flatten_dict(full_params_shape)
    vla_shape = traverse_util.unflatten_dict({k[1:]: v for k, v in flat_shape.items() if k[0] == "vla"})
    loaded_vla = loader.load(vla_shape)
    at.check_pytree_equality(expected=vla_shape, got=loaded_vla, check_shapes=True, check_dtypes=True)
    flat_loaded = traverse_util.flatten_dict(loaded_vla)
    reprefixed = {("vla",) + k: v for k, v in flat_loaded.items() if not isinstance(v, jax.ShapeDtypeStruct)}
    return traverse_util.unflatten_dict(reprefixed)


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


def _get_rlt_alpha(config: _config.TrainConfig) -> float:
    return config.rlt_alpha if config.rlt_alpha is not None else 0.0


def _make_rlt_trainable_filter(alpha: float) -> nnx.filterlib.Filter:
    if alpha == 0.0:
        return nnx.All(nnx.Param, nnx_utils.PathRegex(".*rlt_module.*"))
    return nnx.Param


def _make_rlt_freeze_filter(alpha: float) -> nnx.filterlib.Filter:
    if alpha == 0.0:
        return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*rlt_module.*")))
    return nnx.Nothing


def _infer_prefix_seq_len(model_config: _model.BaseModelConfig, image_only: bool = True) -> int:
    def _get_len(rng):
        model = model_config.create(rng)
        obs = model_config.fake_obs(batch_size=1)
        embs, _ = model.extract_prefix_embeddings(rng, obs, image_only=image_only)
        return embs

    embs_shape = jax.eval_shape(_get_len, jax.random.key(0))
    return embs_shape.shape[1]


class RLTTrainModel(nnx.Module):
    def __init__(
        self,
        vla_model: _model.BaseModel,
        rlt_config: RLTokenConfig,
        rngs: nnx.Rngs,
        prefix_seq_len: int = 768,
    ):
        self.vla = vla_model
        linen_rlt = RLTokenModel(config=rlt_config)
        self.rlt_module = nnx_bridge.ToNNX(linen_rlt)
        dummy_prefix = jnp.zeros((1, prefix_seq_len, rlt_config.input_dim))
        dummy_mask = jnp.ones((1, prefix_seq_len), dtype=jnp.bool_)
        self.rlt_module.lazy_init(dummy_prefix, dummy_mask, rngs=rngs)

        self.deterministic = True

    def compute_rlt_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        alpha: float,
        *,
        train: bool = False,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        if alpha > 0.0:
            # Joint training: single VLA forward for both prefix_embs and vla_loss
            vla_per_sample_loss, prefix_embs, prefix_mask = self.vla.compute_loss_with_prefix(
                rng, observation, actions, train=train, image_only=True
            )
            vla_loss = jnp.mean(vla_per_sample_loss)
        else:
            # Frozen VLA: prefix-only forward
            prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(
                rng, observation, train=train, image_only=True
            )
            vla_loss = None

        # Per paper: Lro ALWAYS uses stop-gradient on VLA embeddings
        prefix_embs_sg = jax.lax.stop_gradient(prefix_embs)
        prefix_embs_f32 = prefix_embs_sg.astype(jnp.float32)

        # RL Token reconstruction loss (Lro)
        rlt_loss, rlt_info = self.rlt_module(prefix_embs_f32, None, train=train)
        mse = rlt_info["mse"]

        info = {"rlt_loss": rlt_loss, "mse": mse}
        total_loss = rlt_loss

        if vla_loss is not None:
            total_loss = rlt_loss + alpha * vla_loss
            info["vla_loss"] = vla_loss

        info["total_loss"] = total_loss
        return total_loss, info


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    rlt_config = _create_rlt_config(config)
    alpha = _get_rlt_alpha(config)
    trainable_filter = _make_rlt_trainable_filter(alpha)
    freeze_filter = _make_rlt_freeze_filter(alpha)
    prefix_seq_len = _infer_prefix_seq_len(config.model)

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng, rlt_rng = jax.random.split(rng, 3)
        vla_model = config.model.create(model_rng)
        model = RLTTrainModel(vla_model, rlt_config, rngs=nnx.Rngs(rlt_rng), prefix_seq_len=prefix_seq_len)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_vla_weights(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    alpha = _get_rlt_alpha(config)
    trainable_filter = _make_rlt_trainable_filter(alpha)

    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model: RLTTrainModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        total_loss, info = model.compute_rlt_loss(rng, observation, actions, alpha, train=True)
        return total_loss, info

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, trainable_filter)
    (loss, info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    step_info = {
        "loss": loss,
        "rlt_loss": info["rlt_loss"],
        "mse": info["mse"],
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    if "vla_loss" in info:
        step_info["vla_loss"] = info["vla_loss"]
    return new_state, step_info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running RLT Stage 1 training on: {platform.node()}")

    if config.rlt_num_tokens is None:
        raise ValueError("RLT config fields (rlt_num_tokens, etc.) must be set for RLT training.")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0 or step == config.num_train_steps - 1:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
