#!/usr/bin/env python3
"""Evaluation and visualization script for RLT (RL Token) encoder-decoder.

Usage:
    python scripts/eval_rlt.py \
        --config rlt_pi05_agilexbag_image \
        --checkpoint-dir checkpoints/rlt_pi05_agilexbag_image/rlt_agilexbag_5k/4999 \
        --output-dir /tmp/rlt_eval \
        --num-samples 32
"""

import argparse
import json
import logging
import pathlib

import matplotlib

matplotlib.use("Agg")
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import openpi.models.model as _model
from openpi.models.rl_token import RLTokenConfig
from openpi.models.rl_token import RLTokenModel
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RLT encoder-decoder.")
    parser.add_argument("--config", type=str, required=True, help="Training config name.")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Path to checkpoint directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--num-samples", type=int, default=32, help="Number of samples to evaluate.")
    return parser.parse_args()


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


def _infer_prefix_seq_len(model_config: _model.BaseModelConfig, image_only: bool = True) -> int:
    def _get_len(rng):
        model = model_config.create(rng)
        obs = model_config.fake_obs(batch_size=1)
        embs, _ = model.extract_prefix_embeddings(rng, obs, image_only=image_only)
        return embs

    embs_shape = jax.eval_shape(_get_len, jax.random.key(0))
    return embs_shape.shape[1]


class RLTEvalModel(nnx.Module):
    """Same structure as RLTTrainModel in train_rlt.py — uses nnx_bridge.ToNNX."""

    def __init__(
        self,
        vla_model: _model.BaseModel,
        rlt_config: RLTokenConfig,
        rngs: nnx.Rngs,
        prefix_seq_len: int = 968,
    ):
        self.vla = vla_model
        linen_rlt = RLTokenModel(config=rlt_config)
        self.rlt_module = nnx_bridge.ToNNX(linen_rlt)
        dummy_prefix = jnp.zeros((1, prefix_seq_len, rlt_config.input_dim))
        dummy_mask = jnp.ones((1, prefix_seq_len), dtype=jnp.bool_)
        self.rlt_module.lazy_init(dummy_prefix, dummy_mask, rngs=rngs)
        self.deterministic = True

    def extract_and_evaluate(
        self,
        rng: jax.Array,
        observation: _model.Observation,
    ) -> dict[str, jnp.ndarray]:
        prefix_embs, prefix_mask = self.vla.extract_prefix_embeddings(rng, observation, train=False, image_only=True)
        prefix_f32 = prefix_embs.astype(jnp.float32)

        # Compute reconstruction loss (with mask fallback)
        rlt_loss, rlt_info = self.rlt_module(prefix_f32, prefix_mask, train=False)
        has_valid = jnp.sum(prefix_mask.astype(jnp.float32)) > 0
        rlt_loss_unmasked, rlt_info_unmasked = self.rlt_module(prefix_f32, None, train=False)
        mse = jnp.where(has_valid, rlt_info["mse"], rlt_info_unmasked["mse"])

        # Encode to RL tokens
        rl_tokens = self.rlt_module(prefix_f32, prefix_mask, method="encode", train=False)

        # Decode back
        reconstructed = self.rlt_module(rl_tokens, prefix_f32.shape[-2], method="decode", train=False)

        # Per-token cosine similarity (unmasked for simplicity)
        orig_norm = prefix_f32 / (jnp.linalg.norm(prefix_f32, axis=-1, keepdims=True) + 1e-8)
        recon_norm = reconstructed / (jnp.linalg.norm(reconstructed, axis=-1, keepdims=True) + 1e-8)
        cosine_sim = jnp.mean(jnp.sum(orig_norm * recon_norm, axis=-1))

        mae = jnp.mean(jnp.abs(reconstructed - prefix_f32))

        return {
            "mse": mse,
            "mae": mae,
            "cosine_sim": cosine_sim,
            "rl_tokens": rl_tokens,
        }


def pca_2d(data: np.ndarray) -> np.ndarray:
    centered = data - data.mean(axis=0, keepdims=True)
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    return u[:, :2] * s[:2]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    rng = jax.random.key(SEED)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading config: {args.config}")
    config = _config.get_config(args.config)
    if config.rlt_num_tokens is None:
        raise ValueError("Config must have RLT fields set.")

    rlt_config = _create_rlt_config(config)
    prefix_seq_len = _infer_prefix_seq_len(config.model)
    logging.info(f"RLT config: {rlt_config}, prefix_seq_len: {prefix_seq_len}")

    # Create model with same structure as training
    logging.info("Creating model...")
    rng, model_rng, rlt_rng = jax.random.split(rng, 3)
    vla_model = config.model.create(model_rng)
    eval_model = RLTEvalModel(vla_model, rlt_config, rngs=nnx.Rngs(rlt_rng), prefix_seq_len=prefix_seq_len)

    # Load checkpoint
    logging.info(f"Loading checkpoint from: {args.checkpoint_dir}")
    checkpoint_path = pathlib.Path(args.checkpoint_dir)
    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        checkpoint_path.parent,
        overwrite=False,
        resume=True,
    )

    # Build train state shape for restore
    mesh = sharding.make_mesh(config.fsdp_devices)
    params = nnx.state(eval_model)
    graphdef = nnx.graphdef(eval_model)
    dummy_tx = jax.tree.map(lambda _: None, params)  # placeholder

    train_state = training_utils.TrainState(
        step=0,
        params=params,
        model_def=graphdef,
        tx=None,
        opt_state=None,
        ema_decay=None,
        ema_params=None,
    )

    try:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, None)
        eval_model = nnx.merge(train_state.model_def, train_state.params)
        logging.info(f"Checkpoint restored at step {train_state.step}")
    except Exception as e:
        logging.warning(f"Could not restore checkpoint: {e}. Using random weights.")

    # Create data loader
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=False)
    data_iter = iter(data_loader)

    # Evaluate
    logging.info(f"Evaluating on {args.num_samples} samples...")
    all_mse, all_mae, all_cosine = [], [], []
    all_rl_tokens = []
    samples_collected = 0
    rng, eval_rng = jax.random.split(rng)

    while samples_collected < args.num_samples:
        batch = next(data_iter)
        observation, _ = batch
        batch_size = observation.state.shape[0]

        results = eval_model.extract_and_evaluate(eval_rng, observation)
        all_mse.append(float(results["mse"]))
        all_mae.append(float(results["mae"]))
        all_cosine.append(float(results["cosine_sim"]))
        all_rl_tokens.append(np.array(results["rl_tokens"]))

        samples_collected += batch_size
        logging.info(
            f"  batch {len(all_mse)}: mse={all_mse[-1]:.4f}, mae={all_mae[-1]:.4f}, cosine={all_cosine[-1]:.4f}"
        )

    # Aggregate metrics
    metrics = {
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse)),
        "mae_mean": float(np.mean(all_mae)),
        "mae_std": float(np.std(all_mae)),
        "cosine_sim_mean": float(np.mean(all_cosine)),
        "cosine_sim_std": float(np.std(all_cosine)),
        "num_samples": samples_collected,
        "config": args.config,
        "checkpoint": args.checkpoint_dir,
    }

    metrics_path = output_dir / "reconstruction_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {metrics_path}")
    logging.info(
        f"Results: MSE={metrics['mse_mean']:.4f}±{metrics['mse_std']:.4f}, "
        f"MAE={metrics['mae_mean']:.4f}±{metrics['mae_std']:.4f}, "
        f"Cosine={metrics['cosine_sim_mean']:.4f}±{metrics['cosine_sim_std']:.4f}"
    )

    # PCA visualization
    all_tokens = np.concatenate(all_rl_tokens, axis=0)[: args.num_samples]
    tokens_avg = all_tokens.mean(axis=1)  # [n_samples, dim]

    if tokens_avg.shape[0] >= 3:
        tokens_2d = pca_2d(tokens_avg)

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            tokens_2d[:, 0], tokens_2d[:, 1], c=np.arange(len(tokens_2d)), cmap="viridis", alpha=0.7, s=50
        )
        plt.colorbar(scatter, ax=ax, label="Sample Index")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA of RL Tokens (n={len(tokens_2d)})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        pca_path = output_dir / "pca_2d.png"
        plt.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved PCA visualization to {pca_path}")
    else:
        logging.warning("Too few samples for PCA visualization.")

    logging.info("Evaluation complete!")


if __name__ == "__main__":
    main()
