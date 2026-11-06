#!/usr/bin/env python3
"""Compare the JAX checkpoint with the converted PyTorch weights on real frames.

Both policies see the same image, the same 7-D state, and the same flow-matching
noise, then pass through the same output transforms the robot server uses
(quantile unnormalize, delta->absolute, 7-D slice, QP smoother). The cosine is
therefore the command the arm would receive, not a raw network dump.

Dataset default is the local screw-sorting SFT set. Pass --pytorch_dir after
examples/convert_jax_model_to_pytorch.py and the norm-stats copy.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import time

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

PROMPT = "Please sort and return the silver screws in the grey box to their proper places."
JAX_DIR = (
    "/home/xlk/work/openpi/checkpoints/pi05_cobot/"
    "pi05_cobot_screw_sorting_single/pi05_cobot_screw_sorting_single/29999"
)
DATASET_ROOT = "/data/huggingface/lerobot/openpi/screw_sorting_single_sft"
REPO_ID = "openpi/screw_sorting_single_sft"
CONFIG_NAME = "pi05_cobot_screw_sorting_single"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_dataset(root: str, repo_id: str, action_horizon: int):
    os.environ.setdefault("HF_LEROBOT_HOME", "/data/huggingface/lerobot")
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    meta_fps = json.loads((pathlib.Path(root) / "meta" / "info.json").read_text())["fps"]
    delta = {"action": [t / meta_fps for t in range(action_horizon)]}
    return LeRobotDataset(repo_id, root=root, delta_timestamps=delta, download_videos=False)


def _obs_from_sample(sample) -> tuple[dict, np.ndarray | None]:
    obs = {
        "state": _as_numpy(sample["observation.state"]).astype(np.float32).reshape(-1)[:7],
        "images": {
            "cam_high": _as_numpy(sample["observation.images.cam_high"]),
            "cam_right_wrist": _as_numpy(sample["observation.images.cam_right_wrist"]),
        },
        "prompt": PROMPT,
    }
    demo = None
    if "action" in sample:
        demo = _as_numpy(sample["action"]).astype(np.float32)
        if demo.ndim == 1:
            demo = demo.reshape(1, -1)
        demo = demo[:, :7]
    return obs, demo


def _timed_infer(policy, obs, noise) -> tuple[np.ndarray, float]:
    started = time.perf_counter()
    out = policy.infer(obs, noise=noise)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return np.asarray(out["actions"], dtype=np.float32)[:, :7], elapsed_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax_dir", default=JAX_DIR)
    parser.add_argument("--pytorch_dir", required=True)
    parser.add_argument("--dataset_root", default=DATASET_ROOT)
    parser.add_argument("--repo_id", default=REPO_ID)
    parser.add_argument("--num", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_cosine", type=float, default=0.99)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    pytorch_dir = pathlib.Path(args.pytorch_dir)
    if not (pytorch_dir / "model.safetensors").is_file():
        raise SystemExit(f"missing {pytorch_dir / 'model.safetensors'}")
    stats = pytorch_dir / "assets" / "screw_sorting_single" / "norm_stats.json"
    if not stats.is_file():
        raise SystemExit(f"missing {stats}. Copy it from the 29999 checkpoint assets directory.")

    train_config = _config.get_config(CONFIG_NAME)
    print("loading JAX", args.jax_dir, flush=True)
    jax_policy = _policy_config.create_trained_policy(train_config, args.jax_dir, default_prompt=PROMPT)
    print("loading PyTorch", pytorch_dir, flush=True)
    torch_policy = _policy_config.create_trained_policy(train_config, pytorch_dir, default_prompt=PROMPT)

    dataset = _load_dataset(args.dataset_root, args.repo_id, train_config.model.action_horizon)
    indices = np.unique(np.linspace(0, len(dataset) - 1, num=args.num, dtype=int))
    noise = np.random.default_rng(args.seed).standard_normal((train_config.model.action_horizon, train_config.model.action_dim))
    noise = noise.astype(np.float32)

    warmup_obs, _ = _obs_from_sample(dataset[int(indices[0])])
    print("warmup (not timed)", flush=True)
    _timed_infer(jax_policy, warmup_obs, noise)
    _timed_infer(torch_policy, warmup_obs, noise)

    rows = []
    for index in indices:
        obs, demo = _obs_from_sample(dataset[int(index)])
        jax_actions, jax_ms = _timed_infer(jax_policy, obs, noise)
        torch_actions, torch_ms = _timed_infer(torch_policy, obs, noise)
        n = min(len(jax_actions), len(torch_actions))
        row = {
            "index": int(index),
            "cosine": _cosine(jax_actions[:n], torch_actions[:n]),
            "max_abs": float(np.max(np.abs(jax_actions[:n] - torch_actions[:n]))),
            "jax_ms": jax_ms,
            "torch_ms": torch_ms,
        }
        if demo is not None:
            m = min(n, len(demo))
            row["jax_demo_cosine"] = _cosine(jax_actions[:m], demo[:m])
            row["torch_demo_cosine"] = _cosine(torch_actions[:m], demo[:m])
        rows.append(row)
        print(
            f"frame {index:6d}  cosine {row['cosine']:.6f}  max_abs {row['max_abs']:.5f}  "
            f"jax {jax_ms:.0f} ms  torch {torch_ms:.0f} ms",
            flush=True,
        )

    cosines = np.array([row["cosine"] for row in rows], dtype=np.float64)
    summary = {
        "num": len(rows),
        "cosine_mean": float(np.nanmean(cosines)),
        "cosine_min": float(np.nanmin(cosines)),
        "max_abs_mean": float(np.mean([row["max_abs"] for row in rows])),
        "jax_ms_p50": float(np.median([row["jax_ms"] for row in rows])),
        "torch_ms_p50": float(np.median([row["torch_ms"] for row in rows])),
        "min_cosine": args.min_cosine,
        "rows": rows,
    }
    if any("jax_demo_cosine" in row for row in rows):
        summary["jax_demo_cosine_mean"] = float(np.mean([row["jax_demo_cosine"] for row in rows]))
        summary["torch_demo_cosine_mean"] = float(np.mean([row["torch_demo_cosine"] for row in rows]))
    print(
        f"summary  n={summary['num']}  cosine_mean={summary['cosine_mean']:.6f}  "
        f"cosine_min={summary['cosine_min']:.6f}  max_abs_mean={summary['max_abs_mean']:.5f}  "
        f"jax_p50={summary['jax_ms_p50']:.0f} ms  torch_p50={summary['torch_ms_p50']:.0f} ms",
        flush=True,
    )
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(summary, indent=2))
        print("wrote", args.out)
    if not np.isfinite(summary["cosine_min"]) or summary["cosine_min"] < args.min_cosine:
        raise SystemExit(f"cosine_min {summary['cosine_min']:.6f} < {args.min_cosine}")


if __name__ == "__main__":
    main()
