#!/usr/bin/env python3
"""Serve pi05_cobot_screw_sorting_single through FlashRT on port 8000.

Run this with ~/flashrt-venv, not with `uv run`. That venv has the Jetson
PyTorch wheel and flash_rt. Installing this repo into it (`pip install -e .`)
would pull torch==2.7.1 and jax[cuda12] from pyproject.toml and replace that
wheel.

This file does not import openpi.training or JAX. Extra packages, installed
into the flashrt venv only:

    pip install websockets msgpack osqp scipy

The robot client is unchanged: {state, images} in, actions (50, 7) out.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import json
import logging
import os
import pathlib
import socket
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "openpi-client" / "src"))

from openpi_client import msgpack_numpy  # noqa: E402
import websockets.asyncio.server as _server  # noqa: E402
import websockets.frames  # noqa: E402

PROMPT = "Please sort and return the silver screws in the grey box to their proper places."
ASSET_ID = "screw_sorting_single"
logger = logging.getLogger(__name__)


def _hwc_uint8(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return np.ascontiguousarray(image)


def _quantile_norm(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _quantile_unnorm(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    width = q01.shape[-1]
    head = (x[..., :width] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    if x.shape[-1] == width:
        return head
    return np.concatenate([head, x[..., width:]], axis=-1)


def _smooth_qp(actions: np.ndarray) -> np.ndarray:
    """Same weights as serve_policy: dt=0.033, w_data=1, w_acc=100, w_jerk=100."""
    import osqp
    from scipy import sparse

    actions = np.asarray(actions, dtype=np.float32)
    squeezed = actions.ndim == 2
    if squeezed:
        actions = actions[None, ...]
    _, horizon, dims = actions.shape
    eye = sparse.eye(horizon, format="csc")
    d2 = sparse.diags([1, -2, 1], [0, 1, 2], shape=(horizon - 2, horizon), format="csc")
    d3 = sparse.diags([1, -3, 3, -1], [0, 1, 2, 3], shape=(horizon - 3, horizon), format="csc")
    hessian = (eye + 100.0 * (d2.T @ d2) + 100.0 * (d3.T @ d3) + 1e-6 * eye).astype(np.float64)
    empty = sparse.csc_matrix((0, horizon), dtype=np.float64)
    out = np.empty_like(actions)
    for channel in range(dims):
        target = actions[0, :, channel].astype(np.float64)
        try:
            prob = osqp.OSQP()
            prob.setup(
                P=hessian,
                q=(-target).astype(np.float64),
                A=empty,
                l=np.zeros(0),
                u=np.zeros(0),
                verbose=False,
                polish=False,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=100,
            )
            result = prob.solve()
            if result.info.status in ("solved", "solved_inaccurate", "solved_relaxed") and result.x is not None:
                out[0, :, channel] = result.x.astype(np.float32)
            else:
                out[0, :, channel] = target.astype(np.float32)
        except Exception:
            out[0, :, channel] = target.astype(np.float32)
    return out[0] if squeezed else out


class FlashRTScrewPolicy:
    def __init__(self, checkpoint: str, *, num_steps: int) -> None:
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 7):
                os.environ["FVK_PI05_RTX_FORCE_INT8"] = "1"
        except ImportError:
            pass

        import flash_rt

        self._model = flash_rt.load_model(
            checkpoint=checkpoint,
            config="pi05",
            framework="torch",
            num_views=3,
            num_steps=num_steps,
            autotune=3,
            use_fp8=True,
            state_prompt_mode="fixed",
            state_prompt_fixed_max_len=200,
        )
        stats_path = pathlib.Path(checkpoint) / "assets" / ASSET_ID / "norm_stats.json"
        stats = json.loads(stats_path.read_text())["norm_stats"]
        self._state_q01 = np.asarray(stats["state"]["q01"], dtype=np.float32)
        self._state_q99 = np.asarray(stats["state"]["q99"], dtype=np.float32)
        self._action_q01 = np.asarray(stats["actions"]["q01"], dtype=np.float32)
        self._action_q99 = np.asarray(stats["actions"]["q99"], dtype=np.float32)

    def infer(self, obs: dict, use_rtc: bool = False, noise=None) -> dict:
        del noise
        if use_rtc:
            raise NotImplementedError("this FlashRT server does not implement RTC")
        raw_state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)[:7]
        images = obs["images"]
        base = _hwc_uint8(images["cam_high"])
        wrist = _hwc_uint8(images["cam_right_wrist"])
        left = np.zeros_like(base)
        actions = self._model.predict(
            images=[base, left, wrist],
            prompt=PROMPT,
            state=_quantile_norm(raw_state, self._state_q01, self._state_q99),
        )
        actions = np.asarray(actions["actions"] if isinstance(actions, dict) else actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[0] != 50:
            raise RuntimeError(
                f"FlashRT returned actions shape {actions.shape}, expected time dimension 50. "
                "A leading 10 means a LIBERO head was loaded. Do not command the arm."
            )
        actions = _quantile_unnorm(actions, self._action_q01, self._action_q99)
        actions = np.array(actions[..., :7], dtype=np.float32, copy=True)
        actions[:, :6] += raw_state[:6]
        return {"actions": _smooth_qp(actions)}


def _health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def _handler(websocket, policy) -> None:
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack({}))
    while True:
        try:
            data = msgpack_numpy.unpackb(await websocket.recv())
            if "obs" in data:
                obs = data["obs"]
                use_rtc = data.get("use_rtc", False)
            else:
                obs = data
                use_rtc = False
            started = time.monotonic()
            action = policy.infer(obs, use_rtc=use_rtc)
            action["server_timing"] = {"infer_ms": (time.monotonic() - started) * 1000.0}
            await websocket.send(packer.pack(action))
        except websockets.ConnectionClosed:
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Internal server error")
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/data/checkpoints/pi05_cobot_screw_sorting_single_pytorch")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_steps", type=int, default=10)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    policy = FlashRTScrewPolicy(args.checkpoint, num_steps=args.num_steps)
    logging.info("FlashRT screw policy on %s port %s", socket.gethostname(), args.port)

    async def run() -> None:
        async with _server.serve(
            lambda ws: _handler(ws, policy),
            args.host,
            args.port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ):
            await asyncio.Future()

    asyncio.run(run())


if __name__ == "__main__":
    main()
