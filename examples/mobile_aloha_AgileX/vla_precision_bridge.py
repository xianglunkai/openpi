"""HTTP bridge exposing Mobile ALOHA real_env to VLA-Precision over localhost.

Run this inside the openpi mobile_aloha environment (ROS already available).
VLA-Precision only talks to this process via HTTP; it does not import rospy.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import tyro
from flask import Flask, jsonify, request

from examples.mobile_aloha_AgileX import real_env as _real_env

LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5001
    use_single_arm: bool = True
    single_arm: str = "right"
    control_freq_hz: float = 30.0


def _normalize_images(raw_images: dict[str, Any]) -> dict[str, np.ndarray]:
    images: dict[str, np.ndarray] = {}
    for name, image in (raw_images or {}).items():
        if image is None or "_depth" in name:
            continue
        array = np.asarray(image)
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
        images[name] = np.asarray(array, dtype=np.uint8)
    return images


def _encode_image(image: np.ndarray) -> str:
    return base64.b64encode(np.asarray(image, dtype=np.uint8).tobytes()).decode("ascii")


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    _real_env.SINGLE_ARM_NAME = args.single_arm
    LOGGER.info(
        "Starting VLA-Precision Mobile ALOHA bridge single_arm=%s bind=%s:%d",
        args.single_arm if args.use_single_arm else "dual",
        args.host,
        args.port,
    )
    env = _real_env.make_real_env(
        init_node=True,
        control_freq_hz=args.control_freq_hz,
        use_single_arm=args.use_single_arm,
    )
    latest: dict[str, Any] = {"qpos": np.zeros(7, dtype=np.float32), "images": {}}

    def refresh() -> dict[str, Any]:
        obs = env.get_observation()
        latest["qpos"] = np.asarray(obs["qpos"], dtype=np.float32).reshape(-1)
        latest["images"] = _normalize_images(obs.get("images") or {})
        return latest

    refresh()
    app = Flask("vla-precision-mobile-aloha")

    @app.get("/state")
    def state():
        refresh()
        return jsonify({"qpos": latest["qpos"].tolist()})

    @app.get("/camera/<camera_name>")
    def camera(camera_name: str):
        if camera_name not in latest["images"]:
            refresh()
        images = latest["images"]
        if camera_name not in images:
            return jsonify({"error": f"unknown camera {camera_name}", "available": list(images)}), 404
        image = images[camera_name]
        return jsonify(
            {
                "dtype": "uint8",
                "shape": list(image.shape),
                "image_b64": _encode_image(image),
            }
        )

    @app.post("/action_chunk")
    def action_chunk():
        body = request.get_json(force=True, silent=True) or {}
        actions = np.asarray(body.get("actions"), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        executed = []
        for step in actions:
            env.step(step)
            executed.append(step.tolist())
        refresh()
        return jsonify({"executed_actions": executed, "qpos": latest["qpos"].tolist()})

    @app.post("/reset")
    def reset():
        body = request.get_json(force=True, silent=True) or {}
        env.reset(fake=bool(body.get("fake", False)))
        refresh()
        return jsonify({"qpos": latest["qpos"].tolist()})

    @app.post("/request")
    def named_request():
        body = request.get_json(force=True, silent=True) or {}
        return jsonify({"ok": True, "name": body.get("name"), "enabled": body.get("enabled")})

    @app.post("/close")
    def close():
        return jsonify({"ok": True})

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
