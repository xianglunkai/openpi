"""Remote LeRobot dataset recording over TCP + msgpack (no lerobot on robot client)."""

from __future__ import annotations

import dataclasses
import logging
import socket
import struct
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np

from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)

_MSG_HEADER = struct.Struct("!I")


def to_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    return arr.copy()


def lerobot_vector_names(dim: int, names: Optional[list[str]]) -> list[str]:
    if names is not None and len(names) == dim:
        return list(names)
    return [f"dim_{i}" for i in range(dim)]


def lerobot_image_names(shape: tuple[int, ...]) -> list[str]:
    if len(shape) == 3:
        return ["channels", "height", "width"]
    return [f"dim_{i}" for i in range(len(shape))]


def build_lerobot_features(observation: dict, action: dict, cfg: Any) -> dict:
    state = to_numpy(observation.get("state", []))
    action_vec = to_numpy(action.get("actions", []))
    state_dim = int(state.reshape(-1).shape[0])
    action_dim = int(action_vec.reshape(-1).shape[0])
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": lerobot_vector_names(state_dim, getattr(cfg, "state_names", None)),
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": lerobot_vector_names(action_dim, getattr(cfg, "action_names", None)),
        },
        "is_intervention": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_intervention"],
        },
    }
    images = observation.get("images") or {}
    visual_dtype = "video" if getattr(cfg, "use_videos", False) else getattr(cfg, "mode", "image")
    for cam_name, image in images.items():
        image_np = to_numpy(image)
        features[f"observation.images.{cam_name}"] = {
            "dtype": visual_dtype,
            "shape": tuple(image_np.shape),
            "names": lerobot_image_names(tuple(image_np.shape)),
        }
    return features


def build_lerobot_frame(observation: dict, action: dict, task: str) -> dict:
    frame = {
        "observation.state": to_numpy(observation.get("state", [])).astype(np.float32).reshape(-1),
        "action": to_numpy(action.get("actions", [])).astype(np.float32).reshape(-1),
        "task": task,
    }
    images = observation.get("images") or {}
    for cam_name, image in images.items():
        frame[f"observation.images.{cam_name}"] = to_numpy(image)
    hil_info = action.get("_hil", {})
    is_intervention = 1.0 if bool(hil_info.get("is_intervention", False)) else 0.0
    frame["is_intervention"] = np.array([is_intervention], dtype=np.float32)
    return frame


def storage_config_to_dict(cfg: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return dict(cfg)


def parse_tcp_endpoint(endpoint: str) -> tuple[str, int]:
    parsed = urlparse(endpoint if "://" in endpoint else f"tcp://{endpoint}")
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8765
    return host, port


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    chunks: list[bytes] = []
    remaining = nbytes
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Remote LeRobot recorder closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_msgpack_message(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, _MSG_HEADER.size)
    (length,) = _MSG_HEADER.unpack(header)
    payload = recv_exact(sock, length)
    return msgpack_numpy.unpackb(payload)


def send_msgpack_message(sock: socket.socket, packer: msgpack_numpy.Packer, message: dict[str, Any]) -> None:
    payload = packer.pack(message)
    sock.sendall(_MSG_HEADER.pack(len(payload)) + payload)


class LeRobotRemoteRecorderClient:
    """Send HIL rollout frames to a recorder service running in the openpi environment."""

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._host, self._port = parse_tcp_endpoint(endpoint)
        self._sock: Optional[socket.socket] = None
        self._packer = msgpack_numpy.Packer()
        self._initialized = False

    @property
    def connected(self) -> bool:
        return self._sock is not None

    @property
    def initialized(self) -> bool:
        return self._initialized

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.create_connection((self._host, self._port), timeout=5.0)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        logger.info("Connected to remote LeRobot recorder at %s:%s", self._host, self._port)

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            send_msgpack_message(self._sock, self._packer, {"msg": "close"})
        except Exception:
            logger.debug("Failed to notify remote LeRobot recorder on close", exc_info=True)
        finally:
            try:
                self._sock.close()
            finally:
                self._sock = None
                self._initialized = False

    def initialize(self, *, cfg: Any, fps: float) -> None:
        self.connect()
        assert self._sock is not None
        send_msgpack_message(
            self._sock,
            self._packer,
            {
                "msg": "init",
                "config": storage_config_to_dict(cfg),
                "fps": float(fps),
            },
        )
        self._initialized = True
        logger.info("Remote LeRobot recorder initialized for repo_id=%s", getattr(cfg, "repo_id", "?"))

    def add_frame(self, *, episode_idx: int, observation: dict, action: dict, task: str) -> None:
        if self._sock is None:
            self.connect()
        assert self._sock is not None
        send_msgpack_message(
            self._sock,
            self._packer,
            {
                "msg": "frame",
                "episode_idx": int(episode_idx),
                "observation": observation,
                "action": action,
                "task": task,
            },
        )

    def finish_episode(self, *, episode_idx: int, status: str) -> None:
        if self._sock is None:
            return
        send_msgpack_message(
            self._sock,
            self._packer,
            {
                "msg": "episode_end",
                "episode_idx": int(episode_idx),
                "status": status,
            },
        )
