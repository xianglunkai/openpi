"""LeRobot dataset recorder service (run in openpi venv, separate from robot client).

Receives HIL rollout frames from ``RuntimeRTCHil`` over TCP + msgpack and writes
LeRobot v2.1 datasets locally.

Example:
    # Terminal 1 (openpi env)
    export HF_LEROBOT_HOME=/data/huggingface/lerobot
    python -m examples.mobile_aloha_AgileX.lerobot_recorder_service --host 0.0.0.0 --port 8765

    # Terminal 2 (mobile_aloha_AgileX venv)
    python -m examples.mobile_aloha_AgileX.main_rtc_hil \\
        --lerobot-storage.remote-endpoint tcp://127.0.0.1:8765
"""

from __future__ import annotations

import dataclasses
import logging
import shutil
import socket
from pathlib import Path
from typing import Any, Optional

import tyro

from openpi_client import msgpack_numpy
from openpi_client.runtime.lerobot_remote import (
    LeRobotRemoteRecorderClient,
    build_lerobot_features,
    build_lerobot_frame,
    recv_msgpack_message,
)
from openpi_client.runtime.runtime_rtc_hil import EpisodeEndStatus, LeRobotStorageConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8765
    # Override dataset root; defaults to HF_LEROBOT_HOME from environment.
    root: Optional[str] = None


class LeRobotRecorderSession:
    def __init__(self, conn: socket.socket) -> None:
        self._conn = conn
        self._packer = msgpack_numpy.Packer()
        self._cfg: Optional[LeRobotStorageConfig] = None
        self._fps = 30
        self._dataset = None

    def run(self) -> None:
        while True:
            message = recv_msgpack_message(self._conn)
            msg_type = message.get("msg")
            if msg_type == "init":
                self._handle_init(message)
            elif msg_type == "frame":
                self._handle_frame(message)
            elif msg_type == "episode_end":
                self._handle_episode_end(message)
            elif msg_type == "close":
                self._close_dataset()
                return
            else:
                logger.warning("Unknown recorder message type: %s", msg_type)

    def _handle_init(self, message: dict[str, Any]) -> None:
        config_dict = dict(message.get("config") or {})
        if self._cfg is not None and self._cfg.root is None and config_dict.get("root") is None:
            config_dict = dict(config_dict)
        self._cfg = LeRobotStorageConfig(**config_dict)
        self._fps = int(message.get("fps") or self._cfg.fps or 30)
        self._close_dataset()
        logger.info(
            "Recorder session init: repo_id=%s root=%s fps=%s",
            self._cfg.repo_id,
            self._cfg.root,
            self._fps,
        )

    def _dataset_root(self) -> Path:
        assert self._cfg is not None
        if self._cfg.root is not None:
            return Path(self._cfg.root)
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME  # type: ignore[import-not-found]

        return Path(HF_LEROBOT_HOME) / self._cfg.repo_id

    def _ensure_dataset(self, observation: dict, action: dict) -> None:
        if self._dataset is not None:
            return
        assert self._cfg is not None
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]

        features = build_lerobot_features(observation, action, self._cfg)
        if self._cfg.clear_existing:
            candidate = self._dataset_root()
            if candidate.exists():
                shutil.rmtree(candidate)
        self._dataset = LeRobotDataset.create(
            repo_id=self._cfg.repo_id,
            fps=self._cfg.fps if self._cfg.fps is not None else self._fps,
            root=self._cfg.root,
            robot_type=self._cfg.robot_type,
            features=features,
            use_videos=self._cfg.use_videos,
            tolerance_s=self._cfg.tolerance_s,
            image_writer_processes=self._cfg.image_writer_processes,
            image_writer_threads=self._cfg.image_writer_threads,
            video_backend=self._cfg.video_backend,
        )
        logger.info("LeRobot dataset created at %s", self._dataset.root)

    def _handle_frame(self, message: dict[str, Any]) -> None:
        if self._cfg is None:
            raise RuntimeError("Received frame before init")
        observation = message["observation"]
        action = message["action"]
        task = str(message.get("task") or self._cfg.task)
        self._ensure_dataset(observation, action)
        assert self._dataset is not None
        frame = build_lerobot_frame(observation, action, task)
        self._dataset.add_frame(frame)

    def _handle_episode_end(self, message: dict[str, Any]) -> None:
        if self._dataset is None:
            return
        status = EpisodeEndStatus(str(message.get("status", EpisodeEndStatus.END.value)))
        episode_idx = int(message.get("episode_idx", -1))
        try:
            if status == EpisodeEndStatus.RERECORD:
                self._dataset.clear_episode_buffer()
                logger.info("Episode %s buffer cleared (rerecord).", episode_idx)
                return
            if self._dataset.episode_buffer is None or self._dataset.episode_buffer.get("size", 0) == 0:
                logger.warning("Episode %s buffer empty; skipping save_episode().", episode_idx)
                self._dataset.clear_episode_buffer()
                return
            self._dataset.save_episode()
            logger.info(
                "Episode %s saved to %s (status=%s)",
                episode_idx,
                self._dataset.root,
                status.value,
            )
        except Exception:
            logger.exception("Failed to finalize episode %s", episode_idx)

    def _close_dataset(self) -> None:
        if self._dataset is None:
            return
        try:
            stop_image_writer = getattr(self._dataset, "stop_image_writer", None)
            if callable(stop_image_writer):
                stop_image_writer()
            if self._cfg is not None and self._cfg.push_to_hub:
                self._dataset.push_to_hub(
                    tags=self._cfg.tags,
                    private=self._cfg.private,
                )
        except Exception:
            logger.exception("Failed to close LeRobot dataset")
        finally:
            self._dataset = None


def serve(settings: Args) -> None:
    if settings.root is not None:
        logger.info("Using dataset root override: %s", settings.root)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((settings.host, settings.port))
    server.listen(1)
    logger.info("LeRobot recorder listening on %s:%s", settings.host, settings.port)

    while True:
        conn, addr = server.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info("Client connected from %s", addr)
        session = LeRobotRecorderSession(conn)
        try:
            session.run()
        except ConnectionError:
            logger.info("Client disconnected")
        except Exception:
            logger.exception("Recorder session failed")
        finally:
            session._close_dataset()
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    settings = tyro.cli(Args)
    serve(settings)
