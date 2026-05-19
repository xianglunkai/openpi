#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.client
import os
from pathlib import Path
import sys
import time
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PIKA_SYNC_ROS = REPO_ROOT / "train_deploy_alignment" / "pika_sync_ros.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tasks" / "agilex_ethernet" / "online_rl.yaml"

from rlt_online_rl.config import load_system_config_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch eval rollout with actor_service only.")
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    args, remaining = parser.parse_known_args()
    args.remaining = remaining
    return args


def _resolve_config_path(run_dir: str | None, config: str | None) -> str:
    if config is not None:
        return config
    if run_dir is None:
        return str(DEFAULT_CONFIG)
    resolved = Path(run_dir) / "checkpoints" / "online_rl_config.yaml"
    if resolved.exists():
        return str(resolved)
    return str(DEFAULT_CONFIG)


def _peek_option(argv: list[str], flag: str) -> str | None:
    for idx, token in enumerate(argv):
        if token == flag:
            if idx + 1 >= len(argv):
                raise ValueError(f"{flag} requires a value.")
            return argv[idx + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _wait_for_http(url: str, *, timeout_sec: float = 30.0) -> None:
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        req = urllib_request.Request(url, method="GET")
        try:
            with urllib_request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (urllib_error.URLError, ConnectionError, http.client.HTTPException) as exc:
            last_error = exc
        time.sleep(0.2)
    raise RuntimeError(f"Service not ready at {url}") from last_error


def main() -> None:
    args = _parse_args()
    config_path = _resolve_config_path(args.run_dir, args.config)
    system = load_system_config_yaml(config_path)
    actor_service_url = _peek_option(args.remaining, "--actor_service_url") or system.env_driver.actor_service_url

    print(f"[launch_actor_eval] using config {config_path}", flush=True)
    print(f"[launch_actor_eval] waiting for actor_service at {actor_service_url}", flush=True)
    _wait_for_http(f"{actor_service_url.rstrip('/')}/version")
    print("[launch_actor_eval] actor_service ready, starting pika_sync_ros eval rollout.", flush=True)

    argv = [
        sys.executable,
        str(PIKA_SYNC_ROS),
        "--config",
        config_path,
        "--eval_actor_only",
        *args.remaining,
    ]
    os.chdir(REPO_ROOT)
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
