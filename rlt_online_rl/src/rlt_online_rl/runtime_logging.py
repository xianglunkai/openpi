from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from rlt_online_rl.config import OnlineRLSystemConfig


def infer_run_dir(system: OnlineRLSystemConfig) -> Path:
    return Path(system.learner_service.checkpoint_dir).expanduser().resolve().parent


def ensure_runtime_dirs(system: OnlineRLSystemConfig) -> dict[str, Path]:
    run_dir = infer_run_dir(system)
    log_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "log_dir": log_dir,
        "metrics_dir": metrics_dir,
    }


def log_path_for(system: OnlineRLSystemConfig, process_name: str) -> Path:
    return ensure_runtime_dirs(system)["log_dir"] / f"{process_name}.log"


def metrics_path_for(system: OnlineRLSystemConfig, filename: str) -> Path:
    return ensure_runtime_dirs(system)["metrics_dir"] / filename


def setup_process_logging(
    process_name: str,
    system: OnlineRLSystemConfig,
    *,
    console: bool = True,
    level: int = logging.INFO,
    console_level: int | None = None,
) -> Path:
    log_path = log_path_for(system, process_name)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(processName)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    root_logger.setLevel(level)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if console:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level if console_level is None else console_level)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)

    logging.captureWarnings(True)
    return log_path


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    payload.setdefault("timestamp", time.time())
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(payload), ensure_ascii=False) + "\n")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(subvalue) for key, subvalue in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
