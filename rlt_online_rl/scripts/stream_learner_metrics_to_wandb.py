#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time
from typing import Any

POLL_INTERVAL_SEC = 1.0
EMA_ALPHA = 0.1
EMA_KEYS = ("critic_loss", "actor_loss", "target_q_mean", "bc_penalty")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream learner metrics jsonl to W&B.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--mode", choices=("offline", "online"), required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--wandb_dir", required=True)
    parser.add_argument("--entity", default=None)
    return parser.parse_args()


def _setup_logging(run_dir: Path) -> Path:
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "wandb_monitor.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        force=True,
    )
    return log_path


def _coerce_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_log_payload(record: dict[str, Any], ema_state: dict[str, float]) -> tuple[int | None, dict[str, float]]:
    step = record.get("global_step")
    resolved_step = int(step) if isinstance(step, (int, float)) else None
    payload: dict[str, float] = {}
    for key, value in record.items():
        scalar = _coerce_scalar(value)
        if scalar is None:
            continue
        payload[f"learner/{key}"] = scalar
        if key not in EMA_KEYS:
            continue
        previous = ema_state.get(key)
        ema = scalar if previous is None else EMA_ALPHA * scalar + (1.0 - EMA_ALPHA) * previous
        ema_state[key] = ema
        payload[f"learner_ema/{key}"] = ema
    return resolved_step, payload


def _wait_for_metrics_path(path: Path) -> None:
    while not path.exists():
        time.sleep(POLL_INTERVAL_SEC)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    log_path = _setup_logging(run_dir)
    logger = logging.getLogger("wandb_monitor")
    metrics_path = run_dir / "metrics" / "learner_metrics.jsonl"
    wandb_dir = Path(args.wandb_dir).expanduser().resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("wandb is required for enable_wandb=true. Install it before launching Machine B.") from exc

    logger.info(
        "wandb monitor starting run_dir=%s mode=%s project=%s log=%s", run_dir, args.mode, args.project, log_path
    )
    _wait_for_metrics_path(metrics_path)
    logger.info("Watching learner metrics at %s", metrics_path)

    wandb_run = wandb.init(
        project=args.project,
        entity=args.entity,
        mode=args.mode,
        name=args.run_name,
        dir=str(wandb_dir),
        config={"run_dir": str(run_dir)},
    )
    offset = 0
    ema_state: dict[str, float] = {}
    try:
        while True:
            file_size = metrics_path.stat().st_size
            if file_size < offset:
                offset = 0
            with open(metrics_path, encoding="utf-8") as f:
                f.seek(offset)
                lines = f.readlines()
                offset = f.tell()
            if not lines:
                time.sleep(POLL_INTERVAL_SEC)
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                step, payload = _build_log_payload(record, ema_state)
                if not payload:
                    continue
                if step is None:
                    wandb_run.log(payload)
                else:
                    wandb_run.log(payload, step=step)
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        logger.info("wandb monitor interrupted, shutting down.")
    finally:
        wandb_run.finish()


if __name__ == "__main__":
    main()
