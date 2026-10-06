#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from rlt_online_rl.config import load_system_config_yaml
from rlt_online_rl.runtime_logging import infer_run_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_ONLINE_RL = REPO_ROOT / "scripts" / "run_online_rl.py"
RUN_WANDB_MONITOR = REPO_ROOT / "scripts" / "stream_learner_metrics_to_wandb.py"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "tasks" / "agilex_ethernet" / "online_rl.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Machine B online RL services.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def _resolve_config_path(config: str) -> str:
    return str(Path(config).expanduser().resolve())


def _spawn(role: str, config: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(RUN_ONLINE_RL),
            "--config",
            config,
            "--system.role",
            role,
        ],
        text=True,
        cwd=str(REPO_ROOT),
    )


def _spawn_wandb_monitor(
    *,
    run_dir: Path,
    project: str,
    mode: str,
    run_name: str,
    wandb_dir: str,
    entity: str | None,
) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.setdefault("WANDB_DIR", str(Path(wandb_dir).expanduser().resolve()))
    command = [
        sys.executable,
        str(RUN_WANDB_MONITOR),
        "--run_dir",
        str(run_dir),
        "--project",
        project,
        "--mode",
        mode,
        "--run_name",
        run_name,
        "--wandb_dir",
        wandb_dir,
    ]
    if entity:
        command.extend(["--entity", entity])
    return subprocess.Popen(command, text=True, cwd=str(REPO_ROOT), env=env)


def _terminate(process: subprocess.Popen[str], grace_sec: float = 5.0) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    deadline = time.time() + grace_sec
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    process.kill()
    process.wait(timeout=1.0)


def main() -> None:
    args = _parse_args()
    config_path = _resolve_config_path(args.config)
    system = load_system_config_yaml(config_path)
    run_dir = infer_run_dir(system)
    processes: list[tuple[str, subprocess.Popen[str]]] = []
    try:
        for role in ("replay_manager", "learner_service", "actor_service"):
            process = _spawn(role, config_path)
            processes.append((role, process))
            print(f"[launch_machine_b] started {role} pid={process.pid}", flush=True)
            time.sleep(0.5)

        if system.monitoring.enable_wandb:
            monitor = _spawn_wandb_monitor(
                run_dir=run_dir,
                project=system.monitoring.wandb_project,
                mode=system.monitoring.wandb_mode,
                run_name=system.monitoring.wandb_run_name,
                wandb_dir=system.monitoring.wandb_dir,
                entity=system.monitoring.wandb_entity,
            )
            processes.append(("wandb_monitor", monitor))
            print(f"[launch_machine_b] started wandb_monitor pid={monitor.pid}", flush=True)

        while True:
            next_processes: list[tuple[str, subprocess.Popen[str]]] = []
            for role, process in processes:
                exitcode = process.poll()
                if exitcode is None:
                    next_processes.append((role, process))
                    continue
                if role == "wandb_monitor":
                    print(
                        f"[launch_machine_b] warning: wandb_monitor exited with code {exitcode}; continuing without W&B streaming.",
                        flush=True,
                    )
                    continue
                raise RuntimeError(f"{role} exited unexpectedly with code {exitcode}")
            processes = next_processes
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[launch_machine_b] received KeyboardInterrupt, shutting down.", flush=True)
    finally:
        for _, process in reversed(processes):
            _terminate(process)
        print("[launch_machine_b] shutdown complete.", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    main()
