from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import pickle
import sys
from typing import Any

import tyro

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rlt_online_rl.replay import ReplayManager
from rlt_online_rl.replay import build_chunk_transitions_from_episode


@dataclasses.dataclass
class Args:
    input_path: str
    output_journal_path: str
    capacity: int = 200_000
    chunk_len: int = 10
    stride: int = 2
    overwrite: bool = False


def _load_episodes(path: str) -> list[Any]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "episodes" in payload:
        return list(payload["episodes"])
    if isinstance(payload, list):
        return payload
    raise ValueError("Unsupported replay input format. Expected a pickled list or {'episodes': ...}.")


def main(args: Args) -> None:
    if args.overwrite and os.path.exists(args.output_journal_path):
        os.remove(args.output_journal_path)

    manager = ReplayManager(args.capacity, journal_path=args.output_journal_path, seed=0)
    episodes = _load_episodes(args.input_path)
    for episode in episodes:
        steps = episode["steps"] if isinstance(episode, dict) and "steps" in episode else episode
        transitions = build_chunk_transitions_from_episode(steps, chunk_len=args.chunk_len, stride=args.stride)
        manager.add_transitions(transitions)

    print(manager.stats())


if __name__ == "__main__":
    main(tyro.cli(Args))
