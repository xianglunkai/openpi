#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
from pathlib import Path
import pickle

import numpy as np

"""
Inspect replay journal contents.

Examples:
python3 scripts/tools/inspect_replay_journal.py runs/agilex_ethernet
python3 scripts/tools/inspect_replay_journal.py runs/agilex_ethernet/offline_train_bcq
python3 scripts/tools/inspect_replay_journal.py runs/agilex_ethernet_4.23_morning/replay/replay_journal.pkl
"""

SOURCE_NAMES = {
    0: "BASE",
    1: "RL",
    2: "HUMAN",
    3: "MIXED",
}
DEFAULT_COLLECTION_PHASE = "unknown"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a replay journal. The input can be a replay file, replay directory, "
            "online run_dir, or offline bundle directory."
        )
    )
    parser.add_argument("input_path", type=Path, help="Replay file, replay dir, run_dir, or offline dir.")
    return parser.parse_args()


def _scalar_int(value: object) -> int:
    return int(np.asarray(value).item())


def _scalar_bool(value: object) -> bool:
    return bool(np.asarray(value).item())


def _collection_phase(record: dict[str, object]) -> str:
    return str(record.get("collection_phase", DEFAULT_COLLECTION_PHASE))


def _resolve_journal_path(input_path: Path) -> Path:
    path = input_path.expanduser().resolve()
    if path.is_file():
        return path
    if not path.is_dir():
        raise FileNotFoundError(f"Input not found: {path}")

    candidates = [
        path / "replay_journal_no_rl.pkl",
        path / "replay_journal.pkl",
        path / "replay" / "replay_journal_no_rl.pkl",
        path / "replay" / "replay_journal.pkl",
    ]
    parent_replay_dir = path.parent / "replay"
    if parent_replay_dir.is_dir():
        candidates.extend(
            [
                parent_replay_dir / "replay_journal_no_rl.pkl",
                parent_replay_dir / "replay_journal.pkl",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find replay journal under {path}")


def _load_records(journal_path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with journal_path.open("rb") as f:
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break
            if isinstance(item, dict):
                records.append(item)
    return records


def main() -> int:
    args = _parse_args()
    journal_path = _resolve_journal_path(args.input_path)
    records = _load_records(journal_path)

    print(f"journal: {journal_path}")
    print(f"total transitions: {len(records)}")
    if not records:
        return 0

    source_counter: collections.Counter[int] = collections.Counter()
    phase_counter: collections.Counter[str] = collections.Counter()
    phase_source_counter: collections.Counter[tuple[str, int]] = collections.Counter()
    success_counter: collections.Counter[int] = collections.Counter()
    done_counter: collections.Counter[bool] = collections.Counter()
    intervention_counter: collections.Counter[bool] = collections.Counter()
    per_episode_source: dict[int, collections.Counter[int]] = collections.defaultdict(collections.Counter)
    per_episode_phase: dict[int, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    episodes_with_reward: dict[int, bool] = collections.defaultdict(bool)
    success_without_reward = 0

    total_reward_sum = 0.0
    total_reward_entries = 0
    positive_reward_entries = 0
    negative_reward_entries = 0
    nonzero_reward_entries = 0
    transitions_with_reward = 0
    transitions_with_positive_reward = 0

    for record in records:
        episode_id = _scalar_int(record["episode_id"])
        source = _scalar_int(record["source"])
        phase = _collection_phase(record)
        success = _scalar_int(record["success"])
        done = _scalar_bool(record["done"])
        intervention = _scalar_bool(record["intervention_flag"])
        rewards = np.asarray(record["rewards"], dtype=np.float64).reshape(-1)

        reward_sum = float(rewards.sum())
        has_reward = bool(np.any(rewards != 0.0))
        has_positive_reward = bool(np.any(rewards > 0.0))

        source_counter[source] += 1
        phase_counter[phase] += 1
        phase_source_counter[(phase, source)] += 1
        success_counter[success] += 1
        done_counter[done] += 1
        intervention_counter[intervention] += 1
        per_episode_source[episode_id][source] += 1
        per_episode_phase[episode_id][phase] += 1

        total_reward_sum += reward_sum
        total_reward_entries += int(rewards.size)
        positive_reward_entries += int(np.count_nonzero(rewards > 0.0))
        negative_reward_entries += int(np.count_nonzero(rewards < 0.0))
        nonzero_reward_entries += int(np.count_nonzero(rewards != 0.0))
        transitions_with_reward += int(has_reward)
        transitions_with_positive_reward += int(has_positive_reward)
        episodes_with_reward[episode_id] = episodes_with_reward[episode_id] or has_reward
        if success > 0 and not has_reward:
            success_without_reward += 1

    print("\nsource counts:")
    for source_id in sorted(SOURCE_NAMES):
        print(f"  {SOURCE_NAMES[source_id]:>5} ({source_id}): {source_counter[source_id]}")

    print("\ncollection phase counts:")
    for phase in sorted(phase_counter):
        print(f"  {phase}: {phase_counter[phase]}")

    print("\ncollection phase x source counts:")
    for phase in sorted(phase_counter):
        parts = [
            f"{SOURCE_NAMES[source_id]}={phase_source_counter[(phase, source_id)]}"
            for source_id in sorted(SOURCE_NAMES)
            if phase_source_counter[(phase, source_id)]
        ]
        print(f"  {phase}: " + ", ".join(parts))

    print("\nsuccess counts:")
    for key in sorted(success_counter):
        print(f"  success={key}: {success_counter[key]}")

    print("\ndone counts:")
    print(f"  done=False: {done_counter[False]}")
    print(f"  done=True : {done_counter[True]}")

    print("\nintervention counts:")
    print(f"  intervention=False: {intervention_counter[False]}")
    print(f"  intervention=True : {intervention_counter[True]}")

    print("\nreward summary:")
    print(f"  total_reward_sum: {total_reward_sum:.6f}")
    print(f"  total_reward_entries: {total_reward_entries}")
    print(f"  nonzero_reward_entries: {nonzero_reward_entries}")
    print(f"  positive_reward_entries: {positive_reward_entries}")
    print(f"  negative_reward_entries: {negative_reward_entries}")
    print(f"  transitions_with_reward: {transitions_with_reward}")
    print(f"  transitions_with_positive_reward: {transitions_with_positive_reward}")
    print(f"  episodes_with_reward: {sum(episodes_with_reward.values())}")
    print(f"  success_transitions_without_reward: {success_without_reward}")

    print("\nper-episode source counts:")
    for episode_id in sorted(per_episode_source):
        counter = per_episode_source[episode_id]
        parts = [
            f"{SOURCE_NAMES[source_id]}={counter[source_id]}"
            for source_id in sorted(SOURCE_NAMES)
            if counter[source_id]
        ]
        phase_parts = [
            f"{phase}={per_episode_phase[episode_id][phase]}" for phase in sorted(per_episode_phase[episode_id])
        ]
        reward_tag = "reward=yes" if episodes_with_reward[episode_id] else "reward=no"
        print(f"  episode {episode_id}: " + ", ".join(phase_parts + parts + [reward_tag]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
