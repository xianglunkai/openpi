"""Unit tests for paper-aligned warmup journal transition construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "offline"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_warmup_journal_from_lerobot import build_overlap_frame_indices  # noqa: E402
from build_warmup_journal_from_lerobot import build_reward_seq  # noqa: E402
from build_warmup_journal_from_lerobot import build_transitions_for_critical_segment  # noqa: E402
from build_warmup_journal_from_lerobot import count_paper_transitions  # noqa: E402


def _fake_features(anchor_frames: list[int], *, chunk_len: int, z_dim: int = 4, action_dim: int = 3):
    features = []
    for frame in anchor_frames:
        ref = np.full((chunk_len, action_dim), float(frame), dtype=np.float32)
        features.append(
            {
                "z_rl": np.full((z_dim,), float(frame), dtype=np.float32),
                "proprio": np.full((action_dim,), float(frame), dtype=np.float32),
                "ref_chunk": ref,
                "action_chunk": ref + 0.5,  # demo ≠ VLA ref
            }
        )
    return features


def test_build_reward_seq_terminal_success() -> None:
    rewards = build_reward_seq(10, is_terminal_chunk=True, episode_success=True)
    assert rewards.shape == (10,)
    assert float(rewards.sum()) == 1.0
    assert float(rewards[-1]) == 1.0


def test_build_reward_seq_non_terminal_or_failure() -> None:
    assert float(build_reward_seq(10, is_terminal_chunk=False, episode_success=True).sum()) == 0.0
    assert float(build_reward_seq(10, is_terminal_chunk=True, episode_success=False).sum()) == 0.0


def test_transitions_use_t_plus_c_and_terminal_reward() -> None:
    chunk_len = 10
    stride = 2
    start, end = 0, 30
    anchors = build_overlap_frame_indices(start, end + 1, chunk_len, stride)
    features = _fake_features(anchors, chunk_len=chunk_len)

    transitions = build_transitions_for_critical_segment(
        episode_id=7,
        anchor_frames=anchors,
        features=features,
        chunk_len=chunk_len,
        stride=stride,
        segment_last_frame=end,
        episode_success=True,
    )

    assert len(transitions) == count_paper_transitions(
        anchors, segment_last_frame=end, chunk_length=chunk_len
    )
    assert len(transitions) >= 1

    # Each transition bootstraps exactly C steps ahead.
    for tr in transitions:
        assert int(tr.actual_steps) == chunk_len
        assert float(tr.z_rl[0]) + chunk_len == float(tr.next_z_rl[0])
        # Demo action must stay distinct from VLA ref (not copied).
        assert not np.allclose(tr.action_chunk, tr.ref_chunk)

    terminal = [tr for tr in transitions if tr.done]
    assert len(terminal) == 1
    assert int(terminal[0].success) == 1
    assert float(terminal[0].rewards[-1]) == 1.0
    assert float(np.sum(terminal[0].rewards)) == 1.0
    assert all(float(np.sum(tr.rewards)) == 0.0 for tr in transitions if not tr.done)


def test_failed_segment_has_zero_terminal_reward() -> None:
    chunk_len = 10
    stride = 2
    start, end = 0, 30
    anchors = build_overlap_frame_indices(start, end + 1, chunk_len, stride)
    features = _fake_features(anchors, chunk_len=chunk_len)
    transitions = build_transitions_for_critical_segment(
        episode_id=1,
        anchor_frames=anchors,
        features=features,
        chunk_len=chunk_len,
        stride=stride,
        segment_last_frame=end,
        episode_success=False,
    )
    assert any(tr.done for tr in transitions)
    assert all(float(np.sum(tr.rewards)) == 0.0 for tr in transitions)
    assert all(int(tr.success) == 0 for tr in transitions)
