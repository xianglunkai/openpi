#!/usr/bin/env python3
"""LeRobot critical segments → Evo v2 warmup replay journal (all-in-one).

Subcommands
-----------
init PATH         Read LeRobot and write one CSV row per episode (edit critical bounds).
template PATH     Write a single-row placeholder CSV (no dataset).
example PATH      Write a hardcoded 5-episode example CSV (no dataset).
list              Print episode lengths (helps annotation).
validate          Check a critical CSV against the LeRobot dataset.
build             Encode VLA features and write replay_journal.pkl (default).

Critical CSV columns (episode-local frame indices, both ends inclusive):

    episode_id,start_frame,end_frame,success
    0,120,280,1

Evo v2 journal semantics (build):

    - Paper-aligned chunk transitions: ``x_t -> x_{t+C}`` (not consecutive anchors)
    - ``ref_chunk`` = VLA prediction at anchor ``t`` (Machine A / pi0.5)
    - ``action_chunk`` = demo actions ``a_t:t+C-1`` from the LeRobot dataset
    - Sparse terminal reward: last step of the terminal chunk is ``1.0`` iff
      the critical segment is marked successful; zeros otherwise
    - ``source=BASE``, ``collection_phase=warmup``

Dataset I/O uses a lightweight local LeRobot v2.1 reader (meta JSONL + parquet
via pyarrow + opencv). It does **not** import the HuggingFace ``lerobot``
package. For ``build``, install: ``pip install 'rlt-online-rl[offline]'``.

Examples
--------
    # 1) Init CSV from LeRobot (recommended), edit critical bounds, validate
    python scripts/offline/build_warmup_journal_from_lerobot.py init critical_segments.csv
    python scripts/offline/build_warmup_journal_from_lerobot.py validate --critical-path critical_segments.csv

    # 2) Build journal (Machine A must be running)
    python scripts/offline/build_warmup_journal_from_lerobot.py build \\
      --critical-path critical_segments.csv \\
      --output-journal runs/screw_sorting/replay/replay_journal.pkl \\
      --machine-a-ws-url ws://127.0.0.1:8000 \\
      --chunk-len 10 --stride 2 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import pickle
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from openpi_client import image_tools  # noqa: E402

from rlt_online_rl.config import RLTOnlineRLConfig  # noqa: E402
from rlt_online_rl.inference import MachineAFeatureClient  # noqa: E402
from rlt_online_rl.inference import normalize_feature_payload  # noqa: E402
from rlt_online_rl.replay import RLTTransition  # noqa: E402
from rlt_online_rl.replay import ReplayManager  # noqa: E402
from rlt_online_rl.replay import TransitionSource  # noqa: E402

logger = logging.getLogger("build_warmup_journal")

CSV_TEMPLATE = """\
episode_id,start_frame,end_frame,success
0,120,280,1
"""

CSV_EXAMPLE = """\
episode_id,start_frame,end_frame,success
0,180,340,1
1,120,260,1
2,130,270,1
3,140,290,1
4,150,310,1
"""


# ---------------------------------------------------------------------------
# Critical segment I/O
# ---------------------------------------------------------------------------


def _as_bool_success(value: Any, *, default: bool = True) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "success", "ok"}:
        return True
    if text in {"0", "false", "no", "failure", "fail", "failed"}:
        return False
    return bool(int(float(text)))


def load_critical_segments(path: Path) -> list[dict[str, Any]]:
    """Load critical intervals from CSV or JSON.

    Required fields: episode_id (or episode_index), start_frame, end_frame.
    Optional: success (default True).
    Frames are episode-local and inclusive on both ends.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, Any]]
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "segments" in payload:
            payload = payload["segments"]
        if not isinstance(payload, list):
            raise ValueError(f"JSON critical file must be a list or {{'segments': [...]}}, got {type(payload)}")
        rows = [dict(item) for item in payload]
    else:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = [dict(row) for row in reader]

    segments: list[dict[str, Any]] = []
    for row in rows:
        episode_raw = row.get("episode_id", row.get("episode_index"))
        if episode_raw is None:
            raise KeyError(f"Critical row missing episode_id/episode_index: {row}")
        start = int(row["start_frame"])
        end = int(row["end_frame"])
        if end < start:
            raise ValueError(f"end_frame < start_frame in {row}")
        segments.append(
            {
                "episode_id": int(episode_raw),
                "start_frame": start,
                "end_frame": end,
                "success": _as_bool_success(row.get("success"), default=True),
            }
        )
    if not segments:
        raise ValueError(f"No critical segments found in {path}")
    return segments


# ---------------------------------------------------------------------------
# Lightweight LeRobot v2.1 reader (no HuggingFace lerobot package)
# ---------------------------------------------------------------------------


def _resize_hwc_u8(image_hwc_u8: np.ndarray, resize_hw: tuple[int, int]) -> np.ndarray:
    h, w = int(resize_hw[0]), int(resize_hw[1])
    resized = image_tools.resize_with_pad(image_hwc_u8[None, ...], h, w)[0]
    return np.asarray(resized, dtype=np.uint8)


def _decode_lerobot_image(value: Any) -> np.ndarray:
    """Decode a LeRobot parquet image cell to HWC uint8 RGB."""
    import cv2

    if isinstance(value, dict):
        raw = value.get("bytes")
        if raw:
            arr = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("cv2.imdecode failed on parquet image bytes")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        path = value.get("path")
        if path:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Failed to read image path: {path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Image dict missing bytes/path: keys={list(value)}")

    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"Expected image array ndim=3, got shape {arr.shape}")
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        max_val = float(arr.max()) if arr.size else 0.0
        arr = (arr * 255.0) if max_val <= 1.0 + 1e-3 else arr
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8, copy=False)
    return arr


def _require_pyarrow():
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "Reading LeRobot parquet frames requires pyarrow. "
            "Install with: pip install 'rlt-online-rl[offline]'  (or: pip install pyarrow)"
        ) from exc
    return pq


class LightLeRobotDataset:
    """Minimal LeRobot v2.1 local reader: meta JSONL + parquet (+ optional mp4).

    Does not import the HuggingFace ``lerobot`` package. Only needs the stdlib,
    numpy, opencv, and pyarrow (for frame access during ``build``).
    """

    def __init__(self, dataset_root: Path, *, repo_id: str | None = None):
        self.root = Path(dataset_root)
        self.repo_id = repo_id or self.root.name
        info_path = self.root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing {info_path}; expected a local LeRobot v2.1 dataset.")
        self.info = json.loads(info_path.read_text())
        self.chunks_size = int(self.info.get("chunks_size", 1000))
        self.data_path_tmpl = str(self.info["data_path"])
        self.video_path_tmpl = str(self.info.get("video_path", ""))
        self.features = dict(self.info.get("features", {}))

        self.episodes = self._load_episodes()
        self.tasks = self._load_tasks()
        self._episode_from = np.cumsum([0] + [ep["length"] for ep in self.episodes[:-1]]).astype(np.int64)
        self._parquet_cache: dict[int, Any] = {}
        self._video_cache: dict[tuple[int, str], list[np.ndarray] | None] = {}

    @property
    def num_episodes(self) -> int:
        return len(self.episodes)

    @property
    def num_frames(self) -> int:
        return int(sum(ep["length"] for ep in self.episodes))

    def episode_length(self, episode_id: int) -> int:
        return int(self.episodes[self._check_episode(episode_id)]["length"])

    def episode_global_range(self, episode_id: int) -> tuple[int, int]:
        """Half-open global index range [from, to) for an episode."""
        ep = self._check_episode(episode_id)
        start = int(self._episode_from[ep])
        return start, start + int(self.episodes[ep]["length"])

    def get_observation(
        self,
        episode_id: int,
        local_frame: int,
        *,
        resize_hw: tuple[int, int],
        prompt: str | None,
        cam_high_key: str,
        cam_wrist_key: str,
    ) -> dict[str, Any]:
        ep = self._check_episode(episode_id)
        ep_len = int(self.episodes[ep]["length"])
        if local_frame < 0 or local_frame >= ep_len:
            raise IndexError(f"episode={episode_id} frame={local_frame} out of [0, {ep_len})")

        table = self._load_episode_parquet(ep)
        state = np.asarray(table.column("observation.state")[local_frame].as_py(), dtype=np.float32).reshape(-1)
        cam_high = self._load_image(ep, local_frame, cam_high_key, table)
        cam_wrist = self._load_image(ep, local_frame, cam_wrist_key, table)
        task = prompt if prompt else self._task_for_frame(ep, local_frame, table)
        return {
            "state": state,
            "images": {
                "cam_high": _resize_hwc_u8(cam_high, resize_hw),
                "cam_right_wrist": _resize_hwc_u8(cam_wrist, resize_hw),
            },
            "prompt": task,
        }

    def get_action_chunk(
        self,
        episode_id: int,
        local_frame: int,
        chunk_len: int,
        *,
        action_dim: int,
    ) -> np.ndarray:
        """Load demo action chunk ``a_t:t+C-1`` from parquet (absolute joint space)."""
        if chunk_len <= 0:
            raise ValueError(f"chunk_len must be positive, got {chunk_len}")
        ep = self._check_episode(episode_id)
        ep_len = int(self.episodes[ep]["length"])
        if local_frame < 0 or local_frame + chunk_len > ep_len:
            raise IndexError(
                f"episode={episode_id} action chunk [{local_frame}, {local_frame + chunk_len}) "
                f"out of [0, {ep_len})"
            )
        table = self._load_episode_parquet(ep)
        if "action" not in table.column_names:
            raise KeyError(f"LeRobot parquet missing 'action' column for episode={episode_id}")
        rows = []
        for offset in range(chunk_len):
            action = np.asarray(table.column("action")[local_frame + offset].as_py(), dtype=np.float32).reshape(-1)
            if action.shape[0] < action_dim:
                raise ValueError(
                    f"episode={episode_id} frame={local_frame + offset}: "
                    f"action dim {action.shape[0]} < action_dim={action_dim}"
                )
            rows.append(action[:action_dim])
        return np.stack(rows, axis=0).astype(np.float32, copy=False)

    def _check_episode(self, episode_id: int) -> int:
        if episode_id < 0 or episode_id >= self.num_episodes:
            raise IndexError(f"episode_id={episode_id} out of range [0, {self.num_episodes})")
        return int(episode_id)

    def _load_episodes(self) -> list[dict[str, Any]]:
        path = self.root / "meta" / "episodes.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: int(r["episode_index"]))
        out: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            ep_idx = int(row["episode_index"])
            if ep_idx != i:
                raise ValueError(f"Expected contiguous episode_index, got {ep_idx} at position {i}")
            out.append({"episode_index": ep_idx, "length": int(row["length"]), "tasks": row.get("tasks", [])})
        declared = int(self.info.get("total_episodes", len(out)))
        if declared != len(out):
            logger.warning("info.total_episodes=%s but episodes.jsonl has %s rows", declared, len(out))
        return out

    def _load_tasks(self) -> dict[int, str]:
        path = self.root / "meta" / "tasks.jsonl"
        if not path.exists():
            return {}
        tasks: dict[int, str] = {}
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tasks[int(row["task_index"])] = str(row["task"])
        return tasks

    def _episode_chunk(self, episode_id: int) -> int:
        return int(episode_id) // self.chunks_size

    def _format_path(self, tmpl: str, *, episode_id: int, video_key: str = "") -> Path:
        rel = tmpl.format(
            episode_chunk=self._episode_chunk(episode_id),
            episode_index=int(episode_id),
            video_key=video_key,
        )
        return self.root / rel

    def _load_episode_parquet(self, episode_id: int):
        if episode_id in self._parquet_cache:
            return self._parquet_cache[episode_id]
        pq = _require_pyarrow()
        path = self._format_path(self.data_path_tmpl, episode_id=episode_id)
        if not path.exists():
            raise FileNotFoundError(path)
        table = pq.read_table(path)
        self._parquet_cache[episode_id] = table
        return table

    def _feature_dtype(self, key: str) -> str | None:
        feat = self.features.get(key)
        if not isinstance(feat, dict):
            return None
        return str(feat.get("dtype", ""))

    def _video_key_from_feature(self, feature_key: str) -> str:
        # observation.images.cam_high -> cam_high
        return feature_key.rsplit(".", 1)[-1]

    def _load_image(self, episode_id: int, local_frame: int, feature_key: str, table) -> np.ndarray:
        dtype = self._feature_dtype(feature_key)
        if feature_key in table.column_names:
            cell = table.column(feature_key)[local_frame].as_py()
            if isinstance(cell, dict) and cell.get("bytes"):
                return _decode_lerobot_image(cell)
            if isinstance(cell, dict) and cell.get("path"):
                return _decode_lerobot_image(cell)
            if cell is not None and not isinstance(cell, dict):
                return _decode_lerobot_image(cell)
        if dtype == "video" or self.video_path_tmpl:
            frames = self._load_video_frames(episode_id, feature_key)
            if frames is not None:
                return frames[local_frame]
        raise KeyError(f"Cannot load image feature '{feature_key}' for episode={episode_id} frame={local_frame}")

    def _load_video_frames(self, episode_id: int, feature_key: str) -> list[np.ndarray] | None:
        import cv2

        cache_key = (episode_id, feature_key)
        if cache_key in self._video_cache:
            return self._video_cache[cache_key]
        if not self.video_path_tmpl:
            self._video_cache[cache_key] = None
            return None
        video_key = self._video_key_from_feature(feature_key)
        path = self._format_path(self.video_path_tmpl, episode_id=episode_id, video_key=video_key)
        if not path.exists():
            # Some datasets store video_key as the full feature name.
            path = self._format_path(self.video_path_tmpl, episode_id=episode_id, video_key=feature_key)
        if not path.exists():
            self._video_cache[cache_key] = None
            return None
        cap = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        if not frames:
            self._video_cache[cache_key] = None
            return None
        self._video_cache[cache_key] = frames
        return frames

    def _task_for_frame(self, episode_id: int, local_frame: int, table) -> str:
        if "task_index" in table.column_names:
            task_index = int(table.column("task_index")[local_frame].as_py())
            if task_index in self.tasks:
                return self.tasks[task_index]
        tasks = self.episodes[episode_id].get("tasks") or []
        if tasks:
            return str(tasks[0])
        return ""


def load_lerobot_dataset(repo_id: str, dataset_root: Path) -> LightLeRobotDataset:
    """Load a local LeRobot v2.1 dataset without the HuggingFace lerobot package."""
    return LightLeRobotDataset(dataset_root, repo_id=repo_id)


def episode_global_range(dataset: LightLeRobotDataset, episode_id: int) -> tuple[int, int]:
    return dataset.episode_global_range(episode_id)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------


class FeatureEncoder:
    def encode_many(self, observations: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
        raise NotImplementedError


class MachineAEncoder(FeatureEncoder):
    def __init__(
        self,
        ws_url: str,
        *,
        rl_config: RLTOnlineRLConfig,
        batch_size: int,
        connect_timeout_sec: float,
        recv_timeout_sec: float,
    ):
        self._client = MachineAFeatureClient(
            ws_url,
            connect_timeout_sec=connect_timeout_sec,
            recv_timeout_sec=recv_timeout_sec,
        )
        self._rl_config = rl_config
        self._batch_size = max(1, int(batch_size))

    def encode_many(self, observations: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
        out: list[dict[str, np.ndarray]] = []
        for start in range(0, len(observations), self._batch_size):
            chunk = observations[start : start + self._batch_size]
            raw = self._client.get_features_batch(chunk)
            for obs, payload in zip(chunk, raw, strict=True):
                out.append(normalize_feature_payload(payload, self._rl_config, observation=obs))
        return out

    def close(self) -> None:
        self._client.close()


class FakeFeatureEncoder(FeatureEncoder):
    """Deterministic placeholder features for dry-run / schema checks."""

    def __init__(self, rl_config: RLTOnlineRLConfig):
        self._cfg = rl_config

    def encode_many(self, observations: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
        results = []
        for i, obs in enumerate(observations):
            state = np.asarray(obs["state"], dtype=np.float32).reshape(-1)
            proprio = state[: self._cfg.proprio_dim].astype(np.float32)
            seed = abs(hash((float(proprio[0]), float(proprio[-1]), i))) % (2**31)
            rng = np.random.default_rng(seed)
            z_rl = rng.standard_normal(self._cfg.z_dim).astype(np.float32) * 0.01
            ref = np.tile(proprio[None, :], (self._cfg.chunk_len, 1)).astype(np.float32)
            # Dry-run: demo action ≈ ref with a small offset so schema stays distinct.
            action = ref + 0.01
            results.append({"z_rl": z_rl, "proprio": proprio, "ref_chunk": ref, "action_chunk": action})
        return results


# ---------------------------------------------------------------------------
# Transition construction (Evo v2: paper x_t -> x_{t+C} overlap)
# ---------------------------------------------------------------------------


def build_overlap_frame_indices(
    segment_start: int,
    segment_stop: int,
    chunk_length: int,
    stride: int,
) -> list[int]:
    """Return sampled episode-local frames for Evo v2 cache building.

    Mirrors ``evo_rlt.adapters.lerobot.offline_dataset.build_overlap_frame_indices``.
    ``segment_stop`` is half-open (one past the last inclusive frame).

    Includes start anchors (stride), the terminal anchor ``last - C``, and every
    bootstrap state ``t+C`` so each transition has a matching next state.
    """
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if segment_stop <= segment_start:
        return []

    segment_last_frame = segment_stop - 1
    indices = set(range(segment_start, segment_stop, stride))
    terminal_anchor = segment_last_frame - chunk_length
    if terminal_anchor < segment_start:
        return sorted(indices)

    indices.add(terminal_anchor)
    start_anchors = sorted(idx for idx in indices if idx + chunk_length <= segment_last_frame)
    for start_frame in start_anchors:
        indices.add(start_frame + chunk_length)
    return sorted(indices)


def build_reward_seq(
    chunk_length: int,
    *,
    is_terminal_chunk: bool,
    episode_success: bool,
    actual_steps: int | None = None,
) -> np.ndarray:
    """Sparse terminal reward for one chunk (matches Evo-RLT ``build_reward_seq``).

    Returns shape ``(C,)`` with ``1.0`` at the last valid step iff this chunk
    ends a successful episode/segment; zeros otherwise.
    """
    steps = chunk_length if actual_steps is None else max(0, min(chunk_length, int(actual_steps)))
    rewards = np.zeros((chunk_length,), dtype=np.float32)
    if is_terminal_chunk and episode_success and steps > 0:
        rewards[steps - 1] = 1.0
    return rewards


def count_paper_transitions(
    anchor_frames: list[int],
    *,
    segment_last_frame: int,
    chunk_length: int,
) -> int:
    """Count ``x_t -> x_{t+C}`` transitions that can be formed from anchors."""
    frame_set = set(anchor_frames)
    count = 0
    for start_frame in anchor_frames:
        next_frame = start_frame + chunk_length
        if next_frame > segment_last_frame:
            continue
        if next_frame in frame_set:
            count += 1
    return count


def build_transitions_for_critical_segment(
    *,
    episode_id: int,
    anchor_frames: list[int],
    features: list[dict[str, np.ndarray]],
    chunk_len: int,
    stride: int,
    segment_last_frame: int,
    episode_success: bool = True,
) -> list[RLTTransition]:
    """Build Evo v2 warmup transitions for one critical interval.

    Mirrors ``evo_rlt`` ``_encoded_to_transitions``:

    - state / ref at anchor ``t`` (VLA); action = demo ``a_t:t+C-1``
    - next state / next ref at ``t + chunk_len`` (not the next list neighbor)
    - ``done`` when ``t + C`` is the segment's last frame
    - terminal reward ``1.0`` on the last step iff ``episode_success``
    """
    if len(features) != len(anchor_frames):
        raise ValueError(f"features length {len(features)} != anchor_frames {len(anchor_frames)}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")
    if chunk_len % stride != 0:
        raise ValueError(f"chunk_len={chunk_len} must be divisible by stride={stride}")
    if len(features) < 2:
        logger.warning(
            "episode=%s anchors=%s too few for transitions (need >=2); skipped",
            episode_id,
            anchor_frames,
        )
        return []

    frame_to_idx = {int(frame): idx for idx, frame in enumerate(anchor_frames)}
    base = int(TransitionSource.BASE)
    transitions: list[RLTTransition] = []

    for start_frame in anchor_frames:
        next_frame = int(start_frame) + chunk_len
        if next_frame > segment_last_frame:
            continue
        next_idx = frame_to_idx.get(next_frame)
        if next_idx is None:
            raise ValueError(
                f"episode={episode_id}: missing next-state anchor for frame "
                f"{start_frame} -> {next_frame}; overlap indices must include every t+C"
            )
        cur_idx = frame_to_idx[int(start_frame)]
        current = features[cur_idx]
        nxt = features[next_idx]
        ref_chunk = np.asarray(current["ref_chunk"], dtype=np.float32)
        if ref_chunk.shape[0] < chunk_len:
            raise ValueError(f"ref_chunk length {ref_chunk.shape[0]} < chunk_len {chunk_len}")
        ref_chunk = ref_chunk[:chunk_len].astype(np.float32, copy=False)
        if "action_chunk" not in current:
            raise KeyError(
                f"episode={episode_id} frame={start_frame}: feature missing demo action_chunk "
                "(load from LeRobot 'action' column)"
            )
        action_chunk = np.asarray(current["action_chunk"], dtype=np.float32)
        if action_chunk.shape[0] < chunk_len:
            raise ValueError(
                f"action_chunk length {action_chunk.shape[0]} < chunk_len {chunk_len} "
                f"(episode={episode_id} frame={start_frame})"
            )
        action_chunk = action_chunk[:chunk_len].astype(np.float32, copy=False)
        if action_chunk.shape[1:] != ref_chunk.shape[1:]:
            raise ValueError(
                f"action_chunk shape {action_chunk.shape} incompatible with ref_chunk {ref_chunk.shape} "
                f"(episode={episode_id} frame={start_frame})"
            )
        next_ref = np.asarray(nxt["ref_chunk"], dtype=np.float32)[:chunk_len]
        is_terminal = next_frame == int(segment_last_frame)
        rewards = build_reward_seq(
            chunk_len,
            is_terminal_chunk=is_terminal,
            episode_success=bool(episode_success),
            actual_steps=chunk_len,
        )

        transitions.append(
            RLTTransition(
                z_rl=np.asarray(current["z_rl"], dtype=np.float32),
                proprio=np.asarray(current["proprio"], dtype=np.float32),
                ref_chunk=ref_chunk,
                action_chunk=action_chunk,
                rewards=rewards,
                done=bool(is_terminal),
                next_z_rl=np.asarray(nxt["z_rl"], dtype=np.float32),
                next_proprio=np.asarray(nxt["proprio"], dtype=np.float32),
                next_ref_chunk=next_ref,
                source=base,
                source_chunk=np.full((chunk_len,), base, dtype=np.uint8),
                collection_phase="warmup",
                success=int(is_terminal and episode_success),
                intervention_flag=False,
                episode_id=int(episode_id),
                step_id=int(start_frame),
                actual_steps=chunk_len,
            )
        )

    if episode_success and transitions and not any(t.done for t in transitions):
        raise ValueError(
            f"episode={episode_id}: successful segment has no terminal chunk under "
            f"stride={stride} chunk_len={chunk_len} last_frame={segment_last_frame}"
        )
    return transitions


# ---------------------------------------------------------------------------
# CSV helpers (template / validate / list)
# ---------------------------------------------------------------------------


def _write_critical_csv(path: Path, rows: list[dict[str, int]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["episode_id", "start_frame", "end_frame", "success"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_csv_template(path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CSV_TEMPLATE)
    print(f"Wrote template -> {path}")
    print("Columns: episode_id, start_frame, end_frame, success (optional)")


def write_csv_example(path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CSV_EXAMPLE)
    print(f"Wrote example -> {path}")
    print("Replace frame ranges with your annotations before build.")


def init_csv_from_dataset(
    path: Path,
    repo_id: str,
    dataset_root: Path,
    *,
    max_episodes: int = 0,
    critical_frac: float = 1.0,
    overwrite: bool = False,
) -> None:
    """Read LeRobot and write one critical-segment row per episode.

    By default each row spans the full episode ``[0, num_frames-1]``. Pass
    ``--critical-frac`` in (0, 1] to pre-fill the middle fraction as a starting
    guess (e.g. 0.4 → central 40% of each episode).
    """
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it.")
    if not 0.0 < critical_frac <= 1.0:
        raise ValueError(f"--critical-frac must be in (0, 1], got {critical_frac}")

    dataset = load_lerobot_dataset(repo_id, dataset_root)
    n_eps = dataset.num_episodes if max_episodes <= 0 else min(dataset.num_episodes, max_episodes)
    rows: list[dict[str, int]] = []
    for ep in range(n_eps):
        ep_len = dataset.episode_length(ep)
        if ep_len <= 0:
            continue
        last = ep_len - 1
        if critical_frac >= 1.0:
            start, end = 0, last
        else:
            seg_len = max(1, int(round(ep_len * critical_frac)))
            start = max(0, (ep_len - seg_len) // 2)
            end = min(last, start + seg_len - 1)
        rows.append(
            {
                "episode_id": ep,
                "start_frame": start,
                "end_frame": end,
                "success": 1,
            }
        )

    _write_critical_csv(path, rows)
    frac_note = "full episode" if critical_frac >= 1.0 else f"middle {critical_frac:.0%}"
    print(
        f"Wrote {path}: {len(rows)} rows from {dataset.root} ({frac_note} per episode). "
        "Edit start_frame/end_frame to your critical intervals."
    )


def list_episodes(repo_id: str, dataset_root: Path, *, max_rows: int) -> None:
    dataset = load_lerobot_dataset(repo_id, dataset_root)
    n = dataset.num_episodes if max_rows <= 0 else min(dataset.num_episodes, max_rows)
    print(f"{'episode_id':>10}  {'num_frames':>10}  {'global_from':>11}  {'global_to':>9}")
    print("-" * 46)
    for ep in range(n):
        g_from, g_to = dataset.episode_global_range(ep)
        print(f"{ep:10d}  {g_to - g_from:10d}  {g_from:11d}  {g_to:9d}")
    if max_rows > 0 and dataset.num_episodes > max_rows:
        print(f"... ({dataset.num_episodes - max_rows} more episodes; use --max-episodes 0 for all)")


def validate_critical_segments(
    critical_path: Path,
    repo_id: str,
    dataset_root: Path,
    *,
    chunk_len: int,
    stride: int,
) -> int:
    segments = load_critical_segments(critical_path)
    dataset = load_lerobot_dataset(repo_id, dataset_root)
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[int, int, int]] = set()

    for i, seg in enumerate(segments):
        row_label = f"row {i + 2}"
        ep = int(seg["episode_id"])
        start = int(seg["start_frame"])
        end = int(seg["end_frame"])
        if (ep, start, end) in seen:
            warnings.append(f"{row_label}: duplicate episode={ep} [{start},{end}]")
        seen.add((ep, start, end))

        if ep < 0 or ep >= dataset.num_episodes:
            errors.append(f"{row_label}: episode_id={ep} out of range [0, {dataset.num_episodes})")
            continue

        g_from, g_to = dataset.episode_global_range(ep)
        ep_len = g_to - g_from
        if start < 0:
            errors.append(f"{row_label}: start_frame={start} must be >= 0")
        if end >= ep_len:
            errors.append(f"{row_label}: end_frame={end} >= episode length {ep_len}")
        if end < start:
            errors.append(f"{row_label}: end_frame < start_frame")

        seg_len = end - start + 1
        if seg_len < chunk_len:
            warnings.append(f"{row_label}: segment length {seg_len} < chunk_len={chunk_len}")

        anchors = build_overlap_frame_indices(start, end + 1, chunk_len, stride)
        n_transitions = count_paper_transitions(
            anchors, segment_last_frame=end, chunk_length=chunk_len
        )
        if n_transitions < 1:
            errors.append(
                f"{row_label}: ep={ep} [{start},{end}] -> {len(anchors)} anchor(s), "
                f"{n_transitions} paper transitions (x_t->x_t+C); need >= 1"
            )
        else:
            print(
                f"OK {row_label}: ep={ep} [{start},{end}] len={seg_len} "
                f"anchors={len(anchors)} transitions={n_transitions} success={int(bool(seg.get('success', True)))}"
            )

    for msg in warnings:
        print(f"WARN: {msg}")
    for msg in errors:
        print(f"ERROR: {msg}")
    print(f"\nSummary: segments={len(segments)} errors={len(errors)} warnings={len(warnings)}")
    return 1 if errors else 0


# ---------------------------------------------------------------------------
# Build journal
# ---------------------------------------------------------------------------


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/huggingface/lerobot/openpi/screw_sorting_single_rl"),
    )
    parser.add_argument("--repo-id", type=str, default="openpi/screw_sorting_single_rl")


def _add_chunk_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--stride", type=int, default=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p_tpl = sub.add_parser("template", help="Write single-row placeholder CSV (no dataset).")
    p_tpl.add_argument("path", type=Path, help="Output CSV path.")

    p_init = sub.add_parser("init", help="Read LeRobot and write one CSV row per episode.")
    p_init.add_argument("path", type=Path, help="Output CSV path.")
    _add_dataset_args(p_init)
    p_init.add_argument("--max-episodes", type=int, default=0, help="0 = all episodes.")
    p_init.add_argument(
        "--critical-frac",
        type=float,
        default=1.0,
        help="Fraction of each episode to pre-fill (1.0=full episode, 0.4=middle 40%%).",
    )
    p_init.add_argument("--overwrite", action="store_true")

    p_ex = sub.add_parser("example", help="Write hardcoded 5-episode example CSV (no dataset).")
    p_ex.add_argument("path", type=Path, help="Output CSV path.")

    p_list = sub.add_parser("list", help="List episode frame counts.")
    _add_dataset_args(p_list)
    p_list.add_argument("--max-episodes", type=int, default=0, help="0 = print all.")

    p_val = sub.add_parser("validate", help="Validate critical CSV against dataset.")
    p_val.add_argument("--critical-path", type=Path, required=True)
    _add_dataset_args(p_val)
    _add_chunk_args(p_val)

    p_build = sub.add_parser("build", help="Build warmup replay journal (default).")
    _add_dataset_args(p_build)
    p_build.add_argument("--critical-path", type=Path, required=True)
    p_build.add_argument(
        "--output-journal",
        type=Path,
        default=Path("runs/screw_sorting/replay/replay_journal.pkl"),
    )
    p_build.add_argument("--machine-a-ws-url", type=str, default="ws://127.0.0.1:8000")
    p_build.add_argument("--fake-features", action="store_true")
    _add_chunk_args(p_build)
    p_build.add_argument("--z-dim", type=int, default=2048)
    p_build.add_argument("--action-dim", type=int, default=7)
    p_build.add_argument("--proprio-dim", type=int, default=7)
    p_build.add_argument("--resize-h", type=int, default=224)
    p_build.add_argument("--resize-w", type=int, default=224)
    p_build.add_argument("--feature-batch-size", type=int, default=16)
    p_build.add_argument("--machine-a-connect-timeout-sec", type=float, default=5.0)
    p_build.add_argument("--machine-a-recv-timeout-sec", type=float, default=120.0)
    p_build.add_argument("--prompt", type=str, default="")
    p_build.add_argument("--cam-high-key", type=str, default="observation.images.cam_high")
    p_build.add_argument("--cam-wrist-key", type=str, default="observation.images.cam_right_wrist")
    p_build.add_argument("--feature-cache", type=Path, default=None)
    p_build.add_argument("--capacity", type=int, default=200_000)
    p_build.add_argument("--overwrite", action="store_true")
    p_build.add_argument("--max-segments", type=int, default=0)
    p_build.add_argument("--log-every", type=int, default=1)

    # Backward compat: `build` flags at top level when no subcommand given.
    parser.add_argument("--critical-path", type=Path, default=None)
    parser.add_argument("--output-journal", type=Path, default=Path("runs/screw_sorting/replay/replay_journal.pkl"))
    parser.add_argument("--machine-a-ws-url", type=str, default="ws://127.0.0.1:8000")
    parser.add_argument("--fake-features", action="store_true")
    _add_dataset_args(parser)
    _add_chunk_args(parser)
    parser.add_argument("--z-dim", type=int, default=2048)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--proprio-dim", type=int, default=7)
    parser.add_argument("--resize-h", type=int, default=224)
    parser.add_argument("--resize-w", type=int, default=224)
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--machine-a-connect-timeout-sec", type=float, default=5.0)
    parser.add_argument("--machine-a-recv-timeout-sec", type=float, default=120.0)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--cam-high-key", type=str, default="observation.images.cam_high")
    parser.add_argument("--cam-wrist-key", type=str, default="observation.images.cam_right_wrist")
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=1)

    args = parser.parse_args()
    if args.command is None:
        if args.critical_path is None:
            parser.error("missing COMMAND or --critical-path (use: template | example | list | validate | build)")
        args.command = "build"
    return args


def run_build(args: argparse.Namespace) -> None:
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.chunk_len <= 0:
        raise ValueError("--chunk-len must be > 0")
    if args.chunk_len % args.stride != 0:
        raise ValueError(f"--chunk-len={args.chunk_len} must be divisible by --stride={args.stride}")

    output_journal = args.output_journal
    if output_journal.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_journal} exists; pass --overwrite to replace it.")
        output_journal.unlink()

    segments = load_critical_segments(args.critical_path)
    if args.max_segments > 0:
        segments = segments[: args.max_segments]
    logger.info("Loaded %d critical segments from %s", len(segments), args.critical_path)

    dataset = load_lerobot_dataset(args.repo_id, args.dataset_root)
    logger.info(
        "LeRobot dataset episodes=%s frames=%s root=%s (lightweight reader, no lerobot pkg)",
        dataset.num_episodes,
        dataset.num_frames,
        args.dataset_root,
    )

    rl_config = RLTOnlineRLConfig(
        action_dim=args.action_dim,
        chunk_len=args.chunk_len,
        z_dim=args.z_dim,
        proprio_dim=args.proprio_dim,
    )
    resize_hw = (args.resize_h, args.resize_w)
    prompt = args.prompt.strip() or None

    feature_cache: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    if args.feature_cache is not None and args.feature_cache.exists():
        with args.feature_cache.open("rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            feature_cache = loaded
            logger.info("Loaded %d cached feature anchors from %s", len(feature_cache), args.feature_cache)

    encoder: FeatureEncoder
    if args.fake_features:
        logger.warning("Using FakeFeatureEncoder — journal is NOT valid for real training.")
        encoder = FakeFeatureEncoder(rl_config)
    else:
        encoder = MachineAEncoder(
            args.machine_a_ws_url,
            rl_config=rl_config,
            batch_size=args.feature_batch_size,
            connect_timeout_sec=args.machine_a_connect_timeout_sec,
            recv_timeout_sec=args.machine_a_recv_timeout_sec,
        )

    manager = ReplayManager(args.capacity, journal_path=str(output_journal), seed=0)
    total_transitions = 0
    skipped = 0

    try:
        for seg_i, seg in enumerate(segments):
            episode_id = int(seg["episode_id"])
            local_start = int(seg["start_frame"])
            local_end = int(seg["end_frame"])
            episode_success = bool(seg.get("success", True))
            ep_len = dataset.episode_length(episode_id)
            if local_start < 0 or local_end >= ep_len:
                raise ValueError(
                    f"episode={episode_id} critical=[{local_start},{local_end}] "
                    f"out of episode length {ep_len}"
                )

            anchor_frames = build_overlap_frame_indices(
                local_start,
                local_end + 1,
                args.chunk_len,
                args.stride,
            )
            n_expected = count_paper_transitions(
                anchor_frames,
                segment_last_frame=local_end,
                chunk_length=args.chunk_len,
            )
            if n_expected < 1:
                logger.warning(
                    "episode=%s critical=[%s,%s] anchors=%s paper_transitions=%s; skipped",
                    episode_id,
                    local_start,
                    local_end,
                    anchor_frames,
                    n_expected,
                )
                skipped += 1
                continue

            missing_keys: list[tuple[int, int]] = []
            missing_obs: list[dict[str, Any]] = []

            for local_frame in anchor_frames:
                cache_key = (episode_id, local_frame)
                if cache_key in feature_cache:
                    continue
                obs = dataset.get_observation(
                    episode_id,
                    local_frame,
                    resize_hw=resize_hw,
                    prompt=prompt,
                    cam_high_key=args.cam_high_key,
                    cam_wrist_key=args.cam_wrist_key,
                )
                missing_keys.append(cache_key)
                missing_obs.append(obs)

            if missing_obs:
                logger.info(
                    "episode=%s critical=[%s,%s] encoding %d/%d anchors via %s",
                    episode_id,
                    local_start,
                    local_end,
                    len(missing_obs),
                    len(anchor_frames),
                    type(encoder).__name__,
                )
                encoded = encoder.encode_many(missing_obs)
                for key, feat in zip(missing_keys, encoded, strict=True):
                    feature_cache[key] = feat

            features = []
            for local_frame in anchor_frames:
                cache_key = (episode_id, local_frame)
                feat = dict(feature_cache[cache_key])
                # Demo action chunk only needed for start anchors (t -> t+C).
                # Bootstrap-only frames near the segment end may not have C future actions.
                if local_frame + args.chunk_len <= ep_len:
                    feat["action_chunk"] = dataset.get_action_chunk(
                        episode_id,
                        local_frame,
                        args.chunk_len,
                        action_dim=args.action_dim,
                    )
                feature_cache[cache_key] = feat
                features.append(feat)
            transitions = build_transitions_for_critical_segment(
                episode_id=episode_id,
                anchor_frames=anchor_frames,
                features=features,
                chunk_len=args.chunk_len,
                stride=args.stride,
                segment_last_frame=local_end,
                episode_success=episode_success,
            )
            if not transitions:
                skipped += 1
                continue
            manager.add_transitions(transitions)
            total_transitions += len(transitions)
            if (seg_i + 1) % max(1, args.log_every) == 0:
                terminal_reward = float(
                    sum(float(np.sum(t.rewards)) for t in transitions if t.done)
                )
                logger.info(
                    "[%d/%d] ep=%d [%d,%d] success=%s anchors=%d -> %d transitions "
                    "terminal_reward=%.1f cumulative=%d",
                    seg_i + 1,
                    len(segments),
                    episode_id,
                    local_start,
                    local_end,
                    int(episode_success),
                    len(anchor_frames),
                    len(transitions),
                    terminal_reward,
                    total_transitions,
                )
    finally:
        if hasattr(encoder, "close"):
            encoder.close()  # type: ignore[attr-defined]
        if args.feature_cache is not None:
            args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
            with args.feature_cache.open("wb") as f:
                pickle.dump(feature_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("Wrote feature cache (%d anchors) -> %s", len(feature_cache), args.feature_cache)

    stats = manager.stats()
    logger.info(
        "Done. transitions=%s skipped_segments=%s journal=%s stats=%s",
        total_transitions,
        skipped,
        output_journal,
        stats,
    )
    if total_transitions == 0:
        raise RuntimeError("No transitions written. Check critical lengths vs --chunk-len.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if args.command == "template":
        write_csv_template(args.path)
        return
    if args.command == "init":
        init_csv_from_dataset(
            args.path,
            args.repo_id,
            args.dataset_root,
            max_episodes=args.max_episodes,
            critical_frac=args.critical_frac,
            overwrite=args.overwrite,
        )
        return
    if args.command == "example":
        write_csv_example(args.path)
        return
    if args.command == "list":
        list_episodes(args.repo_id, args.dataset_root, max_rows=args.max_episodes)
        return
    if args.command == "validate":
        raise SystemExit(
            validate_critical_segments(
                args.critical_path,
                args.repo_id,
                args.dataset_root,
                chunk_len=args.chunk_len,
                stride=args.stride,
            )
        )
    if args.command == "build":
        run_build(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
