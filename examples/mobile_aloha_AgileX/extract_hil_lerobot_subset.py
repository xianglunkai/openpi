"""Build one LeRobot v2.1 dataset from HIL rollouts.

Keeps two kinds of data, both only from episodes marked success:

* the whole episode, when every frame is autonomous
* each contiguous human-intervention run, when the episode also contains autonomous frames

Failure and timeout episodes are left out. Autonomous frames inside a successful
intervention episode are left out too.

Success is read from ``meta/hil_outcomes.jsonl`` written when the episode is saved.
``s`` is success. ``f`` and timeout are failure and are left out. No episode list
is passed on the command line.

Example:
    bash examples/mobile_aloha_AgileX/run_extract_hil_lerobot_subset.sh \
        /data/huggingface/lerobot/openpi_hil/hil_screw_sorting_20261029_171747
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

_DEFAULT_FEATURE_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
_INTERVENTION_KEYS = ("is_intervention", "complementary_info.is_intervention")


@dataclass(frozen=True)
class Piece:
    source_episode: int
    start: int
    end: int
    kind: str
    task: str

    @property
    def length(self) -> int:
        return self.end - self.start


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def intervention_key(column_names: list[str]) -> str:
    for key in _INTERVENTION_KEYS:
        if key in column_names:
            return key
    raise KeyError(f"No intervention column in {column_names}. Expected one of {_INTERVENTION_KEYS}.")


def episode_parquet(root: Path, info: dict, episode_index: int) -> Path:
    chunk = int(episode_index) // int(info.get("chunks_size", 1000))
    relative = info["data_path"].format(episode_chunk=chunk, episode_index=episode_index, chunk_index=chunk, file_index=0)
    path = root / relative
    if path.is_file():
        return path
    matches = sorted((root / "data").glob(f"chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Missing parquet for episode {episode_index}: {path}")


def read_flags(path: Path) -> np.ndarray:
    schema_names = pq.read_schema(path).names
    key = intervention_key(schema_names)
    values = pq.read_table(path, columns=[key]).column(key).to_pylist()
    return np.asarray([float(np.asarray(value).reshape(-1)[0]) for value in values], dtype=np.float32)


def contiguous_runs(flags: np.ndarray) -> list[tuple[int, int, bool]]:
    if flags.size == 0:
        return []
    human = flags > 0.5
    runs: list[tuple[int, int, bool]] = []
    start = 0
    current = bool(human[0])
    for index, flag in enumerate(human):
        if bool(flag) != current:
            runs.append((start, index, current))
            start = index
            current = bool(flag)
    runs.append((start, int(flags.size), current))
    return runs


def success_from_sidecar(root: Path) -> dict[int, str]:
    outcomes: dict[int, str] = {}
    for row in load_jsonl(root / "meta" / "hil_outcomes.jsonl"):
        outcomes[int(row["episode_index"])] = str(row["status"])
    return outcomes


def select_pieces(
    episode_index: int,
    flags: np.ndarray,
    task: str,
    *,
    success: bool | None,
    min_frames: int,
) -> list[Piece]:
    pieces: list[Piece] = []
    if not success:
        return pieces
    fully_auto = flags.size > 0 and not bool(np.any(flags > 0.5))
    if fully_auto:
        if flags.size >= min_frames:
            pieces.append(Piece(episode_index, 0, int(flags.size), "auto_success", task))
        return pieces
    for start, end, is_human in contiguous_runs(flags):
        if is_human and end - start >= min_frames:
            pieces.append(Piece(episode_index, start, end, "intervention", task))
    return pieces


def image_keys(info: dict) -> list[str]:
    return [key for key, feature in info["features"].items() if feature.get("dtype") in {"image", "video"}]


def output_features(info: dict) -> dict:
    features = {}
    for key, feature in info["features"].items():
        if key in _DEFAULT_FEATURE_KEYS:
            continue
        copied = dict(feature)
        copied["shape"] = tuple(copied["shape"])
        if copied.get("dtype") == "video":
            raise ValueError(f"{key} is video. This extractor reads PNG frames stored in the parquet.")
        if key in _INTERVENTION_KEYS:
            copied["shape"] = (1,)
            features["is_intervention"] = copied
            continue
        features[key] = copied
    return features


def decode_image(cell) -> np.ndarray:
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw:
            with Image.open(io.BytesIO(raw)) as image:
                return np.asarray(image.convert("RGB"))
        path = cell.get("path")
        if path:
            with Image.open(path) as image:
                return np.asarray(image.convert("RGB"))
    if isinstance(cell, (bytes, bytearray)):
        with Image.open(io.BytesIO(cell)) as image:
            return np.asarray(image.convert("RGB"))
    raise TypeError(f"Cannot decode image cell of type {type(cell)}")


def as_vector(value, dtype: str) -> np.ndarray:
    return np.asarray(value, dtype=dtype).reshape(-1)


def write_dataset(src: Path, output: Path, pieces: list[Piece], info: dict) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id=f"local/{output.name}",
        fps=int(info["fps"]),
        root=output,
        robot_type=info.get("robot_type"),
        features=output_features(info),
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    cameras = image_keys(info)
    provenance = []
    try:
        for new_index, piece in enumerate(pieces):
            path = episode_parquet(src, info, piece.source_episode)
            table = pq.read_table(path).slice(piece.start, piece.length)
            rows = table.to_pylist()
            for row in rows:
                frame = {
                    "observation.state": as_vector(row["observation.state"], np.float32),
                    "action": as_vector(row["action"], np.float32),
                    "is_intervention": np.asarray(
                        [1.0 if piece.kind == "intervention" else 0.0],
                        dtype=np.float32,
                    ),
                    "task": piece.task,
                }
                for camera in cameras:
                    frame[camera] = decode_image(row[camera])
                dataset.add_frame(frame)
            dataset.save_episode()
            provenance.append(
                {
                    "episode_index": new_index,
                    "kind": piece.kind,
                    "source_dataset": str(src),
                    "source_episode": piece.source_episode,
                    "source_start": piece.start,
                    "source_end": piece.end,
                    "length": piece.length,
                    "task": piece.task,
                }
            )
            print(
                f"wrote episode {new_index}: {piece.kind} from source {piece.source_episode} "
                f"frames [{piece.start}, {piece.end}) length {piece.length}"
            )
    finally:
        stop_image_writer = getattr(dataset, "stop_image_writer", None)
        if callable(stop_image_writer):
            stop_image_writer()
    extract_path = output / "meta" / "hil_extract.jsonl"
    extract_path.write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
    print(f"Wrote {len(pieces)} episodes to {output}")


def build_plan(src: Path, min_frames: int) -> tuple[list[Piece], list[str]]:
    info = load_json(src / "meta" / "info.json")
    episodes = load_jsonl(src / "meta" / "episodes.jsonl")
    if not episodes:
        raise FileNotFoundError(f"No episodes in {src / 'meta' / 'episodes.jsonl'}")
    sidecar = success_from_sidecar(src)
    notes: list[str] = []
    pieces: list[Piece] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        task = episode["tasks"][0] if episode.get("tasks") else ""
        flags = read_flags(episode_parquet(src, info, episode_index))
        human_frames = int(np.sum(flags > 0.5))
        fully_auto = human_frames == 0 and flags.size > 0
        if not sidecar:
            success = None
            status = "unrecorded"
        elif episode_index not in sidecar:
            success = None
            status = "missing"
            notes.append(f"episode {episode_index} has no outcome record and is left out")
        else:
            status = sidecar[episode_index]
            success = status == "success"
        if success is True and not fully_auto:
            notes.append(
                f"episode {episode_index} succeeded with {human_frames} intervention frames; "
                "only those intervention frames are kept"
            )
        if fully_auto and success is None:
            notes.append(
                f"episode {episode_index} is fully automatic ({flags.size} frames) but success was not recorded"
            )
        selected = select_pieces(episode_index, flags, task, success=success is True, min_frames=min_frames)
        pieces.extend(selected)
        print(
            f"source episode {episode_index}: length {flags.size} intervention {human_frames} "
            f"fully_auto {fully_auto} status {status} "
            f"kept {len(selected)}"
        )
    return pieces, notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="Existing LeRobot dataset directory")
    parser.add_argument("--output", type=Path, help="New dataset directory")
    parser.add_argument("--min-frames", type=int, default=1, help="Drop episodes and intervention runs shorter than this")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Print what would be kept and do not write")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    if not (src / "meta" / "info.json").is_file():
        raise SystemExit(f"Not a LeRobot dataset: {src}")
    pieces, notes = build_plan(src, args.min_frames)
    for note in notes:
        print(f"note: {note}")
    auto_count = sum(piece.kind == "auto_success" for piece in pieces)
    intervention_count = sum(piece.kind == "intervention" for piece in pieces)
    print(
        f"plan: {auto_count} fully automatic success episodes, "
        f"{intervention_count} intervention segments, {sum(piece.length for piece in pieces)} frames"
    )
    if not success_from_sidecar(src):
        message = (
            "This dataset has no meta/hil_outcomes.jsonl, so success and failure were not saved. "
            "Fully automatic episodes cannot be separated from timeouts."
        )
        if args.plan_only:
            print(message)
            return
        raise SystemExit(message)
    if args.output is None:
        raise SystemExit("--output is required unless --plan-only is set")
    if not pieces:
        raise SystemExit("Nothing selected. No dataset written.")
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise SystemExit(f"{output} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    info = load_json(src / "meta" / "info.json")
    write_dataset(src, output, pieces, info)


if __name__ == "__main__":
    main()
