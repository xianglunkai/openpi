# Replay To Real Robot

This directory contains tools for exporting recorded replay chunks and playing
them back on the real robot.

The playback tools are meant for alignment, inspection, and debugging. They are
not part of the online replay buffer itself.

## Timing Semantics

The export step reads the online RL replay journal and writes a compact playback
artifact. The playback step sends joint chunks to the robot at a chosen control
rate.

The exported chunks preserve the replay-level inputs and actor-comparison
semantics:

- `ref`: Machine A / VLA reference chunk
- `actor`: deterministic actor output from the selected actor snapshot

For human or mixed replay windows, the actor is still evaluated on the replay
state and reference. The tool is for comparing what the selected actor would do
against the replay reference; it is not a raw teleop event player.

## Files

### `export_episode_joint_playback.py`

Exports selected replay windows from a replay journal into a playback file.

Common commands:

List available episodes:

```bash
python scripts/replay_real_robot/export_episode_joint_playback.py \
  --run-dir runs/agilex_ethernet \
  --replay-path runs/agilex_ethernet/replay/replay_journal.pkl \
  --list-episodes
```

Export one episode:

```bash
python scripts/replay_real_robot/export_episode_joint_playback.py \
  --run-dir runs/agilex_ethernet \
  --replay-path runs/agilex_ethernet/replay/replay_journal.pkl \
  --episode-id 12 \
  --output-dir runs/agilex_ethernet/playback/episode_000012
```

Useful options:

- `--run-dir`: run directory that owns `replay/` and `actor_snapshot/`
- `--replay-path`: replay journal to read
- `--episode-id`: episode to export
- `--output-dir`: playback artifact directory
- `--list-episodes`: print exportable episodes and exit
- `--offline-dir`: resolve the actor snapshot from an offline experiment
- `--snapshot-path`: explicit actor snapshot path
- `--disable-ref-input`: zero actor reference input when producing actor chunks
- `--keep-terminal-success`: keep the final terminal success chunk

### `play_exported_joint_chunks_on_robot.py`

Plays an exported playback artifact on the real robot.

Dry-run first:

```bash
python scripts/replay_real_robot/play_exported_joint_chunks_on_robot.py \
  --input-dir runs/agilex_ethernet/playback/episode_000012 \
  --mode ref \
  --dry-run
```

Play the VLA reference:

```bash
python scripts/replay_real_robot/play_exported_joint_chunks_on_robot.py \
  --input-dir runs/agilex_ethernet/playback/episode_000012 \
  --mode ref \
  --step-hz 20
```

Play the recorded executed action:

```bash
python scripts/replay_real_robot/play_exported_joint_chunks_on_robot.py \
  --input-dir runs/agilex_ethernet/playback/episode_000012 \
  --mode actor \
  --step-hz 20
```

Useful options:

- `--input-dir`: playback directory produced by the exporter
- `--mode`: `ref` or `actor`
- `--step-hz`: per-step playback frequency inside a chunk
- `--dry-run`: validate and print commands without publishing actions
- `--start-chunk`: first chunk index to play
- `--max-chunks`: maximum number of chunks to play
- `--chunk-boundary-interval-ms`: interval between chunk boundaries
- `--topic`: ROS JointState command topic
- `--startup-ramp-duration-sec`: ramp from current state to the first frame
- `--post-reset-hold-sec`: hold the first frame after startup ramp

## Recommended Workflow

1. List candidate episodes.
2. Export one episode into a playback artifact.
3. Run dry-run playback.
4. Play `ref` at a conservative speed.
5. Play `actor` only after the reference playback looks safe.

## Export Semantics

The exporter reads `RLTTransition` records from `replay_journal.pkl`. For each
selected replay window it records:

- episode id and step id
- collection phase
- source and source chunk
- success flag
- `ref_chunk`
- deterministic actor output generated from the selected actor snapshot

The tool does not reconstruct raw episode files and does not request Machine A.
It reads replay features from the journal and uses the selected actor snapshot
to produce `actor_chunks`.

## Reset And Replay Boundaries

Reset behavior is handled by the normal robot runtime, not by the replay
journal. The playback tool simply publishes the selected chunks. Before
playback, reset or position the robot manually according to the episode you are
trying to inspect.

Replay windows can overlap when dense replay is enabled. When playing back
exported chunks, remember that each chunk is a replay training sample, not
necessarily a unique contiguous episode segment unless the export settings
select chunk-boundary windows only.

## Safety Notes

- Always start with `--dry-run`.
- Use conservative `--control-hz` values for first playback.
- Verify robot start pose before publishing any chunk.
- Prefer playing `ref` before `actor`.
- Stop playback immediately if the robot state diverges from the expected
  episode context.
