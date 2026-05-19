# Offline Scripts

This directory contains offline replay training and analysis tools for the
online RL runtime.

The tools use two explicit inputs:

- `--replay-path`: replay journal or exported replay subset
- `--model-dir`: actor/checkpoint directory for analysis

Actor analysis tools support:

- `--actor-mode mean`: analyze deterministic actor mean
- `--actor-mode sample`: sample from the actor fixed-std distribution

Replay filtering uses two semantic axes:

- `collection_phase`: `warmup`, `online`, or `all`
- `source`: `base`, `rl`, `human`, `mixed`, or `all`

Do not invent secondary source labels such as `policy/intervention` for offline
analysis. Use `source` and `collection_phase` together.

## Replay Semantics

Each replay transition should be interpreted with:

- `source`: chunk-level control source, one of `BASE / RL / HUMAN / MIXED`
- `source_chunk`: per-step control source inside the chunk
- `collection_phase_id`: `warmup`, `online`, or `unknown`

`unknown` should only appear in old replay files or hand-written incomplete
journals.

The online runtime builds replay at episode end and saves raw episode traces
under `replay/episodes/`. Offline scripts read `replay_journal.pkl`; they do not
read raw episode files by default.

Human data in replay is already aligned to control ticks. It is not the raw
teleop event stream.

## Files

### `_common.py`

Shared utility module. It is not intended to be run directly.

It handles:

- loading actor snapshots and replay journals
- inferring task directories from replay paths
- replay filtering
- actor mean/sample inference helpers

### `offline_train_from_replay.py`

Trains actor/critic from an existing replay journal without running the robot.

Typical uses:

- train from a fixed replay dataset
- train from only warmup, online, human, or mixed subsets
- export an online-compatible bundle for continued online RL
- run no-reference ablations with `--disable-ref-input`

Command template:

```bash
python scripts/offline/offline_train_from_replay.py \
  --replay-path runs/agilex_ethernet/replay/replay_journal.pkl \
  --steps 10000 \
  --batch-size 128 \
  --seed 0 \
  --bc-weight 2.0 \
  --q-weight 0.1 \
  --output-dir runs/agilex_ethernet/offline_train_bcq
```

Useful options:

- `--replay-path`: replay journal to train from
- `--steps`: total offline training steps
- `--batch-size`: transitions sampled per training step
- `--seed`: random seed
- `--bc-weight`: BC term weight
- `--q-weight`: Q term weight
- `--delta-weight`: overrides `rl_config.delta_weight`
- `--fixed-std`: overrides actor sampling std
- `--actor-hidden-dim`, `--actor-num-layers`: actor capacity overrides
- `--critic-hidden-dim`, `--critic-num-layers`: critic capacity overrides
- `--eval-every`: train/validation fit evaluation interval
- `--val-ratio`: validation split ratio
- `--disable-ref-input`: zero out actor `ref_chunk` input
- `--phase`: `all`, `warmup`, or `online`
- `--source`: `all`, `base`, `rl`, `human`, or `mixed`
- `--output-dir`: training output directory

The output directory contains snapshots, checkpoints, metrics, config metadata,
and, when requested, an online-compatible bundle.

To continue online from an exported bundle, launch Machine B with the bundle
config:

```bash
python launch/launch_machine_b.py \
  --config runs/agilex_ethernet/offline_train_bcq/online_bundle/online_rl_config.yaml
```

### `eval_action_fit.py`

Compares actor outputs or recorded actions against replay references.

Typical uses:

- inspect how closely an actor follows `ref_chunk`
- inspect the recorded `action_chunk` vs `ref_chunk` gap
- compare deterministic actor mean with sampled actor outputs

Command template:

```bash
python scripts/offline/eval_action_fit.py \
  --replay-path runs/agilex_ethernet/replay/replay_journal.pkl \
  --model-dir runs/agilex_ethernet/offline_train_bcq \
  --actor-mode mean
```

Useful options:

- `--replay-path`: replay journal to analyze
- `--model-dir`: directory used to discover actor files
- `--actor-path`: explicit actor snapshot/checkpoint
- `--output-dir`: output directory for plots and tables
- `--actor-mode`: `mean` or `sample`
- `--recorded-action`: analyze replay actions instead of actor outputs
- `--phase`, `--source`: replay subset filters

This is a ref/action diagnostics tool. For human intervention samples, the
training BC target can be `action_chunk` even though the diagnostic reference is
still `ref_chunk`.

### `eval_episode_q.py`

Evaluates critic values over replay transitions or episode subsets.

Typical uses:

- inspect Q values across warmup and online data
- compare successful and failed transitions
- detect critic drift after offline training

Command template:

```bash
python scripts/offline/eval_episode_q.py \
  --replay-path runs/agilex_ethernet/replay/replay_journal.pkl \
  --model-dir runs/agilex_ethernet/offline_train_bcq \
  --episode-ids 12
```

Useful options:

- `--replay-path`: replay journal to analyze
- `--model-dir`: model/checkpoint directory
- `--episode-ids`: one or more ids or inclusive ranges such as `4 12 20-25`
- `--output-dir`: output directory
- `--phase`, `--source`: replay subset filters
- `--batch-size`: analysis batch size
- `--actor-mode`: `mean` or `sample`
- `--disable-ref-input`: evaluate actor Q with zeroed reference input

### `eval_intervention_risk.py`

Analyzes intervention-related replay subsets and action deviations.

Typical uses:

- compare human/mixed windows against policy windows
- inspect action deviation around intervention-heavy episodes
- identify candidate failure modes before restarting online training

### `visualize_offline_training.py`

Plots offline training metrics.

Command template:

```bash
python scripts/offline/visualize_offline_training.py \
  --train-dir runs/agilex_ethernet/offline_train_bcq
```

The script reads offline metrics and writes figures/tables under the offline
training directory unless `--output-dir` is provided.

## Practical Workflow

1. Inspect the replay journal.
2. Train offline from a chosen replay subset.
3. Run action-fit and Q diagnostics.
4. Choose a final or best actor snapshot.
5. Export or use the online bundle.
6. Resume online RL or run eval-only rollout.
