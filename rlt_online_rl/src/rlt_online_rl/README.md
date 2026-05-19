# `src/rlt_online_rl`

This directory contains the Python package for the online RL runtime. It holds
the algorithm, replay, learner, actor-service, and EnvDriver abstractions. ROS
adapters, launchers, keyboard clients, and task-level scripts live outside this
package.

Related outer directories:

- `../../train_deploy_alignment/`
- `../../launch/`
- `../../keyboard_*.py`
- `../../scripts/`
- `../../configs/`

## Modules

```text
rlt_online_rl/
|-- action_representation.py
|-- config.py
|-- inference.py
|-- networks.py
|-- replay.py
|-- runtime_logging.py
`-- trainer.py
```

## `action_representation.py`

Converts between the training action representation and executable absolute
action chunks.

Responsibilities:

- load quantile normalization statistics
- support `abs` and `delta_chunk` action representations
- normalize `ref_chunk`, `action_chunk`, and `next_ref_chunk` for training
- denormalize actor outputs back to robot-executable absolute chunks
- provide both NumPy and JAX helpers for training and inference

It does not own ROS publishing, replay storage, or reward logic.

## `config.py`

Defines the runtime configuration dataclasses and YAML loading/saving helpers.

Main dataclasses:

- `RLTOnlineRLConfig`
- `ActorServiceConfig`
- `LearnerServiceConfig`
- `ReplayConfig`
- `EnvDriverConfig`
- `MonitoringConfig`
- `OnlineRLSystemConfig`

It also resolves config-relative paths and can save portable configs with
project-relative paths.

## `networks.py`

Contains the JAX numerical core:

- `ChunkActor`
- `TwinCritic`
- target-Q construction
- reference-action dropout
- low-level actor/critic loss helpers

The actor consumes `z_rl`, `proprio`, and `ref_chunk`, then outputs a Gaussian
distribution over action chunks with a fixed standard deviation. The critic
uses twin Q networks and target networks.

Reference dropout is whole-chunk, sample-wise dropout applied to actor inputs
during training. It is not used during inference.

## `replay.py`

Defines replay data structures, the CPU replay buffer, journal persistence, and
HTTP client/server wrappers.

Key records:

- `RawEpisodeTrace`: episode-level raw observations, raw steps, and chunks
- `RLTTransition`: replay transition consumed by the learner
- `TransitionSource`: `BASE`, `RL`, `HUMAN`, `MIXED`

Replay stores both chunk-level `source` and per-step `source_chunk`. The learner
uses `source_chunk` to decide whether each step should imitate the executed
human action or the VLA reference.

The replay journal is append-only and is restored on startup.

## `trainer.py`

Implements the Machine B learner service.

Responsibilities:

- initialize actor, critic, target networks, and optimizers
- wait for warmup replay size
- train with a fixed update-to-data budget
- use different BC/Q weights before and after warmup readiness
- export actor snapshots for the actor service
- save and restore learner checkpoints
- write learner metrics and status JSON

The current actor loss is:

```text
actor_loss = bc_weight * bc_penalty - q_weight * actor_q + delta_weight * delta_penalty
```

For `HUMAN / MIXED` steps the BC target is the executed action. For other steps
the BC target is the Machine A / VLA reference.

## `inference.py`

Contains the runtime orchestration layer:

- actor service and actor client
- Machine A feature client
- feature payload validation
- EnvDriver
- chunk planning and fallback behavior
- raw episode collection
- episode-end replay window construction
- Machine A feature backfill

The ROS-specific robot adapter is not in this module. `EnvDriver` only assumes
an environment-like object with `reset`, `step`, or `execute_chunk`.

## `runtime_logging.py`

Defines run-directory layout and process logging helpers.

Typical runtime outputs:

- `runs/<task>/logs/`
- `runs/<task>/metrics/`
- `runs/<task>/checkpoints/`
- `runs/<task>/actor_snapshot/`
- `runs/<task>/replay/`

`actor_snapshot/actor_snapshot.pkl` is the latest snapshot read by
`actor_service`. The learner also writes
`actor_snapshot/history/actor_vXXXXXX.pkl` so rollout metrics can be mapped back
to exact actor parameters.

## Package Boundary

This package owns:

- algorithm code
- replay and learner code
- actor service/client code
- generic EnvDriver coordination
- runtime logging and config utilities

This package does not own:

- ROS topic names or robot-specific message conversion
- keyboard clients
- task-specific launch scripts
- openpi/RLT stage-1 training

## Suggested Reading Order

1. `config.py`
2. `replay.py`
3. `networks.py`
4. `trainer.py`
5. `inference.py`
6. `action_representation.py`
7. `../../train_deploy_alignment/pika_sync_ros.py`
8. `../../launch/launch_machine_b.py`
9. `../../launch/launch_robot_rollout.py`
10. `../../configs/tasks/agilex_ethernet/online_rl.yaml`

