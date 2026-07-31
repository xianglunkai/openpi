#!/usr/bin/env bash
# Start actor_service only (no learner / replay). Used for eval rollout.
#
# Override defaults:
#   RLT_EVAL_CONFIG=configs/tasks/screw_sorting/online_rl.yaml
#   RLT_ACTOR_SNAPSHOT=runs/screw_sorting/best_actor_snapshot.pkl

CONFIG="${RLT_EVAL_CONFIG:-configs/tasks/screw_sorting/online_rl.yaml}"
ACTOR_SNAPSHOT="${RLT_ACTOR_SNAPSHOT:-runs/screw_sorting/best_actor_snapshot.pkl}"

python scripts/run_online_rl.py \
  --config "${CONFIG}" \
  --system.role actor_service \
  --system.actor_service.snapshot_path "${ACTOR_SNAPSHOT}"
