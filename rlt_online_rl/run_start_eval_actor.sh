#!/usr/bin/env bash
# Start actor_service only (no learner / replay). Used for eval rollout.
#
# Override defaults:
#   RLT_EVAL_CONFIG=configs/tasks/screw_sorting/online_rl.yaml
#   RLT_ACTOR_SNAPSHOT=runs/screw_sorting/best_actor_snapshot.pkl

CONFIG="${RLT_EVAL_CONFIG:-/home/xlk/work/openpi/rlt_online_rl/runs/screw_sorting/good_cpt_20261013-1348/checkpoints/online_rl_config.yaml}"
ACTOR_SNAPSHOT="${RLT_ACTOR_SNAPSHOT:-runs/screw_sorting/good_cpt_20261013-1348/actor_snapshot/actor_snapshot.pkl}"

python scripts/run_online_rl.py \
  --config "${CONFIG}" \
  --system.role actor_service \
  --system.actor_service.snapshot_path "${ACTOR_SNAPSHOT}"