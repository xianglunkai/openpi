#!/usr/bin/env bash
# Open-loop VLA vs RL comparison on a replay journal.
# Plots action, velocity, and acceleration for the current actor snapshot.
#
# Usage (from rlt_online_rl/, conda env rl-tokens):
#   bash run_eval_vla_vs_rl.sh
#   bash run_eval_vla_vs_rl.sh --episode-ids 21 12
#   RLT_MAX_SAMPLES=200 bash run_eval_vla_vs_rl.sh
#   RLT_ACTOR_PATH=runs/screw_sorting/offline_train_bcq/best_actor_snapshot.pkl bash run_eval_vla_vs_rl.sh
#
# Defaults:
#   replay  runs/<task>/replay/replay_journal_sft_chunk10.pkl
#   actor   runs/<task>/offline_train_bcq/actor_snapshot/actor_snapshot.pkl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK="${RLT_TASK:-screw_sorting}"
TASK_DIR="${RLT_TASK_DIR:-runs/${TASK}}"
REPLAY_PATH="${RLT_REPLAY_PATH:-${TASK_DIR}/replay/replay_journal_sft_chunk10.pkl}"
MODEL_DIR="${RLT_MODEL_DIR:-${TASK_DIR}/offline_train_bcq}"
PHASE="${RLT_PHASE:-all}"
SOURCE="${RLT_SOURCE:-all}"
ACTOR_MODE="${RLT_ACTOR_MODE:-mean}"
ACTOR_SEED="${RLT_ACTOR_SEED:-0}"
NUM_EPISODE_PLOTS="${RLT_NUM_EPISODE_PLOTS:-3}"

args=(
  --replay-path "${REPLAY_PATH}"
  --model-dir "${MODEL_DIR}"
  --phase "${PHASE}"
  --source "${SOURCE}"
  --actor-mode "${ACTOR_MODE}"
  --actor-seed "${ACTOR_SEED}"
  --num-episode-plots "${NUM_EPISODE_PLOTS}"
)
[[ -n "${RLT_ACTOR_PATH:-}" ]] && args+=(--actor-path "${RLT_ACTOR_PATH}")
[[ -n "${RLT_OUTPUT_DIR:-}" ]] && args+=(--output-dir "${RLT_OUTPUT_DIR}")
[[ -n "${RLT_MAX_SAMPLES:-}" ]] && args+=(--max-samples "${RLT_MAX_SAMPLES}")
[[ -n "${RLT_CONTROL_HZ:-}" ]] && args+=(--control-hz "${RLT_CONTROL_HZ}")
if [[ -n "${RLT_EPISODE_IDS:-}" ]]; then
  # shellcheck disable=SC2206
  episode_ids=(${RLT_EPISODE_IDS})
  args+=(--episode-ids "${episode_ids[@]}")
fi

echo "==> VLA vs RL  replay=${REPLAY_PATH}"
echo "    model=${MODEL_DIR} actor_mode=${ACTOR_MODE} phase=${PHASE} source=${SOURCE}"
python scripts/offline/eval_vla_vs_rl.py "${args[@]}" "$@"
