#!/usr/bin/env bash
# Offline BCQ pipeline for screw_sorting: train -> eval action fit -> visualize.
#
# Default hyperparameters mirror configs/tasks/screw_sorting/online_rl.yaml.
# The exported online bundle (checkpoints/online_rl_config.yaml) is built from
# that task yaml with updated rl.* fields and bundle-local runtime paths.
#
# Usage:
#   bash run_eval_action_fit.sh              # train + eval + viz (default)
#   bash run_eval_action_fit.sh train
#   bash run_eval_action_fit.sh eval
#   bash run_eval_action_fit.sh viz
#   bash run_eval_action_fit.sh eval-q --episode-ids 1 2 3
#
# Examples:
#   RLT_SOURCE=base bash run_eval_action_fit.sh all
#   RLT_DISABLE_REF_INPUT=1 bash run_eval_action_fit.sh train
#   RLT_CRITIC_LOSS_MODE=cql bash run_eval_action_fit.sh train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Task paths (runtime.replay / runtime.learner_service in online_rl.yaml) ──
TASK="${RLT_TASK:-screw_sorting}"
TASK_DIR="${RLT_TASK_DIR:-runs/${TASK}}"
REPLAY_PATH="${RLT_REPLAY_PATH:-${TASK_DIR}/replay/replay_journal.pkl}"
TASK_CONFIG="${RLT_TASK_CONFIG:-configs/tasks/${TASK}/online_rl.yaml}"

# ── Replay filters ────────────────────────────────────────────────────────────
PHASE="${RLT_PHASE:-warmup}"          # all | warmup | online | unknown
SOURCE="${RLT_SOURCE:-all}"           # all | base | rl | human | mixed

# ── BCQ / RL hyperparameters (experiment.rl in online_rl.yaml) ───────────────
STEPS="${RLT_STEPS:-20000}"                              # warmup_post_collect_updates
BATCH_SIZE="${RLT_BATCH_SIZE:-128}"                      # runtime.learner_service.sample_batch_size
SEED="${RLT_SEED:-0}"                                    # runtime.replay.seed
FIXED_STD="${RLT_FIXED_STD:-0.001}"                      # experiment.rl.fixed_std
REFERENCE_DROPOUT_PROB="${RLT_REFERENCE_DROPOUT_PROB:-0.5}"  # experiment.rl.reference_dropout_prob
DELTA_WEIGHT="${RLT_DELTA_WEIGHT:-10.0}"                 # experiment.rl.delta_weight
ACTOR_HIDDEN_DIM="${RLT_ACTOR_HIDDEN_DIM:-256}"          # experiment.rl.actor_hidden_dim
ACTOR_NUM_LAYERS="${RLT_ACTOR_NUM_LAYERS:-2}"            # experiment.rl.actor_num_layers
CRITIC_HIDDEN_DIM="${RLT_CRITIC_HIDDEN_DIM:-256}"        # experiment.rl.critic_hidden_dim
CRITIC_NUM_LAYERS="${RLT_CRITIC_NUM_LAYERS:-2}"          # experiment.rl.critic_num_layers
EVAL_EVERY="${RLT_EVAL_EVERY:-500}"
VAL_RATIO="${RLT_VAL_RATIO:-0.05}"
DISABLE_REF_INPUT="${RLT_DISABLE_REF_INPUT:-0}"          # 1 = add --disable-ref-input

# BC/Q weights follow warmup vs online fields in task yaml (both 10.0 / 0.1 for screw_sorting).
WARMUP_BC_WEIGHT="${RLT_WARMUP_BC_WEIGHT:-30.0}"         # experiment.rl.warmup_bc_weight
WARMUP_Q_WEIGHT="${RLT_WARMUP_Q_WEIGHT:-0.1}"            # experiment.rl.warmup_q_weight
ONLINE_BC_WEIGHT="${RLT_ONLINE_BC_WEIGHT:-10.0}"         # experiment.rl.online_bc_weight
ONLINE_Q_WEIGHT="${RLT_ONLINE_Q_WEIGHT:-0.1}"            # experiment.rl.online_q_weight

# Critic objective (experiment.rl.critic_loss_mode; default td matches config.py)
CRITIC_LOSS_MODE="${RLT_CRITIC_LOSS_MODE:-cql}"           # td | cql
CQL_ALPHA="${RLT_CQL_ALPHA:-0.01}"                        # experiment.rl.cql_alpha
CQL_N_ACTIONS="${RLT_CQL_N_ACTIONS:-5}"                  # experiment.rl.cql_n_actions
CQL_TEMP="${RLT_CQL_TEMP:-1.0}"                          # experiment.rl.cql_temp

# ── Eval action fit ───────────────────────────────────────────────────────────
ACTOR_MODE="${RLT_ACTOR_MODE:-mean}"                     # runtime.env_driver.actor_deterministic=true -> mean
ACTOR_SEED="${RLT_ACTOR_SEED:-0}"
COMPARE_TARGET="${RLT_COMPARE_TARGET:-snapshot}"          # snapshot | recorded-action
TOP_K="${RLT_TOP_K:-5}"
EVAL_DISABLE_REF_INPUT="${RLT_EVAL_DISABLE_REF_INPUT:-0}"

# ── Eval episode Q (optional subcommand) ──────────────────────────────────────
EPISODE_IDS=()
EVAL_Q_BATCH_SIZE="${RLT_EVAL_Q_BATCH_SIZE:-256}"

# ── Resolve output / model dir (mirrors offline_train_from_replay.py) ─────────
_filter_suffix() {
  local parts=()
  [[ "${PHASE}" != "all" ]] && parts+=("phase-${PHASE}")
  [[ "${SOURCE}" != "all" ]] && parts+=("source-${SOURCE}")
  if ((${#parts[@]} == 0)); then
    echo ""
  else
    local IFS=_
    echo "_${parts[*]}"
  fi
}

_default_train_dir() {
  local name="offline_train_bcq"
  [[ "${CRITIC_LOSS_MODE}" == "cql" ]] && name+="_cql"
  [[ "${DISABLE_REF_INPUT}" == "1" ]] && name+="_noref"
  name+="$(_filter_suffix)"
  echo "${TASK_DIR}/${name}"
}

TRAIN_DIR="${RLT_TRAIN_DIR:-$(_default_train_dir)}"
MODEL_DIR="${RLT_MODEL_DIR:-${TRAIN_DIR}}"

_resolve_bc_q_weights() {
  case "${PHASE}" in
    online)
      BC_WEIGHT="${ONLINE_BC_WEIGHT}"
      Q_WEIGHT="${ONLINE_Q_WEIGHT}"
      ;;
    *)
      BC_WEIGHT="${WARMUP_BC_WEIGHT}"
      Q_WEIGHT="${WARMUP_Q_WEIGHT}"
      ;;
  esac
}

usage() {
  cat <<EOF
Usage: bash run_eval_action_fit.sh [command] [options]

Offline BCQ pipeline aligned with ${TASK_CONFIG}.

Commands:
  all       Train + eval action fit + visualize (default)
  train     offline_train_from_replay.py only
  eval      eval_action_fit.py only
  viz       visualize_offline_training.py only
  eval-q    eval_episode_q.py (requires --episode-ids)

Options:
  -h, --help              Show this help

eval-q options:
  --episode-ids ID ...    Episode ids or ranges, e.g. --episode-ids 1 2 10-15

Key environment overrides:
  RLT_TASK / RLT_TASK_DIR / RLT_REPLAY_PATH / RLT_TASK_CONFIG
  RLT_PHASE / RLT_SOURCE
  RLT_BC_WEIGHT is not used; weights follow RLT_WARMUP_* or RLT_ONLINE_* by phase
  RLT_FIXED_STD / RLT_REFERENCE_DROPOUT_PROB / RLT_DELTA_WEIGHT
  RLT_CRITIC_LOSS_MODE=td|cql / RLT_CQL_ALPHA / RLT_CQL_N_ACTIONS / RLT_CQL_TEMP
  RLT_DISABLE_REF_INPUT=1
  RLT_TRAIN_DIR / RLT_MODEL_DIR
  RLT_ACTOR_MODE / RLT_COMPARE_TARGET

Resolved paths:
  task config: ${TASK_CONFIG}
  replay:      ${REPLAY_PATH}
  train_dir:   ${TRAIN_DIR}
  model_dir:   ${MODEL_DIR}
EOF
}

_ref_input_train_args=()
[[ "${DISABLE_REF_INPUT}" == "1" ]] && _ref_input_train_args+=(--disable-ref-input)

_ref_input_eval_args=()
[[ "${EVAL_DISABLE_REF_INPUT}" == "1" ]] && _ref_input_eval_args+=(--disable-ref-input)

_source_args=()
[[ "${SOURCE}" != "all" ]] && _source_args+=(--source "${SOURCE}")

cmd_train() {
  _resolve_bc_q_weights
  echo "==> BCQ offline train"
  echo "    task=${TASK} phase=${PHASE} source=${SOURCE}"
  echo "    bc_weight=${BC_WEIGHT} q_weight=${Q_WEIGHT} delta_weight=${DELTA_WEIGHT}"
  echo "    fixed_std=${FIXED_STD} reference_dropout_prob=${REFERENCE_DROPOUT_PROB}"
  echo "    critic_loss_mode=${CRITIC_LOSS_MODE} cql_alpha=${CQL_ALPHA}"
  echo "    output=${TRAIN_DIR}"
  python scripts/offline/offline_train_from_replay.py \
    --replay-path "${REPLAY_PATH}" \
    --output-dir "${TRAIN_DIR}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --bc-weight "${BC_WEIGHT}" \
    --q-weight "${Q_WEIGHT}" \
    --delta-weight "${DELTA_WEIGHT}" \
    --fixed-std "${FIXED_STD}" \
    --reference-dropout-prob "${REFERENCE_DROPOUT_PROB}" \
    --critic-loss-mode "${CRITIC_LOSS_MODE}" \
    --cql-alpha "${CQL_ALPHA}" \
    --cql-n-actions "${CQL_N_ACTIONS}" \
    --cql-temp "${CQL_TEMP}" \
    --actor-hidden-dim "${ACTOR_HIDDEN_DIM}" \
    --actor-num-layers "${ACTOR_NUM_LAYERS}" \
    --critic-hidden-dim "${CRITIC_HIDDEN_DIM}" \
    --critic-num-layers "${CRITIC_NUM_LAYERS}" \
    --eval-every "${EVAL_EVERY}" \
    --val-ratio "${VAL_RATIO}" \
    --phase "${PHASE}" \
    "${_source_args[@]}" \
    "${_ref_input_train_args[@]}"
}

cmd_eval() {
  echo "==> Action fit eval: compare_target=${COMPARE_TARGET} actor_mode=${ACTOR_MODE}"
  python scripts/offline/eval_action_fit.py \
    --replay-path "${REPLAY_PATH}" \
    --model-dir "${MODEL_DIR}" \
    --compare-target "${COMPARE_TARGET}" \
    --actor-mode "${ACTOR_MODE}" \
    --actor-seed "${ACTOR_SEED}" \
    --top-k "${TOP_K}" \
    --phase "${PHASE}" \
    "${_source_args[@]}" \
    "${_ref_input_eval_args[@]}"
}

cmd_viz() {
  echo "==> Visualize offline training: ${TRAIN_DIR}"
  python scripts/offline/visualize_offline_training.py \
    --train-dir "${TRAIN_DIR}"
}

cmd_eval_q() {
  if ((${#EPISODE_IDS[@]} == 0)); then
    echo "eval-q requires --episode-ids, e.g.: bash run_eval_action_fit.sh eval-q --episode-ids 1 2 3" >&2
    exit 1
  fi
  echo "==> Episode Q eval: episodes=${EPISODE_IDS[*]}"
  python scripts/offline/eval_episode_q.py \
    --replay-path "${REPLAY_PATH}" \
    --model-dir "${MODEL_DIR}" \
    --episode-ids "${EPISODE_IDS[@]}" \
    --batch-size "${EVAL_Q_BATCH_SIZE}" \
    --actor-mode "${ACTOR_MODE}" \
    --actor-seed "${ACTOR_SEED}" \
    --phase "${PHASE}" \
    "${_source_args[@]}" \
    "${_ref_input_eval_args[@]}"
}

# ── Parse command + eval-q args ───────────────────────────────────────────────
COMMAND="${1:-all}"
shift || true

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --episode-ids)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        EPISODE_IDS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${COMMAND}" in
  all)
    cmd_train
    cmd_eval
    cmd_viz
    ;;
  train)  cmd_train ;;
  eval)   cmd_eval ;;
  viz)    cmd_viz ;;
  eval-q) cmd_eval_q ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown command: ${COMMAND}" >&2
    usage
    exit 1
    ;;
esac
