#!/usr/bin/env bash
# Offline BCQ pipeline: train -> eval action fit -> visualize.
#
# Hyperparameters come from the shared task YAML (same as online):
#   configs/tasks/<task>/online_rl.yaml
# Only set RLT_* when you intentionally override that file.
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
#   RLT_STEPS=5000 bash run_eval_action_fit.sh train

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Paths (must match runtime.* in the task YAML) ─────────────────────────────
TASK="${RLT_TASK:-screw_sorting}"
TASK_DIR="${RLT_TASK_DIR:-runs/${TASK}}"
REPLAY_PATH="${RLT_REPLAY_PATH:-${TASK_DIR}/replay/replay_journal_sft.pkl}"
TASK_CONFIG="${RLT_TASK_CONFIG:-configs/tasks/${TASK}/online_rl.yaml}"

# ── Replay filters / output naming ────────────────────────────────────────────
PHASE="${RLT_PHASE:-all}"          # all | warmup | online | unknown
SOURCE="${RLT_SOURCE:-all}"           # all | base | rl | human | mixed
DISABLE_REF_INPUT="${RLT_DISABLE_REF_INPUT:-0}"

# ── Eval-only knobs (not in task YAML) ────────────────────────────────────────
EVAL_EVERY="${RLT_EVAL_EVERY:-500}"
VAL_RATIO="${RLT_VAL_RATIO:-0.05}"
ACTOR_MODE="${RLT_ACTOR_MODE:-mean}"
ACTOR_SEED="${RLT_ACTOR_SEED:-0}"
COMPARE_TARGET="${RLT_COMPARE_TARGET:-snapshot}"
TOP_K="${RLT_TOP_K:-5}"
EVAL_DISABLE_REF_INPUT="${RLT_EVAL_DISABLE_REF_INPUT:-0}"
EVAL_Q_BATCH_SIZE="${RLT_EVAL_Q_BATCH_SIZE:-256}"
# Optional LPF overrides for smooth eval (unset → TASK_CONFIG env_driver.*)
# RLT_LPF_CUTOFF_FREQ / RLT_LPF_DT / RLT_SIMULATE_LPF=0|1
EPISODE_IDS=()

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
  [[ "${DISABLE_REF_INPUT}" == "1" ]] && name+="_noref"
  name+="$(_filter_suffix)"
  echo "${TASK_DIR}/${name}"
}

TRAIN_DIR="${RLT_TRAIN_DIR:-$(_default_train_dir)}"
MODEL_DIR="${RLT_MODEL_DIR:-${TRAIN_DIR}}"

usage() {
  cat <<EOF
Usage: bash run_eval_action_fit.sh [command] [options]

Offline BCQ pipeline. Training hyperparams: ${TASK_CONFIG}

Commands:
  all       Train + eval action fit + visualize (default)
  train     offline_train_from_replay.py only
  eval      eval_action_fit.py only
  viz       visualize_offline_training.py only
  smooth    eval_action_smoothness.py (intra/inter-chunk + vs gt plots)
  eval-q    eval_episode_q.py (requires --episode-ids)

Options:
  -h, --help              Show this help
  --episode-ids ID ...    For eval-q

Path / filter env:
  RLT_TASK / RLT_TASK_DIR / RLT_REPLAY_PATH / RLT_TASK_CONFIG
  RLT_PHASE / RLT_SOURCE / RLT_DISABLE_REF_INPUT=1
  RLT_TRAIN_DIR / RLT_MODEL_DIR

Optional train overrides (unset = YAML):
  RLT_STEPS / RLT_BATCH_SIZE / RLT_SEED / RLT_DELTA_WEIGHT
  RLT_FIXED_STD / RLT_REFERENCE_DROPOUT_PROB
  RLT_ACTOR_HIDDEN_DIM / RLT_ACTOR_NUM_LAYERS
  RLT_CRITIC_HIDDEN_DIM / RLT_CRITIC_NUM_LAYERS
  RLT_WARMUP_BC_WEIGHT / RLT_WARMUP_Q_WEIGHT
  RLT_ONLINE_BC_WEIGHT / RLT_ONLINE_Q_WEIGHT

Eval env:
  RLT_ACTOR_MODE / RLT_COMPARE_TARGET / RLT_EVAL_EVERY / RLT_VAL_RATIO
  RLT_LPF_CUTOFF_FREQ / RLT_LPF_DT / RLT_SIMULATE_LPF=0|1
  (smooth; default follows YAML use_actions_filter; RLT_SIMULATE_LPF overrides)

Deploy LPF (in ${TASK_CONFIG} runtime.env_driver):
  use_actions_filter / action_lpf_cutoff_freq / action_lpf_dt

Resolved:
  config=${TASK_CONFIG}
  replay=${REPLAY_PATH}
  train_dir=${TRAIN_DIR}
EOF
}

_ref_input_train_args=()
[[ "${DISABLE_REF_INPUT}" == "1" ]] && _ref_input_train_args+=(--disable-ref-input)

_ref_input_eval_args=()
[[ "${EVAL_DISABLE_REF_INPUT}" == "1" ]] && _ref_input_eval_args+=(--disable-ref-input)

_source_args=()
[[ "${SOURCE}" != "all" ]] && _source_args+=(--source "${SOURCE}")

cmd_train() {
  echo "==> BCQ offline train (config=${TASK_CONFIG})"
  echo "    task=${TASK} phase=${PHASE} source=${SOURCE} output=${TRAIN_DIR}"
  local train_args=(
    --replay-path "${REPLAY_PATH}"
    --config "${TASK_CONFIG}"
    --output-dir "${TRAIN_DIR}"
    --phase "${PHASE}"
    --eval-every "${EVAL_EVERY}"
    --val-ratio "${VAL_RATIO}"
  )
  [[ -n "${RLT_STEPS:-}" ]] && train_args+=(--steps "${RLT_STEPS}")
  [[ -n "${RLT_BATCH_SIZE:-}" ]] && train_args+=(--batch-size "${RLT_BATCH_SIZE}")
  [[ -n "${RLT_SEED:-}" ]] && train_args+=(--seed "${RLT_SEED}")
  [[ -n "${RLT_DELTA_WEIGHT:-}" ]] && train_args+=(--delta-weight "${RLT_DELTA_WEIGHT}")
  [[ -n "${RLT_FIXED_STD:-}" ]] && train_args+=(--fixed-std "${RLT_FIXED_STD}")
  [[ -n "${RLT_REFERENCE_DROPOUT_PROB:-}" ]] && train_args+=(--reference-dropout-prob "${RLT_REFERENCE_DROPOUT_PROB}")
  [[ -n "${RLT_ACTOR_HIDDEN_DIM:-}" ]] && train_args+=(--actor-hidden-dim "${RLT_ACTOR_HIDDEN_DIM}")
  [[ -n "${RLT_ACTOR_NUM_LAYERS:-}" ]] && train_args+=(--actor-num-layers "${RLT_ACTOR_NUM_LAYERS}")
  [[ -n "${RLT_CRITIC_HIDDEN_DIM:-}" ]] && train_args+=(--critic-hidden-dim "${RLT_CRITIC_HIDDEN_DIM}")
  [[ -n "${RLT_CRITIC_NUM_LAYERS:-}" ]] && train_args+=(--critic-num-layers "${RLT_CRITIC_NUM_LAYERS}")
  if [[ "${PHASE}" == "online" ]]; then
    [[ -n "${RLT_ONLINE_BC_WEIGHT:-}" ]] && train_args+=(--bc-weight "${RLT_ONLINE_BC_WEIGHT}")
    [[ -n "${RLT_ONLINE_Q_WEIGHT:-}" ]] && train_args+=(--q-weight "${RLT_ONLINE_Q_WEIGHT}")
  else
    [[ -n "${RLT_WARMUP_BC_WEIGHT:-}" ]] && train_args+=(--bc-weight "${RLT_WARMUP_BC_WEIGHT}")
    [[ -n "${RLT_WARMUP_Q_WEIGHT:-}" ]] && train_args+=(--q-weight "${RLT_WARMUP_Q_WEIGHT}")
  fi
  # Allow generic overrides regardless of phase naming.
  [[ -n "${RLT_BC_WEIGHT:-}" ]] && train_args+=(--bc-weight "${RLT_BC_WEIGHT}")
  [[ -n "${RLT_Q_WEIGHT:-}" ]] && train_args+=(--q-weight "${RLT_Q_WEIGHT}")
  train_args+=("${_source_args[@]}" "${_ref_input_train_args[@]}")
  python scripts/offline/offline_train_from_replay.py "${train_args[@]}"
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

cmd_smooth() {
  echo "==> Action smoothness eval: actor_mode=${ACTOR_MODE} model=${MODEL_DIR} config=${TASK_CONFIG}"
  local smooth_args=(
    --replay-path "${REPLAY_PATH}"
    --model-dir "${MODEL_DIR}"
    --config "${TASK_CONFIG}"
    --actor-mode "${ACTOR_MODE}"
    --actor-seed "${ACTOR_SEED}"
    --phase "${PHASE}"
  )
  # Default: follow YAML use_actions_filter. Override with RLT_SIMULATE_LPF=0|1.
  if [[ -n "${RLT_SIMULATE_LPF:-}" ]]; then
    if [[ "${RLT_SIMULATE_LPF}" == "0" ]]; then
      smooth_args+=(--no-simulate-lpf)
    else
      smooth_args+=(--simulate-lpf)
    fi
  fi
  [[ -n "${RLT_LPF_CUTOFF_FREQ:-}" ]] && smooth_args+=(--lpf-cutoff-freq "${RLT_LPF_CUTOFF_FREQ}")
  [[ -n "${RLT_LPF_DT:-}" ]] && smooth_args+=(--lpf-dt "${RLT_LPF_DT}")
  smooth_args+=("${_source_args[@]}" "${_ref_input_eval_args[@]}")
  python scripts/offline/eval_action_smoothness.py "${smooth_args[@]}"
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
  smooth) cmd_smooth ;;
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
