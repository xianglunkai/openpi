#!/usr/bin/env bash
# Launch eval stack in one tmux session (actor inference + rollout only; no learner/replay).
#
# Per README "Launching Eval":
#   1. actor_service (loads actor snapshot, serves refined chunks)
#   2. eval rollout (deterministic actor mean, no replay writes)
#   3. eval keyboard (a/b = actor/base, o = start, c = critical, s = end)
#
# Machine A (RLT policy server) is required for rollout but not started here by default.
# Start it separately: bash run_start_machineA.sh
#
# Usage:
#   bash run_start_eval.sh
#   bash run_start_eval.sh --attach
#   RLT_ACTOR_SNAPSHOT=runs/screw_sorting/actor_snapshot/actor_snapshot.pkl bash run_start_eval.sh
#
# Inside tmux:
#   Ctrl+b then window number   switch window
#   Ctrl+b d                      detach (services keep running)
#   bash run_stop_eval.sh         stop everything

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION_NAME="${RLT_EVAL_TMUX_SESSION:-rlt_eval}"
CONDA_ENV="${RLT_CONDA_ENV:-rl-tokens}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ATTACH_ONLY=0

usage() {
  cat <<EOF
Usage: bash run_start_eval.sh [options]

Options:
  --attach          Attach to existing tmux session "${SESSION_NAME}"
  -h, --help        Show this help

Environment overrides:
  RLT_EVAL_CONFIG       Config YAML (default: configs/tasks/screw_sorting/online_rl.yaml)
  RLT_ACTOR_SNAPSHOT    Actor snapshot for eval (default: runs/screw_sorting/best_actor_snapshot.pkl)
  RLT_MACHINE_A_WS_URL  Machine A websocket URL (default: ws://0.0.0.0:8000)

Windows created:
  actor      actor_service only (no learner/replay)     [conda rl-tokens]
  rollout    ROS1 eval rollout (--eval_actor_only)      [conda rl-tokens + ROS]
  keyboard   eval manual signals (default attach)       [conda rl-tokens + ROS]

Start Machine A separately before rollout:
  bash run_start_machineA.sh

Stop:
  bash run_stop_eval.sh
  Ctrl+b Q (inside tmux)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attach)
      ATTACH_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required to run all services in one terminal window." >&2
  echo "Install: sudo apt install tmux" >&2
  exit 1
fi

if [[ "${ATTACH_ONLY}" -eq 1 ]]; then
  exec tmux attach-session -t "${SESSION_NAME}"
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "tmux session '${SESSION_NAME}' already exists."
  echo "Attach:  tmux attach -t ${SESSION_NAME}"
  echo "Stop:    bash run_stop_eval.sh"
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "Conda init script not found: ${CONDA_SH}" >&2
  echo "Set CONDA_SH=/path/to/conda.sh if needed." >&2
  exit 1
fi

_run_rlt() {
  local cmd=$1
  cat <<EOF
cd '${SCRIPT_DIR}'
source '${CONDA_SH}'
conda activate '${CONDA_ENV}'
${cmd}
echo
echo '[${SESSION_NAME}] process exited with code '\$?
read -p 'Press Enter to close this window...'
EOF
}

# 0: actor (start first; rollout waits for actor_service HTTP)
tmux new-session -d -s "${SESSION_NAME}" -n actor "$(_run_rlt "bash run_start_eval_actor.sh")"

# 1: rollout (waits for actor, then starts eval-only robot adapter)
tmux new-window -t "${SESSION_NAME}" -n rollout "$(_run_rlt "bash run_start_eval_rollout.sh")"

# 2: keyboard (interactive; attach here)
tmux new-window -t "${SESSION_NAME}" -n keyboard "$(_run_rlt "bash run_start_eval_keyboard.sh")"

tmux bind-key -T prefix Q confirm-before -p \
  "Stop all RLT eval services?" \
  "run-shell \"bash '${SCRIPT_DIR}/run_stop_eval.sh'\""

tmux select-window -t "${SESSION_NAME}:keyboard"

cat <<EOF
Started tmux session: ${SESSION_NAME}
  window actor     : actor_service (eval snapshot)
  window rollout     : eval rollout (deterministic actor mean)
  window keyboard    : eval manual control (current)

Ensure Machine A is running before rollout:
  bash run_start_machineA.sh

Switch window: Ctrl+b then 0/1/2
Detach:        Ctrl+b d
Stop all:      Ctrl+b Q   (or: bash run_stop_eval.sh)
Force stop:    bash run_stop_eval.sh --force

Eval flow:
  1. Reset robot to start pose
  2. Press 'a' or 'b' to select actor/base for next critical phase
  3. Press 'o' to begin episode
  4. In full_task, press 'c' at critical-phase boundary
  5. Press 's' to end episode

Attaching to keyboard window...
EOF

exec tmux attach-session -t "${SESSION_NAME}"
