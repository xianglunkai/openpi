#!/usr/bin/env bash
# Stop the tmux session started by run_start_eval.sh.
#
# Default: graceful shutdown (keyboard 'q' -> wait for rollout reset -> stop rest).
#   bash run_stop_eval.sh
#   Inside tmux: Ctrl+b then Q
#
# Immediate kill (skip graceful robot reset):
#   bash run_stop_eval.sh --force

set -euo pipefail

SESSION_NAME="${RLT_EVAL_TMUX_SESSION:-rlt_eval}"
GRACEFUL=1
ROLLOUT_WAIT_SEC="${RLT_ROLLOUT_SHUTDOWN_WAIT_SEC:-90}"
STEP_WAIT_SEC="${RLT_STEP_SHUTDOWN_WAIT_SEC:-2}"

usage() {
  cat <<EOF
Usage: bash run_stop_eval.sh [options]

Options:
  --force           Kill tmux session immediately (no graceful rollout shutdown)
  -h, --help        Show this help

Graceful shutdown (default):
  1. Send 'q' to the keyboard window (requests rollout stop + robot reset)
  2. Wait for rollout to finish
  3. Ctrl+C remaining windows, then kill the tmux session

From inside tmux: Ctrl+b then Q
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      GRACEFUL=0
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
  echo "tmux not found." >&2
  exit 1
fi

if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "No tmux session named '${SESSION_NAME}'."
  exit 0
fi

_has_window() {
  tmux list-windows -t "${SESSION_NAME}" -F '#{window_name}' 2>/dev/null | grep -qx "$1"
}

_send_keys() {
  local window=$1
  shift
  tmux send-keys -t "${SESSION_NAME}:${window}" "$@"
}

_pane_main_command() {
  local window=$1
  tmux list-panes -t "${SESSION_NAME}:${window}" -F '#{pane_current_command}' 2>/dev/null | head -1 || true
}

_window_running_python() {
  local cmd
  cmd="$(_pane_main_command "$1")"
  [[ "${cmd}" == python* ]]
}

_interrupt_window() {
  local window=$1
  if ! _has_window "${window}"; then
    return 0
  fi
  if _window_running_python "${window}"; then
    echo "Stopping ${window} (Ctrl+C)..."
    _send_keys "${window}" C-c
    sleep "${STEP_WAIT_SEC}"
  fi
}

_wait_for_rollout() {
  local waited=0
  echo "Waiting for rollout to reset and exit (up to ${ROLLOUT_WAIT_SEC}s)..."
  while (( waited < ROLLOUT_WAIT_SEC )); do
    if ! _has_window rollout; then
      echo "Rollout window exited."
      return 0
    fi
    if ! _window_running_python rollout; then
      echo "Rollout process exited."
      return 0
    fi
    sleep 2
    waited=$((waited + 2))
    if (( waited % 10 == 0 )); then
      echo "  still waiting (${waited}s)..."
    fi
  done
  echo "Rollout shutdown timed out; will force-stop remaining windows."
}

if [[ "${GRACEFUL}" -eq 1 ]]; then
  echo "Graceful shutdown of tmux session: ${SESSION_NAME}"
  if _has_window keyboard && _window_running_python keyboard; then
    echo "Sending 'q' to keyboard (rollout shutdown + quit)..."
    _send_keys keyboard q
    sleep 1
  fi
  if _has_window rollout; then
    _wait_for_rollout
  fi
  for window in machine-a actor; do
    _interrupt_window "${window}"
  done
else
  echo "Force-stopping tmux session: ${SESSION_NAME}"
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  tmux kill-session -t "${SESSION_NAME}"
fi
echo "Stopped tmux session: ${SESSION_NAME}"
