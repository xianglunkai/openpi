#!/usr/bin/env bash
# Stop the tmux session started by run_start_training.sh

set -euo pipefail

SESSION_NAME="${RLT_TMUX_SESSION:-rlt_training}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found." >&2
  exit 1
fi

if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "No tmux session named '${SESSION_NAME}'."
  exit 0
fi

tmux kill-session -t "${SESSION_NAME}"
echo "Stopped tmux session: ${SESSION_NAME}"
