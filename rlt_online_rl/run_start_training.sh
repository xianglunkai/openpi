#!/usr/bin/env bash
# Launch the full online RL training stack in one tmux session (one terminal window).
#
# Each tmux window uses a different Python env on purpose:
#   machine-a   openpi uv (repo root, Python 3.11+) — policy server
#   machine-b   conda rl-tokens — learner / replay / actor
#   rollout     conda rl-tokens + ROS Noetic — robot adapter (cv_bridge)
#   spacemouse  conda rl-tokens + ROS + cmeel/placo libs
#   keyboard    conda rl-tokens + ROS
#
# Usage:
#   bash run_start_training.sh              # start all services + attach to keyboard
#   bash run_start_training.sh --no-spacemouse
#   bash run_start_training.sh --attach     # attach to an existing session
#
# Inside tmux:
#   Ctrl+b then window number   switch window (0=machine-b, 1=machine-a, ...)
#   Ctrl+b d                      detach (services keep running)
#   bash run_stop_training.sh     stop everything

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SESSION_NAME="${RLT_TMUX_SESSION:-rlt_training}"
CONDA_ENV="${RLT_CONDA_ENV:-rl-tokens}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ENABLE_SPACEMOUSE=1
ATTACH_ONLY=0

usage() {
  cat <<EOF
Usage: bash run_start_training.sh [options]

Options:
  --attach          Attach to existing tmux session "${SESSION_NAME}"
  --no-spacemouse   Do not start SpaceMouse teleop window
  -h, --help        Show this help

Windows created (env noted):
  machine-b   learner / replay / actor / wandb monitor  [conda rl-tokens]
  machine-a   RLT policy server (ws://0.0.0.0:8000)     [openpi uv]
  rollout     ROS1 robot rollout                        [conda rl-tokens + ROS]
  spacemouse  SpaceMouse teleop (optional)            [conda rl-tokens + ROS]
  keyboard    manual signals (default attach target)    [conda rl-tokens + ROS]

Plot metrics on demand (separate command):
  bash run_start_plot.sh [run_dir]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --attach)
      ATTACH_ONLY=1
      shift
      ;;
    --no-spacemouse)
      ENABLE_SPACEMOUSE=0
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
  echo "Stop:    bash run_stop_training.sh"
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

_run_machine_a() {
  cat <<EOF
cd '${OPENPI_ROOT}'
# Policy server: openpi uv env (py>=3.11). Do not inherit conda rl-tokens / ROS paths.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PYTHON_EXE CONDA_EXE VIRTUAL_ENV
if [[ -n "\${LD_LIBRARY_PATH:-}" ]]; then
  IFS=':' read -ra _ld_parts <<< "\${LD_LIBRARY_PATH}"
  _clean_ld=""
  for _entry in "\${_ld_parts[@]}"; do
    [[ -z "\${_entry}" ]] && continue
    [[ "\${_entry}" == *miniconda* || "\${_entry}" == *conda* || "\${_entry}" == *cmeel* || "\${_entry}" == /opt/ros/* ]] && continue
    _clean_ld="\${_clean_ld:+\${_clean_ld}:}\${_entry}"
  done
  export LD_LIBRARY_PATH="\${_clean_ld}"
fi
export HF_ENDPOINT=https://hf-mirror.com
bash '${SCRIPT_DIR}/run_start_machineA.sh'
echo
echo '[${SESSION_NAME}] machine-a exited with code '\$?
read -p 'Press Enter to close this window...'
EOF
}

# 0: machine-b (start first; rollout waits for its HTTP services)
tmux new-session -d -s "${SESSION_NAME}" -n machine-b "$(_run_rlt "bash run_start_machineB.sh")"

# 1: machine-a
tmux new-window -t "${SESSION_NAME}" -n machine-a "$(_run_machine_a)"

# 2: rollout (waits for actor/replay, then starts robot adapter)
tmux new-window -t "${SESSION_NAME}" -n rollout "$(_run_rlt "bash run_start_rollout.sh")"

# 3: spacemouse (optional)
if [[ "${ENABLE_SPACEMOUSE}" -eq 1 ]]; then
  tmux new-window -t "${SESSION_NAME}" -n spacemouse "$(_run_rlt "bash run_start_spacemouse.sh")"
fi

# last: keyboard (interactive; attach here)
tmux new-window -t "${SESSION_NAME}" -n keyboard "$(_run_rlt "bash run_start_keyborad.sh")"

tmux select-window -t "${SESSION_NAME}:keyboard"

cat <<EOF
Started tmux session: ${SESSION_NAME}
  window machine-b   : learner / replay / actor
  window machine-a   : policy server
  window rollout     : robot rollout
EOF
if [[ "${ENABLE_SPACEMOUSE}" -eq 1 ]]; then
  echo "  window spacemouse  : SpaceMouse teleop"
fi
cat <<EOF
  window keyboard    : manual control (current)

Switch window: Ctrl+b then 0/1/2/...
Detach:        Ctrl+b d
Stop all:      bash run_stop_training.sh
Plot:          bash run_start_plot.sh runs/screw_sorting

Attaching to keyboard window...
EOF

exec tmux attach-session -t "${SESSION_NAME}"
