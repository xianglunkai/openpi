#!/usr/bin/env bash
# HTTP bridge for VLA-Precision Stage-II / evaluation.
# Runs inside the openpi mobile_aloha environment (ROS required here, not in VLA-P).
#
# Prerequisite: normal Mobile ALOHA ROS bringup already running.
# Do NOT start serve_policy / run_client / main.py for VLA-Precision.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5001}"
SINGLE_ARM="${SINGLE_ARM:-right}"
CONTROL_FREQ_HZ="${CONTROL_FREQ_HZ:-30}"

source examples/mobile_aloha_AgileX/.venv/bin/activate

# Flask is only needed by this bridge; install once if missing.
python -c "import flask" 2>/dev/null || pip install "flask==3.1.3"

python -m examples.mobile_aloha_AgileX.vla_precision_bridge \
  --host "${HOST}" \
  --port "${PORT}" \
  --use-single-arm \
  --single-arm "${SINGLE_ARM}" \
  --control-freq-hz "${CONTROL_FREQ_HZ}"
