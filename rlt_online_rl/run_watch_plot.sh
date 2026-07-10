#!/usr/bin/env bash
# Local live-ish learner metrics plotter for online training (no network required).
#
# Periodically refreshes runs/<task>/plots/learner_curves.png from
# metrics/learner_metrics.jsonl. Open the png in an image viewer that auto-reloads.
#
# Usage:
#   bash run_watch_plot.sh                      # default runs/screw_sorting, 30s interval
#   bash run_watch_plot.sh runs/screw_sorting
#   RLT_PLOT_INTERVAL=15 bash run_watch_plot.sh
#   bash run_watch_plot.sh --once               # single refresh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_DIR="${RLT_RUN_DIR:-runs/screw_sorting}"
INTERVAL="${RLT_PLOT_INTERVAL:-30}"
ONCE=0

usage() {
  cat <<EOF
Usage: bash run_watch_plot.sh [options] [run_dir]

Local offline-friendly monitoring: refresh learner_curves.png every ${INTERVAL}s.

Options:
  --once                  Plot once and exit
  -h, --help              Show this help

Environment:
  RLT_RUN_DIR             Default run directory (default: runs/screw_sorting)
  RLT_PLOT_INTERVAL       Refresh interval in seconds (default: 30)

Output:
  <run_dir>/plots/learner_curves.png

Tip: open the png with an image viewer that reloads on file change, e.g.:
  eog runs/screw_sorting/plots/learner_curves.png &
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --once)
      ONCE=1
      shift
      ;;
    *)
      RUN_DIR="$1"
      shift
      ;;
  esac
done

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi
RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

METRICS_PATH="${RUN_DIR}/metrics/learner_metrics.jsonl"
STATUS_PATH="${RUN_DIR}/metrics/learner_status.json"
OUTPUT_PATH="${RUN_DIR}/plots/learner_curves.png"
PLOT_SCRIPT="${SCRIPT_DIR}/scripts/tools/plot_learner_metrics.py"

_plot_once() {
  python3 "${PLOT_SCRIPT}" "${RUN_DIR}" --latest-run-only --output "${OUTPUT_PATH}"
}

_print_status() {
  if [[ ! -f "${STATUS_PATH}" ]]; then
    echo "  status: (waiting for ${STATUS_PATH})"
    return
  fi
  python3 - <<PY
import json
from pathlib import Path
d = json.loads(Path("${STATUS_PATH}").read_text())
print(
    f"  step={d.get('global_step', '?')}  replay={d.get('replay_size', '?')}  "
    f"ready_online={d.get('ready_for_online', '?')}  actor_ver={d.get('actor_version', '?')}"
)
PY
}

echo "Watching learner metrics (local, no network):"
echo "  run_dir:  ${RUN_DIR}"
echo "  metrics:  ${METRICS_PATH}"
echo "  output:   ${OUTPUT_PATH}"
echo "  interval: ${INTERVAL}s"
echo

if [[ "${ONCE}" -eq 1 ]]; then
  if [[ ! -f "${METRICS_PATH}" ]]; then
    echo "Metrics file not found yet: ${METRICS_PATH}" >&2
    exit 1
  fi
  _plot_once
  _print_status
  exit 0
fi

echo "Waiting for metrics file..."
while [[ ! -f "${METRICS_PATH}" ]]; do
  sleep 2
done

while true; do
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] refresh"
  if _plot_once; then
    _print_status
  else
    echo "  plot failed (metrics may be empty); retrying..."
  fi
  sleep "${INTERVAL}"
done
