#!/usr/bin/env bash
# One-shot learner metrics plot (latest run only).
#
# Usage:
#   bash run_start_plot.sh
#   bash run_start_plot.sh runs/screw_sorting
#
# For live refresh during training, use:
#   bash run_watch_plot.sh runs/screw_sorting

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RUN_DIR="${1:-${RLT_RUN_DIR:-runs/screw_sorting}}"

python3 scripts/tools/plot_learner_metrics.py \
  "${RUN_DIR}" \
  --latest-run-only
