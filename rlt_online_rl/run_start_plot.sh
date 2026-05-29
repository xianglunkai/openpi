#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-runs/screw_sorting}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Run directory not found: ${RUN_DIR}"
  echo "Usage: bash run_start_plot.sh [run_dir]"
  exit 1
fi

for journal_path in \
  "${RUN_DIR}/replay/replay_journal_no_rl.pkl" \
  "${RUN_DIR}/replay/replay_journal.pkl" \
  "${RUN_DIR}/replay_journal_no_rl.pkl" \
  "${RUN_DIR}/replay_journal.pkl"
do
  if [[ -f "${journal_path}" ]]; then
    python scripts/tools/inspect_replay_journal.py "${journal_path}"
    break
  fi
done

python scripts/tools/plot_learner_metrics.py "${RUN_DIR}"