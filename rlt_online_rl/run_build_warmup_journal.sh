#!/usr/bin/env bash
# Build Evo v2 warmup replay journal from local LeRobot v2.1 demos + critical CSV.
# Uses a lightweight reader (meta + parquet via pyarrow). No HuggingFace lerobot package.
# Optional once: pip install 'rlt-online-rl[offline]'   # or: pip install pyarrow
#
# Usage (pick one or more steps):
#   ./run_build_warmup_journal.sh init
#   ./run_build_warmup_journal.sh list
#   ./run_build_warmup_journal.sh validate
#   ./run_build_warmup_journal.sh build
#   ./run_build_warmup_journal.sh list validate
#   ./run_build_warmup_journal.sh all
set -euo pipefail
cd "$(dirname "$0")"

DATASET_ROOT="${DATASET_ROOT:-/data/huggingface/lerobot/openpi/screw_sorting_single_rl}"
REPO_ID="${REPO_ID:-openpi/screw_sorting_single_rl}"
CRITICAL_CSV="${CRITICAL_CSV:-critical_segments.csv}"
OUTPUT_JOURNAL="${OUTPUT_JOURNAL:-runs/screw_sorting/replay/replay_journal_demo.pkl}"
MACHINE_A_URL="${MACHINE_A_URL:-ws://127.0.0.1:8000}"

usage() {
  cat <<EOF
Usage: $0 <step> [step ...]

Steps:
  init      Generate critical CSV from LeRobot (edit start/end after)
  list      Print episode frame counts
  validate  Check critical CSV against dataset (edit CSV first)
  build     Encode features and write warmup journal (needs Machine A)
  all       Run init → list → validate → build

Env overrides:
  DATASET_ROOT  CRITICAL_CSV  REPO_ID  OUTPUT_JOURNAL  MACHINE_A_URL
EOF
}

run_init() {
  echo "==> init: write $CRITICAL_CSV from $DATASET_ROOT"
  python scripts/offline/build_warmup_journal_from_lerobot.py init "$CRITICAL_CSV" \
    --dataset-root "$DATASET_ROOT" \
    --repo-id "$REPO_ID" \
    --overwrite
}

run_list() {
  echo "==> list: episode lengths in $DATASET_ROOT"
  python scripts/offline/build_warmup_journal_from_lerobot.py list \
    --dataset-root "$DATASET_ROOT" \
    --repo-id "$REPO_ID"
}

run_validate() {
  echo "==> validate: $CRITICAL_CSV"
  python scripts/offline/build_warmup_journal_from_lerobot.py validate \
    --critical-path "$CRITICAL_CSV" \
    --dataset-root "$DATASET_ROOT" \
    --repo-id "$REPO_ID"
}

run_build() {
  echo "==> build: $OUTPUT_JOURNAL (Machine A: $MACHINE_A_URL)"
  python scripts/offline/build_warmup_journal_from_lerobot.py build \
    --critical-path "$CRITICAL_CSV" \
    --dataset-root "$DATASET_ROOT" \
    --repo-id "$REPO_ID" \
    --output-journal "$OUTPUT_JOURNAL" \
    --machine-a-ws-url "$MACHINE_A_URL" \
    --chunk-len 10 --stride 2 --overwrite
}

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

for step in "$@"; do
  case "$step" in
    init) run_init ;;
    list) run_list ;;
    validate) run_validate ;;
    build) run_build ;;
    all)
      run_init
      run_list
      run_validate
      run_build
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown step: $step" >&2
      usage >&2
      exit 1
      ;;
  esac
done
