#!/usr/bin/env bash
# Build one dataset from successful HIL episodes:
#   - a fully automatic success is copied whole
#   - a success that contains intervention keeps only the intervention frames
# Failure and timeout are skipped. Outcomes come from meta/hil_outcomes.jsonl.
#
#   bash examples/mobile_aloha_AgileX/run_extract_hil_lerobot_subset.sh [src] [output]
#
# Output defaults to "<src>_auto_and_intervention".
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/huggingface/lerobot}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

SRC="${1:-${HF_LEROBOT_HOME}/openpi_hil/hil_screw_sorting_20261106_162646}"
OUTPUT="${2:-${SRC}_auto_and_intervention}"

if [[ ! -f "${SRC}/meta/info.json" ]]; then
  echo "Not a LeRobot dataset: ${SRC}" >&2
  exit 1
fi

if [[ ! -f "${SRC}/meta/hil_outcomes.jsonl" ]]; then
  echo "这份数据没有 meta/hil_outcomes.jsonl，当时没有记下成功或失败。" >&2
  echo "超时、按 f、按 s 都没法从帧里区分，所以不能生成只要成功情节的新数据集。" >&2
  echo "之后新录的数据会写这个文件，到时直接运行本脚本即可。" >&2
  exit 1
fi

args=(
  --src "${SRC}"
  --output "${OUTPUT}"
  --overwrite
)
if [[ -n "${MIN_FRAMES:-}" ]]; then
  args+=(--min-frames "${MIN_FRAMES}")
fi

python -m examples.mobile_aloha_AgileX.extract_hil_lerobot_subset "${args[@]}"
