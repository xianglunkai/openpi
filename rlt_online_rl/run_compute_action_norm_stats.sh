#!/usr/bin/env bash
# Build actor norm stats from a replay journal and point the task yaml at them.
#
# Joints are chunk deltas (action/ref - proprio, next_ref - next_proprio).
# Their q01/q99 come from that pool. Gripper stays absolute and uses min/max.
# The previous VLA stats file is not overwritten.
#
# Usage (from rlt_online_rl/):
#   bash run_compute_action_norm_stats.sh
#   RLT_SOURCE=human bash run_compute_action_norm_stats.sh
#   RLT_REPLAY_PATH=runs/screw_sorting/replay/replay_journal.pkl bash run_compute_action_norm_stats.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

TASK="${RLT_TASK:-screw_sorting}"
REPLAY_PATH="${RLT_REPLAY_PATH:-runs/${TASK}/replay/replay_journal.pkl}"
OUTPUT="${RLT_OUTPUT:-configs/tasks/${TASK}/stats/norm_stats_delta_journal.json}"
PREVIOUS="${RLT_PREVIOUS:-configs/tasks/${TASK}/stats/norm_stats_delta.json}"
TASK_CONFIG="${RLT_TASK_CONFIG:-configs/tasks/${TASK}/online_rl.yaml}"
PHASE="${RLT_PHASE:-all}"
SOURCE="${RLT_SOURCE:-all}"
POSE_DIM="${RLT_POSE_DIM:-6}"
UPDATE_CONFIG="${RLT_UPDATE_CONFIG:-1}"

args=(
  --replay-path "${REPLAY_PATH}"
  --output "${OUTPUT}"
  --pose-dim "${POSE_DIM}"
  --phase "${PHASE}"
  --source "${SOURCE}"
)
if [[ -f "${PREVIOUS}" ]]; then
  args+=(--compare "${PREVIOUS}")
fi

echo "==> Actor norm stats"
echo "    replay=${REPLAY_PATH}"
echo "    output=${OUTPUT}"
echo "    phase=${PHASE} source=${SOURCE}"
python scripts/offline/compute_action_norm_stats.py "${args[@]}"

if [[ "${UPDATE_CONFIG}" == "1" ]]; then
  python - "${TASK_CONFIG}" "${OUTPUT}" <<'PY'
import pathlib, sys
config_path, output = sys.argv[1:]
path = pathlib.Path(config_path)
text = path.read_text(encoding="utf-8")
needle = "action_norm_stats_path:"
lines = []
replaced = False
for line in text.splitlines(keepends=True):
    if line.lstrip().startswith(needle):
        indent = line[: len(line) - len(line.lstrip())]
        line = f"{indent}{needle} {output}\n"
        replaced = True
    lines.append(line)
if not replaced:
    raise SystemExit(f"{needle} not found in {path}")
path.write_text("".join(lines), encoding="utf-8")
print(f"updated {path} -> {output}")
PY
else
  echo "config unchanged (${TASK_CONFIG})"
fi
