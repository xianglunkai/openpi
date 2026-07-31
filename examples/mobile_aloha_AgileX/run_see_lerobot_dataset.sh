#!/usr/bin/env bash
# Visualize a local LeRobot v2.1 dataset collected by runtime_rtc_hil.
#
# NOTE: visualize_dataset_html.py only supports video datasets. HIL recordings use
# mode=image (use_videos=False), so use visualize_dataset.py + Rerun instead.
set -euo pipefail

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/huggingface/lerobot}"

REPO_ID="${1:-openpi/screw_sorting_single_rl}"
EPISODE_INDEX="${2:-59}"

python .venv/lib/python3.11/site-packages/lerobot/scripts/visualize_dataset.py \
  --repo-id "${REPO_ID}" \
  --episode-index "${EPISODE_INDEX}"
