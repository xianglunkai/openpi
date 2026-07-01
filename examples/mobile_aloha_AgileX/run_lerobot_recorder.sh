#!/usr/bin/env bash
# Run in openpi venv (lerobot already installed). Receives HIL frames from main_rtc_hil.
set -euo pipefail

cd "$(dirname "$0")/../.."
source .venv/bin/activate

export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/data/huggingface/lerobot}"
export HF_HOME="${HF_HOME:-/data/huggingface}"

python -m examples.mobile_aloha_AgileX.lerobot_recorder_service "$@"
