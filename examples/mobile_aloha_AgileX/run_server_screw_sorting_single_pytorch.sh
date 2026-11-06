#!/usr/bin/env bash
# Same websocket the robot already uses. Only the weights change: safetensors
# produced by examples/convert_jax_model_to_pytorch.py, plus norm stats copied
# to assets/screw_sorting_single/norm_stats.json.
# The client (examples/mobile_aloha_AgileX/main.py) stays on port 8000.
set -euo pipefail

export HF_ENDPOINT=https://hf-mirror.com

DIR="${RLT_PYTORCH_CKPT:-/home/xlk/work/openpi/checkpoints/pi05_cobot_screw_sorting_single_pytorch}"
if [[ ! -f "${DIR}/model.safetensors" ]]; then
  echo "missing ${DIR}/model.safetensors" >&2
  echo "run examples/convert_jax_model_to_pytorch.py on the x86 machine first" >&2
  exit 1
fi
if [[ ! -f "${DIR}/assets/screw_sorting_single/norm_stats.json" ]]; then
  echo "missing ${DIR}/assets/screw_sorting_single/norm_stats.json" >&2
  exit 1
fi

cd "$(dirname "$0")/../.."

uv run --no-sync scripts/serve_policy.py \
    --env COBOT \
    --default_prompt='Please sort and return the silver screws in the grey box to their proper places.' \
    policy:checkpoint \
    --policy.config=pi05_cobot_screw_sorting_single \
    --policy.dir="${DIR}"
