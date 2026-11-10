#!/bin/bash
# Continue RTC-SFT from step 29999 for 5k steps.
# Data: hil_screw_sorting_20261106_162646_auto_and_intervention
#   96 episodes, 10428 frames, 30 Hz, 7D, cam_high + cam_right_wrist.
#   90 intervention snippets + 6 auto-success episodes.
# Model stays pi05 + rtc_max_delay=10 + action_horizon=32 so the checkpoint loads.
# Norm stats are read from the 29999 checkpoint. This set is too small to recompute.

DO_STEP4=1

export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-localhost:1236}"
export JAX_PROCESS_INDEX=0
export JAX_NUM_PROCESSES=1
export NCCL_NVLS_ENABLE=0

export TORCH_NCCL_ENABLE_MONITORING=0

# 0-3 已被占满。4-7 空闲。batch_size=64 需要能被卡数整除。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# 容器到 api.wandb.ai 会超时，在线 init 收到 403 后训练还没开始就退出。
# 离线日志在 wandb/ 目录，网络恢复后可用 wandb sync 上传。
export WANDB_MODE="${WANDB_MODE:-offline}"

export repo_id="${repo_id:-hil_screw_sorting_20261106_162646_auto_and_intervention}"
export config_name="${config_name:-pi05_cobot_screw_sorting_single_rtc_hil}"
export exp_name="${exp_name:-pi05_cobot_screw_sorting_single_rtc_hil}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/huggingface/lerobot}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"

if [[ ! -d "${HF_LEROBOT_HOME}/${repo_id}" ]]; then
    echo "缺少数据集: ${HF_LEROBOT_HOME}/${repo_id}" >&2
    exit 1
fi

echo "归一化: checkpoints/pi05_cobot_screw_sorting_single_rtc/pi05_cobot_screw_sorting_single_rtc/29999/assets/screw_sorting_single_sft_ep279_annotated"

if [ "$DO_STEP4" -eq 1 ]; then
    echo "执行步骤4: 从 29999 继续 RTC 微调 5000 step ($config_name)..."
    uv run --no-sync scripts/train.py "$config_name" --exp-name="$exp_name"
    echo "步骤4完成"
else
    echo "跳过步骤4: 微调训练"
fi

echo "所有选定的步骤已完成!"
echo "ckpt: checkpoints/${config_name}/${exp_name}/4999"
echo "服务: uv run --no-sync scripts/serve_policy.py policy:checkpoint --policy.config=${config_name} --policy.dir=checkpoints/${config_name}/${exp_name}/4999"
