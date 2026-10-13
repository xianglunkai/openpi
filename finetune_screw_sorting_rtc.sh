#!/bin/bash
# RTC-SFT for single-arm screw sorting.
# Dataset: screw_sorting_single_sft_ep279_annotated
# action_horizon H=30, rtc_max_delay=10
# Norm stats reused from pi05_cobot_screw_sorting_single (same action space).

DO_STEP1=0  # 数据已在 HF_LEROBOT_HOME
DO_STEP2=0
DO_STEP3=1  # 复用已有 norm_stats
DO_STEP4=1

export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export JAX_COORDINATOR_ADDRESS="localhost:1234"
export JAX_PROCESS_INDEX=0
export JAX_NUM_PROCESSES=1
export NCCL_NVLS_ENABLE=0

export TORCH_NCCL_ENABLE_MONITORING=0

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

export repo_id="${repo_id:-screw_sorting_single_sft_ep279_annotated}"
export config_name="${config_name:-pi05_cobot_screw_sorting_single_rtc}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/huggingface/lerobot}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"

if [ "$DO_STEP1" -eq 1 ]; then
    echo "执行步骤1: 转换数据到 LeRobot 数据集..."
    uv run --no-sync examples/mobile_aloha_AgileX/convert_aloha_data_to_lerobot.py --raw_dir "$raw_dir" --repo_id "$repo_id"
    echo "步骤1完成"
else
    echo "跳过步骤1: 转换数据到 LeRobot 数据集"
fi

if [ "$DO_STEP2" -eq 1 ]; then
    echo "步骤2: 配置已写入 src/openpi/training/config.py ($config_name)"
    echo "H=30, rtc_max_delay=10，可用 --model.action-horizon / --model.rtc-max-delay 覆盖"
else
    echo "跳过步骤2: 定义训练配置"
fi

if [ "$DO_STEP3" -eq 1 ]; then
    echo "执行步骤3: 计算归一化统计量..."
    uv run --no-sync scripts/compute_norm_stats.py --config-name "$config_name"
    echo "步骤3完成"
else
    echo "跳过步骤3: 复用 assets/pi05_cobot_screw_sorting_single/${repo_id}"
fi

if [ "$DO_STEP4" -eq 1 ]; then
    echo "执行步骤4: 开始 RTC-SFT 微调 ($config_name, H=30)..."
    uv run --no-sync scripts/train.py "$config_name" --exp-name="$config_name"
    echo "步骤4完成"
else
    echo "跳过步骤4: 微调训练"
fi

echo "所有选定的步骤已完成!"
echo "推理测试: uv run --no-sync scripts/test_rtc_fold_shirt.py --mode math"
echo "加载 ckpt:  uv run --no-sync scripts/test_rtc_fold_shirt.py --mode checkpoint --checkpoint-dir checkpoints/${config_name}/${config_name}/30000 --delay 6"
echo "服务:      uv run --no-sync scripts/serve_policy.py policy:checkpoint --policy.config=${config_name} --policy.dir=checkpoints/${config_name}/${config_name}/30000"
