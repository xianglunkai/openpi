#!/bin/bash

# 配置变量 - 设置为 1 表示执行，0 表示跳过
DO_STEP1=0  # 转换数据到 LeRobot 数据集
DO_STEP2=0 # 定义训练配置（此步骤主要是编辑文件，这里保留为提醒）
DO_STEP3=1  # 计算归一化统计量
DO_STEP4=0  # 开始微调训练

export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

export TORCH_NCCL_ENABLE_MONITORING=0  # disable watchdog

export CUDA_VISIBLE_DEVICES=0,1,2,3

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 设置 Hugging Face 镜像端点
export HF_ENDPOINT=https://hf-mirror.com

export repo_id=pour_water
export raw_dir=datasets/pour_water
export config_name=pi05_cobot_pour_water
export HF_LEROBOT_HOME=/workspace/huggingface/lerobot
export HF_HOME=/workspace/huggingface

# 步骤1: 将数据转换为 LeRobot 数据集
if [ "$DO_STEP1" -eq 1 ]; then
    echo "执行步骤1: 转换数据到 LeRobot 数据集..."
    uv run examples/mobile_aloha_AgileX/convert_aloha_data_to_lerobot.py --raw_dir "$raw_dir" --repo_id "$repo_id"
    echo "步骤1完成"
else
    echo "跳过步骤1: 转换数据到 LeRobot 数据集"
fi

#步骤2: 定义训练配置（此步骤需要手动编辑配置文件）
if [ "$DO_STEP2" -eq 1 ]; then
    echo "步骤2: 请手动编辑训练配置文件"
    echo "你需要修改以下配置文件以适应你的数据集:"
    echo "1. 修改 LiberoInputs 和 LiberoOutputs 配置"
    echo "2. 调整 LeRobotLiberoDataConfig"
    echo "3. 设置 TrainConfig 中的超参数"
    echo "编辑完成后按回车继续..."
else
    echo "跳过步骤2: 定义训练配置"
fi

# 步骤3: 计算训练数据的归一化统计量
if [ "$DO_STEP3" -eq 1 ]; then
    echo "执行步骤3: 计算归一化统计量..."
    uv run scripts/compute_norm_stats.py --config-name "$config_name"
    echo "步骤3完成"
else
    echo "跳过步骤3: 计算归一化统计量"
fi

# 步骤4: 开始微调训练
if [ "$DO_STEP4" -eq 1 ]; then
    echo "执行步骤4: 开始微调训练..."
    uv run scripts/train.py "$config_name" \
        --exp-name="$config_name" \
        # --resume \
        
    echo "步骤4完成"
else
    echo "跳过步骤4: 微调训练"
fi

echo "所有选定的步骤已完成!"