#!/bin/bash
# JAX RECAP / π*0.6 完整训练 pipeline（screw_sorting_single）
#
#   1) compute_returns      写 meta/returns_{tag}.parquet
#   2) compute_advantages   写 meta/advantages_{tag}.parquet
#   3) compute_norm_stats   一般复用 SFT 的 assets，默认跳过
#   4) train                从 SFT ckpt 做 advantage-conditioned CFG 微调
#
# 用法:
#   bash finetune_screw_sorting_recap.sh
#   DO_STEP1=0 DO_STEP2=0 bash finetune_screw_sorting_recap.sh   # 标签已有，只训练
#
# 推理（CFG 从 TrainConfig.recap 自动加载）:
#   bash examples/mobile_aloha_AgileX/run_server_screw_sorting_single_recap.sh

set -euo pipefail

DO_STEP1="${DO_STEP1:-1}"  # compute returns
DO_STEP2="${DO_STEP2:-1}"  # compute advantages
DO_STEP3="${DO_STEP3:-0}"  # compute norm stats
DO_STEP4="${DO_STEP4:-1}"  # train

export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export JAX_COORDINATOR_ADDRESS="${JAX_COORDINATOR_ADDRESS:-localhost:1234}"
export JAX_PROCESS_INDEX="${JAX_PROCESS_INDEX:-0}"
export JAX_NUM_PROCESSES="${JAX_NUM_PROCESSES:-1}"
export NCCL_NVLS_ENABLE=0
export TORCH_NCCL_ENABLE_MONITORING=0
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/workspace/huggingface/lerobot}"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"

config_name="${config_name:-pi05_cobot_screw_sorting_single_recap}"
repo_id="${repo_id:-screw_sorting_single}"
dataset_root="${dataset_root:-${HF_LEROBOT_HOME}/${repo_id}}"
exp_name="${exp_name:-${config_name}}"

# DynamicReturn / 分位数标签
returns_tag="${returns_tag:-default}"
advantage_tag="${advantage_tag:-q30}"
dataset_type="${dataset_type:-sft}"          # sft: 全部当成功; rollout: 读 is_success
failure_reward="${failure_reward:--300}"
gamma="${gamma:-1.0}"
positive_fraction="${positive_fraction:-0.3}"
score_source="${score_source:-returns}"      # returns | values
values_path="${values_path:-}"               # score_source=values 时必填
n_step="${n_step:-10}"

returns_path="${dataset_root}/meta/returns_${returns_tag}.parquet"
advantage_path="${dataset_root}/meta/advantages_${advantage_tag}.parquet"

echo "=============================================="
echo "RECAP pipeline"
echo "  dataset_root      = ${dataset_root}"
echo "  dataset_type      = ${dataset_type}"
echo "  returns_path      = ${returns_path}"
echo "  advantage_path    = ${advantage_path}"
echo "  score_source      = ${score_source}"
echo "  positive_fraction = ${positive_fraction}"
echo "  config_name       = ${config_name}"
echo "=============================================="

if [ ! -d "${dataset_root}/data" ]; then
    echo "ERROR: LeRobot dataset not found at ${dataset_root}"
    echo "Set HF_LEROBOT_HOME / dataset_root, or convert data first."
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 1: per-frame return / reward sidecar
# -----------------------------------------------------------------------------
if [ "${DO_STEP1}" -eq 1 ]; then
    echo ">>> Step 1: compute_returns"
    uv run python scripts/recap/compute_returns.py \
        --dataset-root "${dataset_root}" \
        --dataset-type "${dataset_type}" \
        --gamma "${gamma}" \
        --failure-reward "${failure_reward}" \
        --tag "${returns_tag}"
    echo "<<< Step 1 done: ${returns_path}"
else
    echo ">>> Skip Step 1: compute_returns"
    if [ ! -f "${returns_path}" ]; then
        echo "ERROR: ${returns_path} missing. Run with DO_STEP1=1 first."
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 2: binarize positive / negative advantage labels
# -----------------------------------------------------------------------------
if [ "${DO_STEP2}" -eq 1 ]; then
    echo ">>> Step 2: compute_advantages (tag=${advantage_tag}, source=${score_source})"
    adv_args=(
        --dataset-root "${dataset_root}"
        --returns-tag "${returns_tag}"
        --tag "${advantage_tag}"
        --score-source "${score_source}"
        --positive-fraction "${positive_fraction}"
        --gamma "${gamma}"
        --n-step "${n_step}"
    )
    if [ "${score_source}" = "values" ]; then
        if [ -z "${values_path}" ]; then
            echo "ERROR: score_source=values requires values_path=/path/to/values.parquet"
            exit 1
        fi
        adv_args+=(--values-path "${values_path}")
    fi
    uv run python scripts/recap/compute_advantages.py "${adv_args[@]}"
    echo "<<< Step 2 done: ${advantage_path}"
    if [ -f "${dataset_root}/meta/advantages_${advantage_tag}_stats.json" ]; then
        echo "    stats:"
        cat "${dataset_root}/meta/advantages_${advantage_tag}_stats.json"
    fi
else
    echo ">>> Skip Step 2: compute_advantages"
    if [ ! -f "${advantage_path}" ]; then
        echo "ERROR: ${advantage_path} missing. Run with DO_STEP2=1 first."
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Step 3: norm stats (reuse SFT assets by default)
# -----------------------------------------------------------------------------
if [ "${DO_STEP3}" -eq 1 ]; then
    echo ">>> Step 3: compute_norm_stats"
    uv run scripts/compute_norm_stats.py --config-name "${config_name}"
    echo "<<< Step 3 done"
else
    echo ">>> Skip Step 3: compute_norm_stats (using existing SFT assets)"
fi

# -----------------------------------------------------------------------------
# Step 4: JAX CFG fine-tune
# -----------------------------------------------------------------------------
if [ "${DO_STEP4}" -eq 1 ]; then
    if [ ! -f "${advantage_path}" ]; then
        echo "ERROR: ${advantage_path} missing, cannot train."
        exit 1
    fi
    echo ">>> Step 4: train ${config_name}"
    echo "    recap.advantage-path=${advantage_path}"
    uv run scripts/train.py "${config_name}" \
        --exp-name="${exp_name}" \
        --recap.advantage-path="${advantage_path}"
    echo "<<< Step 4 done"
    echo
    echo "Serve (CFG loaded from TrainConfig.recap, no extra flags):"
    echo "  uv run scripts/serve_policy.py --env COBOT \\"
    echo "    policy:checkpoint --policy.config=${config_name} \\"
    echo "    --policy.dir=./checkpoints/${config_name}/${exp_name}/<step>"
else
    echo ">>> Skip Step 4: train"
fi

echo "Pipeline finished."
