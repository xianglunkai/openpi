#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${SCRIPT_DIR}/.wandb.local.env"

echo "=== wandb 个人 Key 配置 ==="
echo "用途：在共享机器上各自用自己的 wandb 账号，不影响同事的全局 login。"
echo ""
echo "1. 浏览器打开 https://wandb.ai/authorize （登录你自己的账号）"
echo "2. 复制 API Key，粘贴到下面"
echo ""

if [ -f "$ENV_FILE" ]; then
    read -r -p ".wandb.local.env 已存在，是否覆盖？[y/N] " ans
    if [[ ! "$ans" =~ ^[Yy]$ ]]; then
        echo "已取消。"
        exit 0
    fi
fi

read -r -s -p "Paste WANDB_API_KEY: " api_key
echo ""

if [ -z "$api_key" ]; then
    echo "ERROR: API Key 不能为空"
    exit 1
fi

read -r -p "WANDB_ENTITY (可选，直接回车跳过): " entity

{
    echo "# Personal wandb credentials — do not commit"
    echo "WANDB_API_KEY=${api_key}"
    if [ -n "$entity" ]; then
        echo "WANDB_ENTITY=${entity}"
    fi
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo ""
echo "已写入 ${ENV_FILE}"
echo "验证连接..."

export http_proxy="${http_proxy:-http://10.0.0.112:3128}"
export https_proxy="${https_proxy:-$http_proxy}"
export HTTP_PROXY="${HTTP_PROXY:-$http_proxy}"
export HTTPS_PROXY="${HTTPS_PROXY:-$https_proxy}"
set -a
# shellcheck source=/dev/null
source "$ENV_FILE"
set +a

cd "$SCRIPT_DIR"
uv run --no-sync python - <<'PY'
import wandb
api = wandb.Api()
viewer = api.viewer
print(f"wandb 账号: {getattr(viewer, 'username', None) or viewer.email}")
print(f"默认 entity: {api.default_entity}")
PY

echo ""
echo "完成。以后直接运行 finetune 脚本即可，无需 wandb login。"
