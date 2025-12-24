# Voice Assistant Dockerfile with CUDA support for RTX 5090
FROM nvidia/cuda:12.8-cudnn9-devel-ubuntu22.04

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    portaudio19-dev \
    alsa-utils \
    libasound2-dev \
    build-essential \
    git \
    wget \
    curl \
    pulseaudio \
    pulseaudio-utils \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 复制项目文件
COPY pyproject.toml .

# 安装Python依赖
RUN uv pip install --system -e .

# 预下载Whisper模型 (CPU，运行时再切换到CUDA)
RUN uv run python3 -c "import whisper; whisper.load_model('large-v3', device='cpu')"

# 预下载TTS模型 (CPU，运行时再切换到CUDA)
ENV COQUI_TOS_AGREED=1
RUN uv run python3 -c "from TTS.api import TTS; tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False)" || echo "TTS model download failed, will download at runtime"

# 预下载Silero VAD模型
RUN uv run python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=False)"

# 默认命令
CMD ["python3", "voice_assistant.py"]
