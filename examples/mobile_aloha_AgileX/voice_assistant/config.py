"""
配置文件
包含API密钥和其他配置信息
"""

# OpenAI API配置
OPENAI_API_KEY = "your-openai-api-key-here"  # 请替换为您的实际API密钥

# Redis配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# 音频配置
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
CHANNELS = 1

# VAD配置 (Silero VAD)
VAD_THRESHOLD = 0.5  # 0.0-1.0，越高越敏感

# Whisper配置
WHISPER_MODEL = "base"  # tiny, base, small, medium, large

# TTS配置
TTS_LANGUAGE = "zh"  # 中文
