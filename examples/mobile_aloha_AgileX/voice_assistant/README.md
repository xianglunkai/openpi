# 语音助手

这是一个为ALOHA机器人设计的语音控制助手，支持通过语音指令控制机器人执行特定任务。

## 功能特性

- **语音活动检测 (Silero VAD)**: 使用Silero VAD自动检测麦克风中的语音输入
- **语音识别**: 使用OpenAI Whisper将语音转换为文字
- **智能理解**: 通过ChatGPT理解用户意图并生成回复
- **任务执行**: 将识别出的任务发送到Redis队列
- **语音反馈**: 使用Coqui TTS将回复转换为语音播放

## 支持的任务

1. **拧瓶盖** - 机器人执行拧瓶盖动作
2. **撕标签** - 机器人执行撕标签动作  
3. **停止** - 停止当前执行的任务

## 系统要求

在开始之前，请确保您的系统满足以下要求：

### 硬件要求
- NVIDIA GPU (建议8GB+显存)
- CUDA 12.0+ 支持
- 足够的系统内存 (建议16GB+)

### 软件要求
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Docker运行时 (nvidia-docker2)
- Ubuntu 20.04+ 或其他支持的Linux发行版

### 系统检查

运行系统要求检查脚本：

```bash
cd voice_assistant
chmod +x check_system.sh
./check_system.sh
```

如果检查失败，请根据提示安装缺失的组件。

## 安装和运行

### 使用Docker Compose

1. 设置环境变量：
```bash
# 复制环境变量模板文件
cp voice_assistant/env.example voice_assistant/.env

# 编辑.env文件，填入您的OpenAI API密钥
nano voice_assistant/.env
```

2. 运行Docker Compose：
```bash
# 在aloha_real目录下运行
docker compose -f compose-local.yml up --build
```

## 使用方法

1. 确保已设置好 `.env` 文件并填入正确的 `OPENAI_API_KEY`
2. 运行完整的ALOHA系统：

```bash
docker compose -f compose-local.yml up --build
```

3. voice_assistant服务会自动启动并监听语音输入
4. 对着麦克风说出您的指令，例如：
   - "请帮我拧瓶盖"
   - "撕掉这个标签"
   - "停止"

5. 系统会：
   - 识别您的语音
   - 理解您的意图
   - 发送任务到Redis
   - 播放语音回复

## 工作流程

1. **语音检测**: 程序持续监听麦克风，使用Silero VAD检测语音活动
2. **语音录制**: 检测到语音后开始录音，静音后停止
3. **语音识别**: 使用Whisper将录音转换为文字
4. **意图理解**: ChatGPT分析文字，判断用户想要执行的任务
5. **任务执行**: 将任务编号发送到Redis队列
6. **语音反馈**: 使用TTS播放确认回复

## Redis消息格式

发送到Redis的消息格式：

```json
{
    "task": "1",
    "task_name": "拧瓶盖", 
    "timestamp": 1234567890.123
}
```

任务编号对应：
- "1": 拧瓶盖
- "2": 撕标签
- "3": 停止

## 故障排除

### 常见问题

1. **麦克风权限**: 确保Docker容器有麦克风访问权限
2. **音频设备**: 检查系统音频设备是否正常工作
3. **网络连接**: 确保能够访问OpenAI API
4. **Redis连接**: Redis服务在Docker Compose中自动启动

### 调试模式

如果遇到问题，可以查看Docker容器的日志：

```bash
docker compose -f compose-local.yml logs voice_assistant
```

### 单独测试构建

如果Docker Compose构建失败，可以单独测试构建：

```bash
cd voice_assistant
chmod +x test_build.sh
./test_build.sh
```

### 常见构建问题

1. **CUDA镜像问题**: 确保系统支持NVIDIA Docker运行时
2. **网络问题**: 模型下载需要稳定的网络连接
3. **内存不足**: 模型下载和构建需要足够的内存空间
4. **GPU不可用**: 系统启动时会检查CUDA，如果不可用会报错退出

## 注意事项

- **必须使用CUDA**: 此系统强制要求NVIDIA GPU和CUDA 12.2支持，不支持CPU模式
- **模型预下载**: 所有AI模型在Docker构建时预下载，确保运行时性能
- 需要稳定的网络连接来访问OpenAI API
- 建议在安静的环境中使用以获得更好的识别效果
- Redis队列名为 `aloha_voice_commands`
- 所有服务都在Docker容器中运行，确保Docker环境配置正确
- 系统启动时会验证CUDA可用性，如果不可用将拒绝启动
- 需要NVIDIA Docker运行时支持
