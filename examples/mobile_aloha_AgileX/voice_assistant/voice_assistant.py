#!/usr/bin/env python3
"""
语音助手脚本
功能：
1. 使用VAD检测麦克风输入
2. 语音识别（Whisper）
3. ChatGPT处理指令并生成回复
4. Redis发送操作指令
5. TTS播放回复
"""

import os
import json
import time
import threading
import queue
import redis
import openai
import pyaudio
import numpy as np
import torch
import whisper
from io import BytesIO
import pygame
from TTS.api import TTS
import os
from config import OPENAI_API_KEY, REDIS_HOST, REDIS_PORT, REDIS_DB, VAD_THRESHOLD

class VoiceAssistant:
    def __init__(self):
        # 初始化配置 - 优先使用环境变量
        self.openai_api_key = os.getenv('OPENAI_API_KEY', OPENAI_API_KEY)
        self.redis_host = os.getenv('REDIS_HOST', REDIS_HOST)
        self.redis_port = int(os.getenv('REDIS_PORT', REDIS_PORT))
        self.redis_db = int(os.getenv('REDIS_DB', REDIS_DB))
        
        # 初始化组件
        self.setup_audio()
        self.setup_vad()
        self.setup_whisper()
        self.setup_redis()
        self.setup_openai()
        self.setup_tts()
        
        # 音频队列
        self.audio_queue = queue.Queue()
        self.is_recording = False
        
        # 任务映射
        self.task_mapping = {
            "1": "Remove the label from the bottle with the knife in the right hand.",
            "2": "Twist off the bottle cap.", 
            "3": "Stop",
            "4": "Return to sleep position"
        }
        
        # ChatGPT提示词模板
        self.chatgpt_prompt_template = """The user is now using voice control to operate the aloha robot. This robot supports the following tasks: {task_list}. Determine which operation the user wants to perform and return the corresponding number. The detected language is: {detected_language}. Generate a response statement in the appropriate language.

User's speech: {user_text}
Detected language: {detected_language}

Please reply in JSON format only:
{{
    "task_number": "X",
    "response_statement": "Z"
}}

Where:
- X is one of the task numbers: {task_numbers}
- Z is a response statement generated in the detected language ({detected_language})

Important: Return ONLY the JSON object, no additional text or explanation."""
        
        print("语音助手初始化完成！")
        print("支持的任务：")
        for key, task in self.task_mapping.items():
            print(f"  {key}: {task}")
        print("等待语音输入...")

    def setup_audio(self):
        """设置音频设备"""
        self.CHUNK = 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        self.audio = pyaudio.PyAudio()
        
        # 查找Pulse设备
        device_count = self.audio.get_device_count()
        pulse_device = None
        
        for i in range(device_count):
            device_info = self.audio.get_device_info_by_index(i)
            # print(f"设备信息: {device_info}")
            if device_info['maxInputChannels'] > 0 and 'pulse' in device_info['name'].lower():
                pulse_device = i
                print(f"找到Pulse设备: {device_info['name']}")
                break
        
        if pulse_device is None:
            raise RuntimeError("未找到Pulse音频设备")
        
        # 保存设备索引以便重新打开时使用
        self.pulse_device = pulse_device
        
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=pulse_device,
            frames_per_buffer=self.CHUNK
        )
        print(f"Pulse设备初始化成功，采样率: {self.RATE}Hz")

    def setup_vad(self):
        """设置语音活动检测"""
        # 使用Silero VAD模型
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.vad_threshold = VAD_THRESHOLD  # VAD阈值
        print("Silero VAD模型加载完成")

    def setup_whisper(self):
        """设置Whisper模型"""
        print("正在加载Whisper模型...")
        
        # 强制检查CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用！此系统必须使用CUDA。")
        
        print(f"使用CUDA设备: {torch.cuda.get_device_name(0)}")
        self.whisper_model = whisper.load_model("large-v3", device="cuda")
        print("Whisper large-v3模型加载完成")

    def setup_redis(self):
        """设置Redis连接"""
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            db=self.redis_db,
            decode_responses=True
        )
        self.redis_client.ping()
        print("Redis连接成功")

    def setup_openai(self):
        """设置OpenAI客户端"""
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        print("OpenAI客户端初始化完成")

    def setup_tts(self):
        """设置TTS"""
        pygame.mixer.init()
        
        # 初始化Coqui TTS
        print("正在加载Coqui TTS模型...")
        
        # 强制检查CUDA
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用！此系统必须使用CUDA。")
        
        print(f"TTS使用CUDA设备: {torch.cuda.get_device_name(0)}")
        
        # 使用多语言TTS模型
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda")
        print("Coqui TTS多语言模型加载完成")
        
        print("TTS初始化完成")

    def detect_voice_activity(self, audio_data):
        """检测语音活动"""
        # 将音频数据转换为tensor
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
        
        # 使用Silero VAD检测语音活动
        speech_prob = self.vad_model(audio_tensor, self.RATE).item()
        return speech_prob > self.vad_threshold

    def record_audio(self):
        """录制音频"""
        print("开始录音...")
        
        # 重新打开音频流以清空缓存
        print("重新打开音频流...")
        self.stream.stop_stream()
        self.stream.close()
        
        # 重新打开音频流
        self.stream = self.audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=self.pulse_device,
            frames_per_buffer=self.CHUNK
        )
        
        frames = []
        silent_frames = 0
        max_silent_frames = 30  # 最多30帧静音后停止录音
        
        while self.is_recording:
            data = self.stream.read(self.CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # VAD检测
            if self.detect_voice_activity(audio_data):
                frames.append(data)
                silent_frames = 0
            else:
                if frames:  # 如果已经开始录音
                    silent_frames += 1
                    frames.append(data)
                    
                    if silent_frames >= max_silent_frames:
                        break
        
        print("录音结束")
        return b''.join(frames)

    def speech_to_text(self, audio_data):
        """语音转文字"""
        print("正在识别语音...")
        
        # 将音频数据转换为numpy数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 使用Whisper识别，让Whisper自动检测语言
        result = self.whisper_model.transcribe(audio_np)
        text = result["text"].strip()
        detected_language = result["language"]
        
        print(f"识别结果: {text}")
        print(f"Whisper检测到的语言: {detected_language}")
        
        # 只支持英语(en)、中文(zh)和日语(ja)
        supported_languages = ["en", "zh", "ja"]
        if detected_language not in supported_languages:
            print(f"不支持的语言: {detected_language}，只支持: {supported_languages}")
            return None, detected_language
        
        return text, detected_language

    def get_chatgpt_response(self, user_text, detected_language):
        """获取ChatGPT响应"""
        try:
            # 动态生成任务列表
            task_list = []
            task_numbers = []
            for key, task in self.task_mapping.items():
                task_list.append(f"{key}) {task}")
                task_numbers.append(key)
            
            task_list_str = ", ".join(task_list)
            task_numbers_str = ", ".join(task_numbers)
            
            # 使用初始化中定义的提示词模板
            prompt = self.chatgpt_prompt_template.format(
                task_list=task_list_str,
                task_numbers=task_numbers_str,
                user_text=user_text,
                detected_language=detected_language
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an intelligent assistant helping users control the aloha robot. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=200,
            )
            
            response_text = response.choices[0].message.content.strip()
            print(f"ChatGPT响应: {response_text}")
            
            # 清理响应文本，移除markdown代码块标记
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # 移除 ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]   # 移除 ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]   # 移除结尾的 ```
            response_text = response_text.strip()
            
            # 解析JSON响应
            try:
                response_json = json.loads(response_text)
                task_num = response_json.get("task_number")
                reply_text = response_json.get("response_statement", "")
                
                return task_num, reply_text
            except json.JSONDecodeError as e:
                print(f"JSON解析失败: {e}")
                print(f"清理后的响应: {response_text}")
                # 如果JSON解析失败，尝试从响应中提取信息
                return None, None
            
        except Exception as e:
            print(f"ChatGPT请求失败: {e}")
            return None, None

    def send_to_redis(self, task_num):
        """发送任务到Redis使用pub/sub模式"""
        task_name = self.task_mapping.get(task_num, "未知任务")
        message = {
            "task": task_num,
            "task_name": task_name,
            "timestamp": time.time()
        }
        
        # 使用pub/sub模式发布消息
        self.redis_client.publish("aloha_voice_commands", json.dumps(message))
        print(f"任务已通过Redis pub/sub发送: {task_name}")

    def text_to_speech(self, text, detected_language=None):
        """文字转语音"""
        print(f"正在播放: {text}")
        
        # 生成临时音频文件
        temp_file = "/tmp/tts_output.wav"
        
        # 使用指定的TTS参数
        self.tts.tts_to_file(
            text=text,
            file_path=temp_file,
            speaker="Alexandra Hisakawa",
            language=detected_language,
            split_sentences=True
        )
        
        # 使用pygame播放
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        
        # 等待播放完成
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)
        
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

    def process_voice_command(self):
        """处理语音指令"""
        # 开始录音
        self.is_recording = True
        audio_data = self.record_audio()
        self.is_recording = False
        
        if len(audio_data) == 0:
            print("没有检测到语音")
            return
        
        # 语音转文字
        text, detected_language = self.speech_to_text(audio_data)
        if text is None:
            print("不支持的语言")
            self.text_to_speech("申し訳ございませんが、対応していない言語です。英語、中国語、または日本語でお話しください。", "ja")
            return
        elif not text:
            print("没有检测到语音")
            self.text_to_speech("申し訳ございませんが、よく聞き取れませんでした。もう一度お話しください。", "ja")
            return
        
        # ChatGPT处理
        task_num, reply_text = self.get_chatgpt_response(text, detected_language)
        
        # 打印检测到的语言
        print(f"检测到的语言: {detected_language}")
        
        # 发送任务到Redis
        if task_num:
            self.send_to_redis(task_num)
            self.text_to_speech(reply_text, detected_language)
        else:
            self.text_to_speech("申し訳ございませんが、お話の内容を理解できませんでした。もう一度お話しください。", "ja")

    def process_numeric_command(self, numeric_input):
        """处理数字输入指令"""
        # 检查输入是否为有效的任务编号
        if numeric_input in self.task_mapping:
            task_name = self.task_mapping[numeric_input]
            print(f"执行数字指令: {numeric_input} - {task_name}")
            
            # 发送任务到Redis
            self.send_to_redis(numeric_input)
            
        else:
            print(f"无效的任务编号: {numeric_input}")
            print("有效的任务编号: 1, 2, 3, 4")

    def run(self):
        """主运行循环"""
        try:
            while True:
                print("\n控制方式:")
                print("  - 按Enter键开始语音识别")
                print("  - 输入数字1-4直接执行任务")
                print("  - 输入'quit'退出")
                print("等待输入...")
                
                user_input = input().strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input == '':
                    self.process_voice_command()
                elif user_input.isdigit():
                    self.process_numeric_command(user_input)
                else:
                    print("无效输入，请按Enter进行语音识别，或输入1-4的数字")
                    
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'audio'):
            self.audio.terminate()
        pygame.mixer.quit()
        print("资源清理完成")

def main():
    """主函数"""
    assistant = VoiceAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
