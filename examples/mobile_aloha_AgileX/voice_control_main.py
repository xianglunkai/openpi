import asyncio
import json
import gzip
import websockets
import subprocess
import signal
import os
import sys

# 存储连接的客户端
connected_clients = set()
# 存储当前运行的策略进程
current_process = None
# 服务器实例
server = None

async def handle_client(websocket, path=""):
    """处理每个websocket客户端连接"""
    print(f"{websocket.remote_address} connected")
    connected_clients.add(websocket)

    try:
        # 接收客户端发送的消息
        async for message in websocket:
            # 处理消息解析
            asr_text = await parse_message(message)
            if not asr_text:
                continue
                
            print(f"Start processing Robot Command action for: {asr_text}")
            
            # 处理ASR文本到机器人命令
            result = await process_asr_text_to_robot_command(asr_text)
            
            # 可选：将结果发送回客户端
            if result and result.get("success"):
                await websocket.send(json.dumps({
                    "status": "success", 
                    "message": f"Command executed: {result.get('command')}"
                }))
            else:
                await websocket.send(json.dumps({
                    "status": "error", 
                    "message": result.get("error", "Unknown error")
                }))
                
    except websockets.ConnectionClosed:
        print(f"{websocket.remote_address} disconnected")
    finally:
        # 从连接的客户端中移除当前客户端
        connected_clients.discard(websocket)

async def parse_message(message):
    """解析接收到的消息"""
    try:
        if isinstance(message, bytes):
            # 解压gzip
            decompressed = gzip.decompress(message).decode("utf-8")
            data = json.loads(decompressed)
        else:
            data = json.loads(message)
        
        asr_text = data.get("asr_text", "").strip()
        print(f"Received ASR text: {asr_text}")
        return asr_text
        
    except Exception as e:
        print(f"Failed to parse message: {e}")
        return None

async def process_asr_text_to_robot_command(asr_text):
    """处理asr_text到robot_command的转换"""
    global current_process
    
    if not asr_text.strip():
        return {"success": False, "error": "Empty ASR text"}
    
    print(f"Processing ASR text: {asr_text}")
    
    # 根据ASR文本选择对应的策略
    config_map = {
        "shirt": {
            "config": "pi05_cobot_fold_shirt",
            "checkpoint": "/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_shirt/checkpoint-15k"
        },
        "bottle": {
            "config": "pi05_cobot_adjust_bottle", 
            "checkpoint": "/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_adjust_bottle/checkpoint-30k"
        },
        "water": {
            "config": "pi05_cobot_pour_water",
            "checkpoint": "/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_pour_water/checkpoint-30k"
        }
    }
    
    # 查找匹配的关键词
    selected_config = None
    for keyword, config_info in config_map.items():
        if keyword in asr_text.lower():
            selected_config = config_info
            break
    
    if not selected_config:
        print("No matching command found!")
        return {"success": False, "error": "No matching command configuration"}
    
    robot_command = asr_text
    print(f"Using config: {selected_config['config']}")
    print(f"Generated robot command: {robot_command}")
    
    # 构建命令
    command = [
        "uv", "run", "scripts/serve_policy.py",
        "--env", "ALOHA",
        "--default_prompt="f'{robot_command}',
        "policy:checkpoint",
        f"--policy.config={selected_config['config']}",
        f"--policy.dir={selected_config['checkpoint']}"
    ]
    
    print("Executing command:", " ".join(command))
    
    try:
        # 如果已有进程在运行，先终止它
        if current_process and current_process.returncode is None:
            print("Terminating previous policy process...")
            current_process.terminate()
            try:
                await asyncio.wait_for(current_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                current_process.kill()
                await current_process.wait()
        
        # 异步执行子进程
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # 创建新的进程组，便于终止整个进程树
        )
        
        current_process = process
        
        # 等待一段时间看进程是否正常启动
        try:
            # 等待进程启动（最多10秒）
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            # 进程仍在运行，说明启动成功
            print("Policy process started successfully and is running")
            return {"success": True, "command": robot_command, "pid": process.pid}
        else:
            # 进程已退出，检查退出状态
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                print("Policy executed successfully and exited")
                if stdout:
                    print(f"Output: {stdout.decode()}")
                return {"success": True, "command": robot_command}
            else:
                error_msg = stderr.decode() if stderr else f"Process exited with code {process.returncode}"
                print(f"Policy process failed: {error_msg}")
                return {"success": False, "error": error_msg}
                
    except Exception as e:
        print(f"Failed to execute command: {e}")
        return {"success": False, "error": str(e)}

async def shutdown():
    """优雅关闭"""
    global current_process, server
    
    print("\nShutting down server...")
    
    # 终止所有子进程
    if current_process and current_process.returncode is None:
        print("Terminating policy process...")
        # 使用进程组终止，确保所有子进程都被终止
        try:
            os.killpg(os.getpgid(current_process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass  # 进程可能已经结束
        
        try:
            await asyncio.wait_for(current_process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            print("Force killing policy process...")
            try:
                os.killpg(os.getpgid(current_process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
    
    # 关闭所有WebSocket连接
    if connected_clients:
        print(f"Closing {len(connected_clients)} WebSocket connections...")
        close_tasks = [client.close() for client in connected_clients.copy()]
        await asyncio.gather(*close_tasks, return_exceptions=True)
        connected_clients.clear()
    
    # 停止服务器
    if server:
        server.close()
        await server.wait_closed()
        print("WebSocket server stopped")
    
    print("Server shutdown complete")

def signal_handler(signum, frame):
    """处理中断信号"""
    print(f"\nReceived signal {signum}, initiating shutdown...")
    # 创建任务来执行关闭流程
    asyncio.create_task(shutdown())

# 启动服务器
async def main():
    global server
    
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动WebSocket服务器
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        8958
    )
    print("🚀 WebSocket服务启动，监听 ws://0.0.0.0:8958")
    print("Press Ctrl+C to stop the server")
    
    try:
        # 保持服务器运行
        await server.wait_closed()
    except asyncio.CancelledError:
        print("Server task cancelled")
    finally:
        # 确保清理资源
        await shutdown()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server interrupted by user")
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        print("Server exited")