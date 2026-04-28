import dataclasses
import logging
import pathlib
import time

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime import runtime_rtc as _runtime_rtc
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.runtime.runtime_rtc import RTCConfig, RTCAttentionSchedule, RuntimeRTC
import tyro

from examples.mobile_aloha_AgileX import sim_env as _env
from queue import Queue

try:
    from examples.mobile_aloha_AgileX.monitor import RealTimeMonitor
except Exception:
    RealTimeMonitor = None

@dataclasses.dataclass
class Args:
    # Network parameters
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Episode parameters
    num_episodes: int = 1
    max_episode_time_s: int = 180
    
    # RTC parameters
    use_rtc: bool = True  
    execution_horizon: int = 25
    action_queue_size_to_get_new_actions: int = 20
   
    # Action interpolation parameters
    inference_fps: int = 50
    use_action_interpolation: bool = True
    multiplier: int = 2
    
    # Simulation parameters
    out_dir: pathlib.Path = pathlib.Path("data/aloha_sim/videos")
    task: str = "gym_aloha/AlohaTransferCube-v0"
    seed: int = 0


    # Small passthrough to enable sim_env ZeroMQ publisher without editing code
    env_zmq: bool = True
    env_zmq_endpoint: str = "tcp://127.0.0.1:5556"
    # If True, do not use physical rendered images from the simulator.
    # When enabled the environment will synthesize deterministic images which
    # avoids GL context / threading issues and is useful for headless tests.
    env_no_physical_images: bool = False
    
    
    # Use single arm
    use_single_arm: bool = False
    
   
def main(args: Args) -> None:
    
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    
    rtc_config = RTCConfig(
        enabled=args.use_rtc,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        execution_horizon=args.execution_horizon,
        debug=False,
    )
    
    # Instantiate environment separately so we can optionally attach a monitor
    env = _env.AlohaSimEnvironment(
        task=args.task,
        seed=args.seed,
        zmq_monitor=args.env_zmq,
        zmq_endpoint=args.env_zmq_endpoint,
        use_physical_images=not args.env_no_physical_images,
        use_single_arm=args.use_single_arm,
    )

    runtime = _runtime_rtc.RuntimeRTC(
        environment=env,
        policy=ws_client_policy,
        subscribers=[],
        fps=args.inference_fps,
        num_episodes=args.num_episodes,
        max_episode_time_s=args.max_episode_time_s,
        use_action_interpolation = args.use_action_interpolation,
        multiplier = args.multiplier,
        rtc_config=rtc_config,
        action_queue_size_to_get_new_actions=args.action_queue_size_to_get_new_actions,
    )

    runtime.run()
 
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)

# python -m examples.mobile_aloha_AgileX.main.py