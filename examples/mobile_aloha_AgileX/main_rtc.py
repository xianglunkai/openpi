import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime import runtime_rtc as _runtime_rtc
from openpi_client.runtime.agents import policy_agent as _policy_agent
from openpi_client.runtime.runtime_rtc import RTCConfig, RTCAttentionSchedule, RuntimeRTC
import tyro

from examples.mobile_aloha_AgileX import env as _env
from examples.mobile_aloha_AgileX import robot_utils

@dataclasses.dataclass
class Args:
    # Network parameters
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Episode parameters
    num_episodes: int = 1
    max_episode_time_s: int = 240
    
    # RTC parameters
    use_rtc: bool = True  
    execution_horizon: int = 25
    action_queue_size_to_get_new_actions: int = 25
   
    # Action interpolation parameters
    inference_fps: int = 30
    use_action_interpolation: bool = True
    multiplier: int = 2
    
    # Use single arm
    use_single_arm: bool = True
    
   
 
def main(args: Args) -> None:
    
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    control_freq_hz = args.inference_fps * args.multiplier if args.use_action_interpolation else args.inference_fps

    metadata = ws_client_policy.get_server_metadata()
    
    rtc_config = RTCConfig(
        enabled=args.use_rtc,
        prefix_attention_schedule=RTCAttentionSchedule.EXP,
        execution_horizon=args.execution_horizon,
        debug=True,
    )
    
    runtime = _runtime_rtc.RuntimeRTC(
        environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose"), control_freq_hz=control_freq_hz, use_single_arm=args.use_single_arm),
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

# python -m examples.mobile_aloha_AgileX.main_rtc.py