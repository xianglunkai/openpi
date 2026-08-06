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
    max_episode_time_s: int = 600
    action_horizon: int = 32
    
    # Action interpolation parameters
    inference_fps: int = 30
    use_action_interpolation: bool = False
    multiplier: int = 1
    
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
    runtime = _runtime.Runtime(
        environment=_env.AlohaRealEnvironment(reset_position=metadata.get("reset_pose"), control_freq_hz=control_freq_hz, use_single_arm=args.use_single_arm),
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
            )
        ),
        subscribers=[],
        max_hz=args.inference_fps,
        num_episodes=args.num_episodes,
        max_episode_time_s=args.max_episode_time_s,
        use_action_interpolation = args.use_action_interpolation,
        multiplier = args.multiplier,
    )

    runtime.run()


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)

# python -m examples.mobile_aloha_AgileX.main.py