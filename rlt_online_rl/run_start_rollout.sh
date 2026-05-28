# cd openpi-RLT/rlt_online_rl
# source /opt/ros/humble/setup.bash
# conda activate rlt_online_rl310
python launch/launch_robot_rollout_ros1_agilex_single_arm.py \
  --config configs/tasks/screw_sorting/online_rl.yaml \
  --machine_a_ws_url ws://0.0.0.0:8000