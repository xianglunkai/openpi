#!/usr/bin/env bash
# cd openpi/rlt_online_rl
# conda activate rl-tokens

source /opt/ros/noetic/setup.bash

# Conda Python loads its own libffi (via ctypes). ROS cv_bridge then pulls in
# system libp11-kit, which expects system libffi symbols (LIBFFI_BASE_7.0).
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:${LD_PRELOAD}}"

# ROS first; drop conda/cmeel paths that conda activate may inject.
_rollout_ld_path="/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu"
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  IFS=':' read -ra _ld_parts <<< "${LD_LIBRARY_PATH}"
  for _entry in "${_ld_parts[@]}"; do
    [[ -z "${_entry}" ]] && continue
    [[ "${_entry}" == *miniconda* || "${_entry}" == *conda* || "${_entry}" == *cmeel* ]] && continue
    [[ "${_entry}" == /opt/ros/noetic/lib* ]] && continue
    _rollout_ld_path="${_rollout_ld_path}:${_entry}"
  done
fi
export LD_LIBRARY_PATH="${_rollout_ld_path}"

# Omit --num_episodes to run continuously (reset pose + wait for keyboard 'o' after each s/f).
python launch/launch_robot_rollout_ros1_agilex_single_arm.py \
  --config runs/screw_sorting/checkpoints/online_rl_config.yaml\
  --machine_a_ws_url ws://0.0.0.0:8000 \
  --policy_resume_delay_s 0.0
