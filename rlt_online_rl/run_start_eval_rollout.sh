#!/usr/bin/env bash
# Eval rollout: actor inference + robot control only (no replay writes).
# conda activate rl-tokens

source /opt/ros/noetic/setup.bash

# Conda Python loads its own libffi (via ctypes). ROS cv_bridge then pulls in
# system libp11-kit, which expects system libffi symbols (LIBFFI_BASE_7.0).
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libffi.so.7${LD_PRELOAD:+:${LD_PRELOAD}}"

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

CONFIG="${RLT_EVAL_CONFIG:-runs/screw_sorting/checkpoints/online_rl_config.yaml}"
MACHINE_A_WS_URL="${RLT_MACHINE_A_WS_URL:-ws://0.0.0.0:8000}"

python launch/launch_actor_eval_ros1_agilex_single_arm.py \
  --config "${CONFIG}" \
  --machine_a_ws_url "${MACHINE_A_WS_URL}" \
  --policy_resume_delay_s 0.0
