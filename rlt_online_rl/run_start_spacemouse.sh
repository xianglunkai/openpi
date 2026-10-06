#!/usr/bin/env bash
# conda activate rl-tokens
# Requires ROS Noetic for rospy.

source /opt/ros/noetic/setup.bash

# placo/pinocchio need boost_python310 from cmeel; ROS setup can pull libboost_python38 (py3.8).
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is empty. Activate rl-tokens before running this script."
  exit 1
fi
CMEEL_LIB="${CONDA_PREFIX}/lib/python3.10/site-packages/cmeel.prefix/lib"
export LD_LIBRARY_PATH="${CMEEL_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

# Toggle teleop/policy with keyboard 't' or Space (keyboard_toggle script).
python spacemouse_teleop_ros1.py "$@"

