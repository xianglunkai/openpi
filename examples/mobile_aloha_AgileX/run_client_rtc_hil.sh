#!/usr/bin/env bash
# Create virtual environment
# export HF_ENDPOINT=https://hf-mirror.com
# uv venv --python 3.10 examples/mobile_aloha_AgileX/.venv
source examples/mobile_aloha_AgileX/.venv/bin/activate

# uv pip sync examples/mobile_aloha_AgileX/requirements.txt
# uv pip install -e packages/openpi-client

# uv pip install pynput pyspacemouse placo

# placo/pinocchio need boost_python310 from cmeel; ROS/system libs can pull libboost_python38 (py3.8).
CMEEL_LIB="${VIRTUAL_ENV}/lib/python3.10/site-packages/cmeel.prefix/lib"
export LD_LIBRARY_PATH="${CMEEL_LIB}:${LD_LIBRARY_PATH}"

# Run the robot
export HF_LEROBOT_HOME=/data/huggingface/lerobot
export HF_HOME=/data/huggingface

# Start recorder first in another terminal:
#   bash examples/mobile_aloha_AgileX/run_lerobot_recorder.sh

python -m examples.mobile_aloha_AgileX.main_rtc_hil 

