# Create virtual environment
# export HF_ENDPOINT=https://hf-mirror.com
# uv venv --python 3.10 examples/mobile_aloha_AgileX/.venv
source examples/mobile_aloha_AgileX/.venv/bin/activate

# uv pip sync examples/mobile_aloha_AgileX/requirements.txt
# uv pip install -e packages/openpi-client

# Run the robot
python examples/mobile_aloha_AgileX/plot_traj.py