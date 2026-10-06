# Create virtual environment
export HF_ENDPOINT=https://hf-mirror.com
# uv venv --python 3.10 examples/aloha_sim/.venv
source examples/aloha_sim/.venv/bin/activate
# uv pip sync examples/aloha_sim/requirements.txt
# uv pip install -e packages/openpi-client
# uv pip install modelscope

modelscope download --model Gnepua/pi0_aloha_sim --local_dir checkpoints/aloha_sim/pi0_aloha_sim

# Run the simulation
MUJOCO_GL=egl python examples/aloha_sim/main.py