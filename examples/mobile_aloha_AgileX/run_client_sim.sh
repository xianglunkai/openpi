# Create virtual environment
# export HF_ENDPOINT=https://hf-mirror.com
# uv venv --python 3.10 examples/mobile_aloha_AgileX/.venv
source examples/mobile_aloha_AgileX/.venv/bin/activate

# Force an offscreen GL backend for MuJoCo to avoid GLFW context conflicts
# Options: 'egl' or 'osmesa' (choose one supported on your system)
export MUJOCO_GL=egl

# uv pip sync examples/mobile_aloha_AgileX/requirements.txt
# uv pip install -e packages/openpi-client

# Run the robot
# Tyro nested dataclass CLI expects args under the `--args.` prefix when invoked
# via the example launcher. Pass the zmq flag as `--args.env-zmq`.
python -m examples.mobile_aloha_AgileX.main_sim --args.env-zmq --args.env-no-physical-images

