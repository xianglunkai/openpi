# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com
uv run scripts/serve_policy.py \
    --env ALOHA_SIM \
    policy:checkpoint --policy.config=pi0_aloha --policy.dir=/home/xlk/work/openpi/checkpoints/aloha_sim/pi0_aloha_sim