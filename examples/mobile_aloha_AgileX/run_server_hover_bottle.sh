# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com

uv run scripts/serve_policy.py \
    --env ALOHA \
    --default_prompt='the one hand picks up the bottle and passes it to the another hand to place on the black book' \
    policy:checkpoint --policy.config=pi05_cobot_handover_bottle --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_hover_bottle_checkpoint/20000/ \