# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com



uv run scripts/serve_policy.py \
    --env ALOHA \
    --default_prompt='Use the correct hand to pick up the bottle on the table, place it in a suitable location or pass it to the right hand, and finally put it on top of the black book with the bottle neck facing up' \
    policy:checkpoint --policy.config=pi05_cobot_adjust_bottle --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_adjust_bottle/checkpoint-30k/ \