# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com

# fold shirt
uv run scripts/serve_policy.py \
    --env ALOHA \
    --default_prompt='Carefully using its two arms to fold the shirt softly.' \
    policy:checkpoint --policy.config=pi05_cobot_fold_shirt --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_lerobot_combined_data/torch_converted_30k \

# # hover over bottle
# uv run scripts/serve_policy.py \
#     --env ALOHA \
#     --default_prompt='the one hand picks up the bottle and passes it to the another hand to place on the black book.' \
#     policy:checkpoint --policy.config=pi05_cobot_fold_shirt --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_lerobot_combined_data/torch_converted_30k \