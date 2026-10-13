# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com

uv run scripts/serve_policy.py \
    --env COBOT \
    --default_prompt='Please take a pack of tissues from the drawer next to you, then pull out one sheet and give it to me.' \
    policy:checkpoint --policy.config=pi05_take_me_tissues --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_mobile_cobot_take_me_tissues/29999 \