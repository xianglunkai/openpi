# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com

uv run scripts/serve_policy.py \
    --env COBOT \
    --default_prompt='Carefully using its two arms to fold the shirt softly.' \
    policy:checkpoint --policy.config=pi05_cobot_fold_shirt --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_shirt/checkpoint-15k \