# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com
uv run scripts/serve_policy.py \
    --env COBOT \
    --default_prompt='Put the zip tie through the lock.' \
    policy:checkpoint --policy.config=pi05_zip_tie --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_zip_tie/pi05_zip_tie/25000