# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com
uv run scripts/serve_policy.py \
    --env ALOHA \
    --default_prompt='lay the towel flat, then carefully fold the towel and then place the folded towel on the black notebook' \
    policy:checkpoint --policy.config=pi05_fold_clothes40 --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_fold_clothes40/pi05_fold_clothes40/25000