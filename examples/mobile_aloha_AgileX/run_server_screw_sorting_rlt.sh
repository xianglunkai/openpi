# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com

uv run scripts/serve_rlt_policy.py \
    --env COBOT \
    --port 8000 \
    --shared-prefix-inference
    --default_prompt='Please sort and return the silver screws in the grey box to their proper places.' \
    policy:checkpoint ---policy.config=pi05_cobot_screw_sorting_rlt --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_screw_sorting_rlt/29999 \
