# uv run --no-sync scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com

uv run --no-sync scripts/serve_policy.py \
    --env COBOT \
    --default_prompt='Please sort and return the silver screws in the grey box to their proper places.' \
    policy:checkpoint --policy.config=pi05_cobot_screw_sorting_single_rtc --policy.dir=/home/xlk/work/openpi/checkpoints/zoey_test_pi0.5_checkpoint \
