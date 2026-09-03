# Same as run_server_screw_sorting_single.sh — RECAP CFG is loaded from the train config.
export HF_ENDPOINT=https://hf-mirror.com

uv run scripts/serve_policy.py \
    --env COBOT \
    --default_prompt='Please sort and return the silver screws in the grey box to their proper places.' \
    policy:checkpoint --policy.config=pi05_cobot_screw_sorting_single_recap --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot_screw_sorting_single_recap/pi05_cobot_screw_sorting_single_recap/19999
