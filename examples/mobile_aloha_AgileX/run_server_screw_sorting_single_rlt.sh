# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com

# uv run scripts/server_rlt_policy.py \
#     --port 8000 \
#     --shared_prefix_inference \
#     --default_prompt 'Please sort and return the silver screws in the grey box to their proper places.' \
#     --checkpoint_dir '/home/xlk/work/openpi/checkpoints/pi05_cobot_screw_sorting_single_rlt/pi05_cobot_screw_sorting_single_rlt/9999' \
#     --config pi05_cobot_screw_sorting_single_two_staged_rlt


uv run scripts/server_rlt_policy.py \
    --port 8000 \
    --shared_prefix_inference \
    --default_prompt 'Please sort and return the silver screws in the grey box to their proper places.' \
    --checkpoint_dir '/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_screw_sorting_single_rlt/rlt-iter1' \
    --config pi05_cobot_screw_sorting_single_joint_vla_rlt