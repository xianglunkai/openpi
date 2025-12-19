# uv run examples/convert_jax_model_to_pytorch.py \
#     --checkpoint_dir /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_shirt/checkpoint-15k \
#     --config_name pi05_cobot_fold_shirt \
#     --output_path /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_shirt/checkpoint-15k-converted \
#     --precision bfloat16

uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_towel/checkpoint-90k \
    --config_name pi05_cobot_fold_towel \
    --output_path /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_towel/checkpoint-90k-converted \
    --precision bfloat16