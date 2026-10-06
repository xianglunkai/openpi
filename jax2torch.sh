# uv run examples/convert_jax_model_to_pytorch.py \
#     --checkpoint_dir /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_shirt/checkpoint-15k \
#     --config_name pi05_cobot_fold_shirt \
#     --output_path /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_shirt/checkpoint-15k-converted \
#     --precision bfloat16

uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_mobile_cobot_take_me_tissues/49999\
    --config_name pi05_take_me_tissues \
    --output_path /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_mobile_cobot_take_me_tissues/49999-converted \
    --precision bfloat16