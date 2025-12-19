# 设置内存限制
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
# export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# # 或者使用CPU模式
export CUDA_VISIBLE_DEVICES=""  # 使用CPU
uv run examples/convert_from_jax.py \
    --jax_path /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_shirt/checkpoint-15k \
    --prompt 'Carefully using its two arms to fold the shirt softly.' \
    --output /home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_shirt/checkpoint-15k-pk/converted_checkpoint.pkl \
    --tokenizer_path /home/xlk/work/openpi/models/paligemma-3b-pt-224
