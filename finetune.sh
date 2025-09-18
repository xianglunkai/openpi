# 1. Convert your data to a LeRobot dataset

uv run examples/mobile_aloha_AgileX/convert_aloha_data_to_lerobot.py --raw_dir /home/xlk/work/my_cool_dataset/adjust_bottle_simple --repo_id adjust_bottle_simple --push_to_hub False


# 2. Defining training configs and running training

# To fine-tune a base model on your own data, you need to define configs for data processing and training. We provide example configs with detailed comments for LIBERO below, which you can modify for your own dataset:

# LiberoInputs and LiberoOutputs: Defines the data mapping from the LIBERO environment to the model and vice versa. Will be used for both, training and inference.
# LeRobotLiberoDataConfig: Defines how to process raw LIBERO data from LeRobot dataset for training.
# TrainConfig: Defines fine-tuning hyperparameters, data config, and weight loader.
# We provide example fine-tuning configs for π₀, π₀-FAST, and π₀.₅ on LIBERO data.

# # 3. Compute the normalization statistics for the training data
# uv run scripts/compute_norm_stats.py --config-name pi05_libero

# # 4. Start fine-tuning
# XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment --overwrite