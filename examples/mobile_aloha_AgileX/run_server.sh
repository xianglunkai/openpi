# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com
uv run scripts/serve_policy.py \
    --env ALOHA \
    --default_prompt='Carefully fold the towel and then place the folded towel on the black notebook' \
    policy:checkpoint --policy.config=pi05_cobot --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_fold_towel/checkpoint-20k \




    # policy:checkpoint --policy.config=pi0_aloha --policy.dir=/home/xlk/work/openpi/models/checkpoints_pi0_aloha_towel/pi0_aloha_towel \
        # policy:checkpoint --policy.config=pi0_aloha --policy.dir=/home/xlk/work/openpi/models/checkpoints_pi0_aloha_towel/pi0_aloha_towel \
  