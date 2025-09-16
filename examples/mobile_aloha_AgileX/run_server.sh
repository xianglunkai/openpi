# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com
uv run scripts/serve_policy.py \
    --env ALOHA \
    --default_prompt='fold the towel' \
    policy:checkpoint --policy.config=pi0_aloha --policy.dir=/home/xlk/work/openpi/models/checkpoints_pi0_aloha_towel/pi0_aloha_towel \

    # policy:checkpoint --policy.config=pi05_aloha --policy.dir=/home/xlk/work/openpi/models/checkpoints_pi05/pi05_base \
    # policy:checkpoint --policy.config=pi0_aloha --policy.dir=/home/xlk/work/openpi/models/checkpoints_pi0_aloha_towel/pi0_aloha_towel \
  