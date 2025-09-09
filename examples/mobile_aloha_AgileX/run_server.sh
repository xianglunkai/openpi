


# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'

uv run scripts/serve_policy.py \
    --env ALOHA \
    --default_prompt='fold the towel' \
    policy:checkpoint --policy.config=pi0_aloha --policy.dir=s3://openpi-assets/checkpoints/pi0_aloha_towel \
    # policy:checkpoint --policy.config=pi05_aloha --policy.dir=gs://openpi-assets/checkpoints/pi05_base \