# uv run scripts/serve_policy.py --env ALOHA --default_prompt='Pick up the bottle on the table headup with the correct arm'
export HF_ENDPOINT=https://hf-mirror.com
uv run scripts/serve_policy.py \
    --env ALOHA \
    --default_prompt='Carefully using its two arms, the robot grasps the bottle and pours water with steady precision into the cup without spilling a drop.' \
    policy:checkpoint --policy.config=pi05_cobot_pour_water --policy.dir=/home/xlk/work/openpi/checkpoints/pi05_cobot/pi05_cobot_pour_water/checkpoint-30k \

#--default_prompt='Carefully using its two arms, the robot grasps the bottle and pours water with steady precision into the cup without spilling a drop.' \
#--default_prompt='Carefully using its two arms, pour me a cup of water without spilling a drop.' \