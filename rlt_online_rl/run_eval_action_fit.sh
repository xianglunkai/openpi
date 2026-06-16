python scripts/offline/offline_train_from_replay.py \
  --replay-path runs/screw_sorting/replay/replay_journal.pkl \
  --steps 20000 \
  --batch-size 128 \
  --seed 0 \
  --bc-weight 10.0 \
  --q-weight 0.0 \
  --delta-weight 10.0 \
  --fixed-std 0.0 \
  --output-dir runs/screw_sorting \
  --phase warmup 


# 只看 policy 应贴近 ref 的子集
python scripts/offline/eval_action_fit.py \
  --replay-path runs/screw_sorting/replay/replay_journal.pkl \
  --model-dir runs/screw_sorting \
  --actor-mode mean \
  --phase warmup 


python scripts/offline/visualize_offline_training.py \
  --train-dir runs/screw_sorting