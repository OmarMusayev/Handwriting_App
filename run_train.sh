#!/bin/bash
# Overnight test run — IAM only, 50 epochs, M4 Mac
python train_transformer.py \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --grad_clip 5.0 \
  --warmup_steps 500 \
  --kl_start 10 \
  --kl_end 35 \
  --checkpoint_dir checkpoints/transformer_v2/ \
  --tqdm \
  --no_amp
