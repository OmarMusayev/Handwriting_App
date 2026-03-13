#!/bin/bash
# Final H100 run — IAM + DeepWriting, 300 epochs
python train_transformer.py \
  --epochs 300 \
  --batch_size 256 \
  --lr 1e-4 \
  --grad_clip 5.0 \
  --warmup_steps 1000 \
  --kl_start 60 \
  --kl_end 180 \
  --max_stroke_len 1000 \
  --deepwriting_path deepwriting_dataset/ \
  --checkpoint_dir checkpoints/transformer/ \
  --gdrive_folder_id handwriting-checkpoints \
  --num_workers 8 \
  --tqdm \
  --wandb
