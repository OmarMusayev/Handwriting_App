#!/bin/bash
python train_transformer.py --epochs 300 --batch_size 256 --max_stroke_len 1000 --lr 3e-5 --checkpoint_dir checkpoints/transformer/ --deepwriting_path deepwriting_dataset/ --tqdm --gdrive_folder_id handwriting-checkpoints --resume --vocab_size_override 77
