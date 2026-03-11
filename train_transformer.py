"""
train_transformer.py — Training script for the Transformer + Style VAE handwriting model.

This module contains:
  - TransformerHandwritingDataset: wraps IAM stroke data with within-sample style/target split
  - collate_fn: pads variable-length sequences and builds teacher-forced inputs
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TransformerHandwritingDataset(Dataset):
    """Dataset that wraps IAM stroke data for Transformer + Style VAE training.

    Each item performs a within-sample style/target split:
      - A random prefix (split_lo–split_hi fraction) becomes the style input to the VAE.
      - The remaining suffix becomes the decoder generation target.
    """

    def __init__(
        self,
        strokes: np.ndarray,       # object array of variable-length stroke arrays, each (T, 3)
        texts: np.ndarray,         # object array of char arrays, same length as strokes
        char_to_id: dict,          # char → int mapping
        train_mean: np.ndarray,    # (2,) mean for dx/dy normalisation
        train_std: np.ndarray,     # (2,) std for dx/dy normalisation
        max_stroke_len: int = 1000,
        split_lo: float = 0.2,    # min fraction for style prefix
        split_hi: float = 0.4,    # max fraction for style prefix
    ):
        self.strokes = strokes
        self.texts = texts
        self.char_to_id = char_to_id
        self.train_mean = train_mean.astype(np.float32)
        self.train_std = train_std.astype(np.float32)
        self.max_stroke_len = max_stroke_len
        self.split_lo = split_lo
        self.split_hi = split_hi

    def __len__(self) -> int:
        return len(self.strokes)

    def __getitem__(self, idx: int) -> dict:
        # 1. Copy and clamp to max_stroke_len
        stroke = self.strokes[idx].copy().astype(np.float32)
        stroke = stroke[: self.max_stroke_len]

        # 2. Normalise dx/dy (cols 1 and 2); leave eos flag (col 0) untouched
        stroke[:, 1:] = (stroke[:, 1:] - self.train_mean) / self.train_std

        # 3. Random split point: k in [split_lo, split_hi] * len(stroke)
        total = len(stroke)
        k = int(total * random.uniform(self.split_lo, self.split_hi))
        k = max(1, min(k, total - 1))  # ensure both parts non-empty

        style_strokes = stroke[:k]
        target_strokes = stroke[k:]

        # 4. Convert text chars to int ids (unknown → 0)
        char_arr = self.texts[idx]
        text_ids = np.array(
            [self.char_to_id.get(c, 0) for c in char_arr], dtype=np.int64
        )
        text_mask = np.ones(len(text_ids), dtype=np.float32)

        return {
            "style_strokes": style_strokes,   # (style_len, 3)
            "target_strokes": target_strokes,  # (target_len, 3)
            "text": text_ids,                  # (text_len,)
            "text_mask": text_mask,            # (text_len,)
        }


def collate_fn(batch: list) -> dict:
    """Pad variable-length sequences and build teacher-forced inputs.

    Args:
        batch: list of dicts from TransformerHandwritingDataset.__getitem__

    Returns:
        dict of tensors:
            style_strokes        (B, max_style_len, 3)
            style_mask           (B, max_style_len)
            target_strokes       (B, max_target_len, 3)   — unshifted labels
            target_mask          (B, max_target_len)
            target_strokes_input (B, max_target_len, 3)   — teacher-forced input
            text                 (B, max_text_len)
            text_mask            (B, max_text_len)
    """
    B = len(batch)

    # Compute max lengths in this batch
    max_style_len = max(item["style_strokes"].shape[0] for item in batch)
    max_target_len = max(item["target_strokes"].shape[0] for item in batch)
    max_text_len = max(item["text"].shape[0] for item in batch)

    # Allocate output tensors (zero-initialised → zero padding)
    style_strokes = torch.zeros(B, max_style_len, 3, dtype=torch.float32)
    style_mask = torch.zeros(B, max_style_len, dtype=torch.float32)

    target_strokes = torch.zeros(B, max_target_len, 3, dtype=torch.float32)
    target_mask = torch.zeros(B, max_target_len, dtype=torch.float32)
    target_strokes_input = torch.zeros(B, max_target_len, 3, dtype=torch.float32)

    text = torch.zeros(B, max_text_len, dtype=torch.long)
    text_mask = torch.zeros(B, max_text_len, dtype=torch.float32)

    for i, item in enumerate(batch):
        s_len = item["style_strokes"].shape[0]
        t_len = item["target_strokes"].shape[0]
        tx_len = item["text"].shape[0]

        # Style
        style_strokes[i, :s_len, :] = torch.from_numpy(item["style_strokes"])
        style_mask[i, :s_len] = 1.0

        # Target labels (unshifted)
        tgt = torch.from_numpy(item["target_strokes"])
        target_strokes[i, :t_len, :] = tgt
        target_mask[i, :t_len] = 1.0

        # Teacher-forced input: [zeros(1,3), target[:-1, :]]
        # Position 0 stays as the zero start token (already allocated)
        if t_len > 1:
            target_strokes_input[i, 1:t_len, :] = tgt[:-1, :]

        # Text
        text[i, :tx_len] = torch.from_numpy(item["text"])
        text_mask[i, :tx_len] = torch.from_numpy(item["text_mask"])

    return {
        "style_strokes": style_strokes,
        "style_mask": style_mask,
        "target_strokes": target_strokes,
        "target_mask": target_mask,
        "target_strokes_input": target_strokes_input,
        "text": text,
        "text_mask": text_mask,
    }
