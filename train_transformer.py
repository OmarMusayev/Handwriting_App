"""
train_transformer.py — Training script for the Transformer + Style VAE handwriting model.

This module contains:
  - TransformerHandwritingDataset: wraps IAM stroke data with within-sample style/target split
  - collate_fn: pads variable-length sequences and builds teacher-forced inputs
  - get_beta: KL annealing schedule
  - compute_loss: per-sample normalised NLL + KL loss
  - train_epoch: one training epoch with grad clipping
  - validation_epoch: one validation epoch (no grad)
  - save_checkpoint / load_checkpoint: checkpoint I/O
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from utils.model_utils import compute_nll_loss
from models.transformer_synthesis import HandWritingSynthesisTransformer


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


# ---------------------------------------------------------------------------
# KL annealing schedule
# ---------------------------------------------------------------------------

def get_beta(epoch: int, stage2_start: int = 20, stage2_end: int = 60) -> float:
    """Return the KL weight β for the given epoch.

    - epoch < stage2_start:                  β = 0.0
    - stage2_start <= epoch < stage2_end:    β = (epoch - stage2_start) / (stage2_end - stage2_start)
    - epoch >= stage2_end:                   β = 1.0
    """
    if epoch < stage2_start:
        return 0.0
    if epoch >= stage2_end:
        return 1.0
    return (epoch - stage2_start) / (stage2_end - stage2_start)


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

def compute_loss(
    y_hat: torch.Tensor,          # (batch, seq_len, 121)
    target_strokes: torch.Tensor,  # (batch, seq_len, 3)
    target_mask: torch.Tensor,    # (batch, seq_len)  float 1=valid 0=pad
    mu: torch.Tensor,             # (batch, latent_dim)
    logvar: torch.Tensor,         # (batch, latent_dim)
    beta: float,
) -> tuple:
    """Return (loss, nll/n, kl/n) where n = total valid timesteps.

    Both nll and kl are normalised by n (total valid positions across the batch),
    giving a per-token loss that is stable across variable batch sizes.
    """
    nll = compute_nll_loss(target_strokes, y_hat, target_mask)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    n = target_mask.sum().clamp(min=1)
    loss = (nll + beta * kl) / n
    return loss, nll / n, kl / n


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    grad_clip: float = 1.0,
) -> float:
    """Run one full training epoch.

    Returns:
        Average total loss across all batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move tensors to device
        style_strokes = batch["style_strokes"].to(device)
        target_strokes = batch["target_strokes"].to(device)
        target_mask = batch["target_mask"].to(device)
        target_strokes_input = batch["target_strokes_input"].to(device)
        text = batch["text"].to(device)
        text_mask = batch["text_mask"].to(device)

        optimizer.zero_grad()

        y_hat, mu, logvar = model(
            target_strokes_input, text, text_mask, style_strokes, use_sampling=True
        )

        loss, _, _ = compute_loss(y_hat, target_strokes, target_mask, mu, logvar, beta)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

def validation_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> float:
    """Run one full validation epoch (no gradient updates).

    Returns:
        Average total loss across all batches.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            style_strokes = batch["style_strokes"].to(device)
            target_strokes = batch["target_strokes"].to(device)
            target_mask = batch["target_mask"].to(device)
            target_strokes_input = batch["target_strokes_input"].to(device)
            text = batch["text"].to(device)
            text_mask = batch["text_mask"].to(device)

            y_hat, mu, logvar = model(
                target_strokes_input, text, text_mask, style_strokes, use_sampling=True
            )

            loss, _, _ = compute_loss(y_hat, target_strokes, target_mask, mu, logvar, beta)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str) -> None:
    """Save a checkpoint dict to path, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    device: torch.device,
) -> tuple:
    """Load a checkpoint and restore model/optimizer/scheduler state.

    Returns:
        (epoch, best_val_loss, beta)
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint["epoch"], checkpoint["best_val_loss"], checkpoint["beta"]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def argparser():
    import argparse
    p = argparse.ArgumentParser(description="Train the Transformer + Style VAE model")
    p.add_argument("--data_path", type=str, default="./data/")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/transformer/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--max_stroke_len", type=int, default=1000)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pt")
    p.add_argument("--bias", type=float, default=1.0, help="Sampling bias for mid-epoch generation")
    return p.parse_args()


def main():
    args = argparser()

    # Device: prefer MPS (M-series Mac), fallback CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    strokes = np.load(os.path.join(args.data_path, "strokes.npy"), allow_pickle=True)
    with open(os.path.join(args.data_path, "sentences.txt")) as f:
        raw_texts = f.read().splitlines()

    # Build vocab from all characters in training texts
    all_chars = sorted(set("".join(raw_texts)))
    char_to_id = {c: i for i, c in enumerate(all_chars)}
    vocab_size = len(char_to_id)
    texts = np.array([np.array(list(t)) for t in raw_texts], dtype=object)

    # Train/valid split (90/10, same as existing pipeline)
    n = len(strokes)
    n_train = int(0.9 * n)
    idx = np.random.permutation(n)
    train_idx, valid_idx = idx[:n_train], idx[n_train:]

    # Compute train_mean, train_std from training strokes
    train_strokes_raw = [strokes[i].astype(np.float32) for i in train_idx]
    all_offsets = np.concatenate([s[:, 1:] for s in train_strokes_raw], axis=0)
    train_mean = all_offsets.mean(axis=0)  # shape (2,)
    train_std  = all_offsets.std(axis=0).clip(min=1e-6)

    train_ds = TransformerHandwritingDataset(
        strokes=strokes[train_idx],
        texts=texts[train_idx],
        char_to_id=char_to_id,
        train_mean=train_mean,
        train_std=train_std,
        max_stroke_len=args.max_stroke_len,
    )
    valid_ds = TransformerHandwritingDataset(
        strokes=strokes[valid_idx],
        texts=texts[valid_idx],
        char_to_id=char_to_id,
        train_mean=train_mean,
        train_std=train_std,
        max_stroke_len=args.max_stroke_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # Model
    model = HandWritingSynthesisTransformer(vocab_size=vocab_size).to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume
    latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    if args.resume and os.path.exists(latest_path):
        start_epoch, best_val_loss, _ = load_checkpoint(latest_path, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        beta = get_beta(epoch)

        train_loss = train_epoch(model, train_loader, optimizer, device, beta, args.grad_clip)
        val_loss   = validation_epoch(model, valid_loader, device, beta)
        scheduler.step()

        print(f"Epoch {epoch:3d} | beta={beta:.3f} | train={train_loss:.4f} | val={val_loss:.4f}")

        # Save latest checkpoint every epoch
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "beta": beta,
            },
            latest_path,
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, "checkpoint_best.pt")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "beta": beta,
                },
                best_path,
            )
            print(f"  → New best: {best_val_loss:.4f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
