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

        # Teacher-forced input: [SOS, target[0], target[1], ..., target[T-2]]
        # SOS = [1, 0, 0] — pen-up flag, zero offsets (matches generate() init)
        target_strokes_input[i, 0, 0] = 1.0  # pen-up
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

def get_beta(epoch: int, stage2_start: int = 10, stage2_end: int = 35) -> float:
    """Return the KL weight β for the given epoch.

    - epoch < stage2_start:                  β = 0.0
    - stage2_start <= epoch < stage2_end:    β linear 0→1
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
    """Return (loss, nll/n, kl/sample).

    NLL is normalised by total valid tokens (per-token loss).
    KL is normalised by batch size (per-sample loss) so it doesn't
    scale with sequence length and stays interpretable in logs.
    """
    nll = compute_nll_loss(target_strokes, y_hat, target_mask)
    B = mu.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    n = target_mask.sum().clamp(min=1)
    loss = nll / n + beta * kl
    return loss, nll / n, kl


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _mps_mem_str(device):
    """Return current MPS allocated memory as a human-readable string, or '' if unavailable."""
    try:
        if device.type == "mps":
            gb = torch.mps.current_allocated_memory() / 1024 ** 3
            return f"mem={gb:.1f}GB"
    except Exception:
        pass
    return ""


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    beta: float,
    grad_clip: float = 5.0,
    use_tqdm: bool = False,
    use_amp: bool = False,
) -> float:
    """Run one full training epoch. Steps scheduler once per batch."""
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    num_batches = 0
    amp_enabled = use_amp and device.type == "cuda"

    it = tqdm(loader, desc="train", leave=False, mininterval=2.0, dynamic_ncols=True) if use_tqdm else loader

    for batch in it:
        style_strokes = batch["style_strokes"].to(device)
        target_strokes = batch["target_strokes"].to(device)
        target_mask = batch["target_mask"].to(device)
        target_strokes_input = batch["target_strokes_input"].to(device)
        text = batch["text"].to(device)
        text_mask = batch["text_mask"].to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp_enabled):
            y_hat, mu, logvar = model(
                target_strokes_input, text, text_mask, style_strokes, use_sampling=True
            )
            loss, _, _ = compute_loss(y_hat, target_strokes, target_mask, mu, logvar, beta)

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

        if use_tqdm:
            mem = _mps_mem_str(device)
            it.set_postfix(loss=f"{loss.item():.4f}", **({mem.split("=")[0]: mem.split("=")[1]} if mem else {}))

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
# DeepWriting dataset loader
# ---------------------------------------------------------------------------

def load_deepwriting(deepwriting_path: str, iam_strokes: np.ndarray) -> tuple:
    """Load DeepWriting .npz files, convert to IAM format, and return merged arrays.

    DeepWriting format: (dx, dy, pen_up) — already pre-normalized
    IAM format:         (eos, dx, dy)    — raw offsets in tablet units

    Steps:
      1. Denormalize DeepWriting dx/dy using their stored mean/std
      2. Rescale to IAM coordinate space using std ratio
      3. Reorder columns to (pen_up, dx, dy)
      4. Concatenate with IAM strokes + texts

    Returns:
        (merged_strokes, merged_texts) as object arrays
    """
    import os

    # Compute IAM dx/dy std for rescaling (use all strokes)
    iam_offsets = np.concatenate([s[:, 1:].astype(np.float32) for s in iam_strokes], axis=0)
    iam_std = iam_offsets.std(axis=0).clip(min=1e-6)  # shape (2,)

    all_dw_strokes = []
    all_dw_texts = []

    for split in ["training", "validation"]:
        path = os.path.join(deepwriting_path, f"deepwriting_{split}.npz")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        d = np.load(path, allow_pickle=True)
        dw_strokes = d["strokes"]   # object array of (T, 3): (dx, dy, pen_up), normalized
        dw_texts = d["texts"]       # string array of shape (N,)
        dw_mean = d["mean"][:2].astype(np.float32)  # (2,) dx/dy mean
        dw_std  = d["std"][:2].astype(np.float32)   # (2,) dx/dy std

        # Scale factor to match IAM units
        scale = iam_std / dw_std.clip(min=1e-6)  # (2,)

        for s, t in zip(dw_strokes, dw_texts):
            s = s.astype(np.float32)
            # Denormalize dx/dy
            s[:, :2] = s[:, :2] * dw_std + dw_mean
            # Rescale to IAM coordinate space
            s[:, :2] *= scale
            # Reorder: (dx, dy, pen_up) → (pen_up, dx, dy)
            s = np.stack([s[:, 2], s[:, 0], s[:, 1]], axis=1)
            # Ensure last point marks end of sequence (match IAM convention)
            s[-1, 0] = 1.0
            all_dw_strokes.append(s)
            all_dw_texts.append(list(str(t)))

    dw_strokes_arr = np.array(all_dw_strokes, dtype=object)
    dw_texts_arr   = np.array([np.array(t) for t in all_dw_texts], dtype=object)

    print(f"  DeepWriting: loaded {len(dw_strokes_arr)} samples from {deepwriting_path}")
    return dw_strokes_arr, dw_texts_arr


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def argparser():
    import argparse
    p = argparse.ArgumentParser(description="Train the Transformer + Style VAE model")
    p.add_argument("--data_path", type=str, default="./data/")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints/transformer/")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--vocab_size_override", type=int, default=None, help="Force vocab_size (use when resuming old checkpoints without saved vocab)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--warmup_steps", type=int, default=1000, help="Linear LR warmup steps")
    p.add_argument("--kl_start", type=int, default=10, help="Epoch to start KL annealing")
    p.add_argument("--kl_end", type=int, default=35, help="Epoch where KL weight reaches 1.0")
    p.add_argument("--max_stroke_len", type=int, default=1000)
    p.add_argument("--deepwriting_path", type=str, default=None, help="Path to deepwriting_dataset/ folder to merge with IAM data")
    p.add_argument("--gdrive_folder_id", type=str, default=None, help="Google Drive folder name (rclone remote path) to auto-backup best checkpoints")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 for MPS, 4-8 for CUDA)")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint_latest.pt")
    p.add_argument("--bias", type=float, default=1.0, help="Sampling bias for mid-epoch generation")
    p.add_argument("--tqdm", action="store_true", help="Show per-batch progress bar with loss and GPU memory")
    p.add_argument("--wandb", action="store_true", help="Log metrics to Weights & Biases")
    p.add_argument("--wandb_project", type=str, default="handwriting-transformer")
    p.add_argument("--no_amp", action="store_true", help="Disable BF16 autocast (always off on MPS)")
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
    strokes = np.load(os.path.join(args.data_path, "strokes.npy"), allow_pickle=True, encoding="bytes")
    with open(os.path.join(args.data_path, "sentences.txt")) as f:
        raw_texts = f.read().splitlines()
    texts = np.array([np.array(list(t)) for t in raw_texts], dtype=object)

    # Optionally merge DeepWriting dataset
    if args.deepwriting_path:
        print(f"Loading DeepWriting dataset from {args.deepwriting_path} ...")
        dw_strokes, dw_texts = load_deepwriting(args.deepwriting_path, strokes)
        strokes = np.concatenate([strokes, dw_strokes])
        texts   = np.concatenate([texts, dw_texts])
        print(f"  Combined dataset: {len(strokes)} samples (IAM + DeepWriting)")

    # Build vocab from all characters in training texts
    all_chars = sorted(set(c for t in texts for c in t))
    char_to_id = {c: i for i, c in enumerate(all_chars)}
    vocab_size = len(char_to_id)

    # If resuming, pre-load checkpoint to restore char_to_id and vocab_size
    # so datasets and model architecture match the saved weights exactly.
    latest_path = os.path.join(args.checkpoint_dir, "checkpoint_latest.pt")
    if args.resume and os.path.exists(latest_path):
        _ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        if "char_to_id" in _ckpt:
            char_to_id = _ckpt["char_to_id"]
            vocab_size = len(char_to_id)
            print(f"  Restored char_to_id from checkpoint: vocab_size={vocab_size}")
        else:
            # Infer vocab_size directly from the saved embedding weight shape
            ckpt_vocab = _ckpt["model_state"]["text_encoder.embedding.weight"].shape[0]
            if ckpt_vocab != vocab_size:
                print(f"  vocab_size mismatch (data={vocab_size}, ckpt={ckpt_vocab}) — truncating char_to_id")
                vocab_size = ckpt_vocab
                char_to_id = {c: i for c, i in char_to_id.items() if i < vocab_size}
        del _ckpt

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

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # Model
    model = HandWritingSynthesisTransformer(vocab_size=vocab_size).to(device)

    # Optimizer: linear warmup then cosine decay, stepped per batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6, end_factor=1.0, total_iters=args.warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_steps - args.warmup_steps, 1)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[args.warmup_steps],
    )

    start_epoch = 0
    best_val_loss = float("inf")

    # Resume
    if args.resume and os.path.exists(latest_path):
        start_epoch, best_val_loss, _ = load_checkpoint(latest_path, model, optimizer, scheduler, device)
        start_epoch += 1
        print(f"Resumed from epoch {start_epoch - 1}")

    # WandB
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    use_amp = not args.no_amp

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        beta = get_beta(epoch, args.kl_start, args.kl_end)

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device,
            beta, args.grad_clip, use_tqdm=args.tqdm, use_amp=use_amp,
        )
        val_loss = validation_epoch(model, valid_loader, device, beta)

        _log = print if not args.tqdm else __import__("tqdm").tqdm.write
        _log(f"Epoch {epoch:3d} | beta={beta:.3f} | train={train_loss:.4f} | val={val_loss:.4f}")

        if args.wandb:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                       "beta": beta, "lr": optimizer.param_groups[0]["lr"]})

        # Save latest checkpoint every epoch
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "beta": beta,
                "train_mean": train_mean,
                "train_std": train_std,
                "vocab_size": vocab_size,
                "char_to_id": char_to_id,
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
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "vocab_size": vocab_size,
                    "char_to_id": char_to_id,
                },
                best_path,
            )
            _log(f"  → New best: {best_val_loss:.4f}")
            if args.gdrive_folder_id:
                try:
                    import subprocess
                    subprocess.Popen([
                        "rclone", "copy", best_path,
                        f"gdrive:{args.gdrive_folder_id}",
                        "--drive-use-trash=false"
                    ])
                    _log("  → Uploading to Google Drive...")
                except Exception as e:
                    _log(f"  → Drive upload failed: {e}")

        # Save epoch-numbered checkpoint every 10 epochs (H100 rollback safety)
        if (epoch + 1) % 10 == 0:
            epoch_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1:03d}.pt")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "beta": beta,
                    "train_mean": train_mean,
                    "train_std": train_std,
                    "vocab_size": vocab_size,
                    "char_to_id": char_to_id,
                },
                epoch_path,
            )

    if args.wandb:
        import wandb
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
