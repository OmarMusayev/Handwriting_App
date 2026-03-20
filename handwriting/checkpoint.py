from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .data import TextVocab
from .tokenizers import RadiusCodebooks, StrokeVocab, TokenizerSpec, tokenizer_spec_to_dict


@dataclass
class TrainState:
    step: int = 0
    completed_epochs: int = 0
    best_val_loss: float = float("inf")
    best_val_epoch: Optional[int] = None
    best_val_step: Optional[int] = None
    best_val_ppl: Optional[float] = None
    best_checkpoint_path: Optional[str] = None
    resume_path: Optional[str] = None


def save_checkpoint(
    out_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    state: TrainState,
    args,
    text_vocab: TextVocab,
    stroke_vocab: StrokeVocab,
    tokenizer_spec: TokenizerSpec,
    radius_codebooks: RadiusCodebooks,
    eval_history: List[dict],
    panel_history: List[dict],
    eos_history: List[dict],
    filename: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / filename
    model_to_save = model.module if hasattr(model, "module") else model
    best_checkpoint_path = str(save_path) if filename == "best.pt" else state.best_checkpoint_path
    payload = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": state.step,
        "completed_epochs": state.completed_epochs,
        "best_val_loss": state.best_val_loss,
        "best_val_epoch": state.best_val_epoch,
        "best_val_step": state.best_val_step,
        "best_val_ppl": state.best_val_ppl,
        "best_checkpoint_path": best_checkpoint_path,
        "resume_path": state.resume_path,
        "args": vars(args),
        "text_vocab_itos": text_vocab.itos,
        "tokenizer_spec": tokenizer_spec_to_dict(tokenizer_spec),
        "stroke_vocab": {"vocab_size": stroke_vocab.vocab_size},
        "radius_edges": radius_codebooks.legacy_radius_edges(),
        "radius_codebooks": radius_codebooks.to_dict(),
        "eval_history": eval_history,
        "panel_history": panel_history,
        "eos_history": eos_history,
    }
    torch.save(payload, save_path)
    return save_path


def load_checkpoint_payload(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_model_weights(model: torch.nn.Module, checkpoint_payload: Dict[str, Any]) -> None:
    if "model" not in checkpoint_payload:
        raise KeyError("Checkpoint is missing model weights")
    model.load_state_dict(checkpoint_payload["model"])


def restore_optimizer_scheduler_scaler(
    *,
    checkpoint_payload: Dict[str, Any],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    reset_optimizer: bool,
    reset_scheduler: bool,
    reset_scaler: bool,
) -> None:
    if not reset_optimizer and checkpoint_payload.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint_payload["optimizer"])
    if scheduler is not None and not reset_scheduler and checkpoint_payload.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint_payload["scheduler"])
    if scaler is not None and not reset_scaler and checkpoint_payload.get("scaler") is not None:
        scaler.load_state_dict(checkpoint_payload["scaler"])


def restore_train_state(checkpoint_payload: Dict[str, Any], resume_path: Path) -> Tuple[TrainState, List[dict], List[dict], List[dict]]:
    state = TrainState(
        step=int(checkpoint_payload.get("step", 0)),
        completed_epochs=int(checkpoint_payload.get("completed_epochs", checkpoint_payload.get("epoch", 0))),
        best_val_loss=float(checkpoint_payload.get("best_val_loss", float("inf"))),
        best_val_epoch=(
            int(checkpoint_payload["best_val_epoch"])
            if checkpoint_payload.get("best_val_epoch") is not None
            else None
        ),
        best_val_step=(
            int(checkpoint_payload["best_val_step"])
            if checkpoint_payload.get("best_val_step") is not None
            else None
        ),
        best_val_ppl=(
            float(checkpoint_payload["best_val_ppl"])
            if checkpoint_payload.get("best_val_ppl") is not None
            else None
        ),
        best_checkpoint_path=str(checkpoint_payload.get("best_checkpoint_path") or ""),
        resume_path=str(resume_path),
    )
    if not state.best_checkpoint_path:
        state.best_checkpoint_path = str(resume_path)
    eval_history = list(checkpoint_payload.get("eval_history", []))
    panel_history = list(checkpoint_payload.get("panel_history", []))
    eos_history = list(checkpoint_payload.get("eos_history", []))
    return state, eval_history, panel_history, eos_history
