from __future__ import annotations

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
    scheduler_type: str = "cosine",
    step_lr_every: int = 0,
    lr_decay: float = 0.5,
):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps - 1))

    if scheduler_type == "step":
        step_lr_every = max(1, int(step_lr_every) if int(step_lr_every) > 0 else total_steps)
        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            step = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_lr_every,
                gamma=float(lr_decay),
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, step],
                milestones=[warmup_steps],
            )
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_lr_every,
            gamma=float(lr_decay),
        )

    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=optimizer.param_groups[0]["lr"] * min_lr_ratio,
    )


def build_optimizer(model, args) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(float(args.adam_beta1), float(args.adam_beta2)),
        eps=float(args.adam_eps),
        weight_decay=args.weight_decay,
    )


def build_scaler(args, device: torch.device):
    if device.type == "cuda":
        return torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    return None
