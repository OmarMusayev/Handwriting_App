from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .checkpoint import (
    TrainState,
    load_checkpoint_payload,
    restore_model_weights,
    restore_optimizer_scheduler_scaler,
    restore_train_state,
    save_checkpoint,
)
from .config import load_decoding_modes, load_word_panel, parse_args
from .data import ProcessedWordDataset, ProcessedWordSplit, TextVocab, collate_batch, load_processed_npz
from .evaluation import compute_eos_diagnostics, evaluate_panel_if_main, run_eval_if_needed, run_tokenizer_diagnostics
from .model import build_model
from .optim import build_optimizer, build_scaler, build_scheduler
from .reporting import (
    write_eos_diagnostics_artifacts,
    write_run_summary,
    write_style_cluster_conditioning_artifacts,
    write_writer_conditioning_artifacts,
)
from .seed import build_worker_init_fn, seed_everything
from .style_clusters import load_panel_clusters, load_style_cluster_map
from .tokenizers import (
    PolarOffsetTokenizer,
    StrokeVocab,
    collect_radius_codebooks_from_offsets,
    resolve_tokenizer_spec,
    tokenizer_spec_to_dict,
)
from .utils import cleanup_ddp, is_main_process, maybe_tqdm, resolve_bundle_path, setup_ddp
from .writers import load_panel_writers, load_writer_vocab


def record_validation_result(
    *,
    result: Optional[Tuple[float, int]],
    epoch: int,
    trigger: str,
    state: TrainState,
    rank: int,
    out_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    args,
    text_vocab: TextVocab,
    stroke_vocab: StrokeVocab,
    tokenizer_spec,
    radius_codebooks,
    eval_history: List[dict],
    panel_history: List[dict],
    eos_history: List[dict],
) -> None:
    if result is None or not is_main_process(rank):
        return

    val_loss_sum, val_token_count = result
    val_loss = val_loss_sum / max(val_token_count, 1)
    val_ppl = math.exp(min(val_loss, 20))
    is_new_best = bool(val_loss < state.best_val_loss)
    entry = {
        "step": int(state.step),
        "epoch": int(epoch),
        "split": "val",
        "trigger": trigger,
        "nll": float(val_loss),
        "ppl": float(val_ppl),
        "is_new_best": is_new_best,
    }
    eval_history.append(entry)

    if is_new_best:
        prev_best = state.best_val_loss
        improvement = prev_best - val_loss if math.isfinite(prev_best) else float("inf")
        prev_best_text = f"{prev_best:.6f}" if math.isfinite(prev_best) else "inf"
        improvement_text = f"{improvement:.6f}" if math.isfinite(improvement) else "inf"
        print(
            f"[new best] epoch={epoch} step={state.step} val_nll={val_loss:.6f} "
            f"val_ppl={val_ppl:.3f} prev_best={prev_best_text} improvement={improvement_text}"
        )
        state.best_val_loss = float(val_loss)
        state.best_val_epoch = int(epoch)
        state.best_val_step = int(state.step)
        state.best_val_ppl = float(val_ppl)
        state.best_checkpoint_path = str((out_dir / "best.pt").resolve())
        save_checkpoint(
            out_dir,
            model,
            optimizer,
            scheduler,
            scaler,
            state,
            args,
            text_vocab,
            stroke_vocab,
            tokenizer_spec,
            radius_codebooks,
            eval_history,
            panel_history,
            eos_history,
            filename="best.pt",
        )
        print(f"[checkpoint] saved best.pt at epoch {epoch}")


def main() -> None:
    args = parse_args()
    bundle_root = Path(args.bundle_root)

    ddp, rank, world_size, local_rank, device = setup_ddp()
    try:
        seed_everything(args.seed)
        out_dir = resolve_bundle_path(args.out_dir, bundle_root)
        sample_dir = resolve_bundle_path(args.sample_dir, bundle_root)
        assert out_dir is not None and sample_dir is not None
        if is_main_process(rank):
            out_dir.mkdir(parents=True, exist_ok=True)
            sample_dir.mkdir(parents=True, exist_ok=True)

        if bool(getattr(args, "use_writer_conditioning", False)) and bool(
            getattr(args, "use_style_cluster_conditioning", False)
        ):
            raise ValueError(
                "use_writer_conditioning and use_style_cluster_conditioning are mutually exclusive in this bundle"
            )

        train_npz = resolve_bundle_path(args.train_npz, bundle_root)
        val_npz = resolve_bundle_path(args.val_npz, bundle_root)
        test_npz = resolve_bundle_path(args.test_npz, bundle_root) if args.test_npz else None
        tokenizer_config_path = resolve_bundle_path(args.tokenizer_config, bundle_root) if args.tokenizer_config else None
        eval_word_panel_path = resolve_bundle_path(args.eval_word_panel_config, bundle_root)
        decoding_modes_path = resolve_bundle_path(args.decoding_modes_config, bundle_root)
        writer_map_path = resolve_bundle_path(args.writer_id_map, bundle_root) if getattr(args, "use_writer_conditioning", False) else None
        panel_writers_path = (
            resolve_bundle_path(args.panel_writers_config, bundle_root)
            if getattr(args, "use_writer_conditioning", False)
            else None
        )
        style_cluster_map_path = (
            resolve_bundle_path(args.style_cluster_map, bundle_root)
            if getattr(args, "use_style_cluster_conditioning", False)
            else None
        )
        panel_clusters_path = (
            resolve_bundle_path(args.panel_clusters_config, bundle_root)
            if getattr(args, "use_style_cluster_conditioning", False)
            else None
        )
        resume_path = resolve_bundle_path(args.resume, bundle_root) if args.resume else None
        assert (
            train_npz is not None
            and val_npz is not None
            and eval_word_panel_path is not None
            and decoding_modes_path is not None
        )

        eval_words = load_word_panel(eval_word_panel_path)
        decoding_modes = load_decoding_modes(decoding_modes_path)
        args.eval_words = eval_words
        args.decoding_modes = [
            {
                "name": mode.name,
                "temperature": mode.temperature,
                "top_k": mode.top_k,
                "greedy": mode.greedy,
            }
            for mode in decoding_modes
        ]

        if is_main_process(rank):
            print("Loading processed NPZ splits...")
        train_split = load_processed_npz(train_npz, max_samples=args.max_train_samples or None)
        val_split = load_processed_npz(val_npz, max_samples=args.max_val_samples or None)
        test_split = load_processed_npz(test_npz, max_samples=args.max_test_samples or None) if test_npz else None
        writer_vocab = None
        panel_writers: List[dict] = []
        style_cluster_map = None
        panel_clusters: List[dict] = []
        dataset_writer_to_index = {"<default>": 0}
        dataset_writer_unknown_index = 0
        dataset_writer_unseen_policy = "map_to_unk"
        dataset_style_cluster_by_writer: Dict[str, int] = {}
        dataset_default_style_cluster_id = 0
        if args.use_writer_conditioning:
            assert writer_map_path is not None and panel_writers_path is not None
            writer_vocab = load_writer_vocab(writer_map_path)
            panel_writers = load_panel_writers(panel_writers_path, writer_vocab)

            train_writer_ids = {str(value) for value in train_split.writer_ids}
            expected_train_writer_ids = set(writer_vocab.train_writer_ids)
            missing_from_map = sorted(train_writer_ids - expected_train_writer_ids)
            if missing_from_map:
                raise ValueError(
                    "writer_id_map.json does not match the current train split: "
                    f"missing_from_map={missing_from_map[:10]}"
                )
            if args.writer_unseen_policy == "error":
                for split_name, split in [("val", val_split), ("test", test_split)]:
                    if split is None:
                        continue
                    unseen = sorted({str(value) for value in split.writer_ids} - expected_train_writer_ids)
                    if unseen:
                        raise ValueError(
                            f"Unseen writer IDs found in split={split_name} while writer_unseen_policy=error: {unseen[:10]}"
                        )
            args.num_writer_embeddings = writer_vocab.num_embeddings
            dataset_writer_to_index = writer_vocab.writer_to_index
            dataset_writer_unknown_index = writer_vocab.unknown_writer_index
            dataset_writer_unseen_policy = args.writer_unseen_policy
        else:
            args.num_writer_embeddings = 1
        if args.use_style_cluster_conditioning:
            assert style_cluster_map_path is not None and panel_clusters_path is not None
            style_cluster_map = load_style_cluster_map(style_cluster_map_path)
            panel_clusters = load_panel_clusters(panel_clusters_path, style_cluster_map)
            known_cluster_writers = set(style_cluster_map.writer_to_cluster_id)
            for split_name, split in [("train", train_split), ("val", val_split), ("test", test_split)]:
                if split is None:
                    continue
                missing = sorted({str(value) for value in split.writer_ids} - known_cluster_writers)
                if missing:
                    raise ValueError(
                        f"Style cluster map does not cover split={split_name}; missing writers={missing[:10]}"
                    )
            args.num_style_clusters = style_cluster_map.num_clusters
            dataset_style_cluster_by_writer = style_cluster_map.writer_to_cluster_id
        else:
            args.num_style_clusters = 1
        args.panel_writers = panel_writers
        args.writer_map_path = str(writer_map_path) if writer_map_path is not None else None
        args.panel_writers_path = str(panel_writers_path) if panel_writers_path is not None else None
        args.panel_clusters = panel_clusters
        args.style_cluster_map_path = str(style_cluster_map_path) if style_cluster_map_path is not None else None
        args.panel_clusters_path = str(panel_clusters_path) if panel_clusters_path is not None else None

        tokenizer_spec = resolve_tokenizer_spec(args.tokenizer_variant, tokenizer_config_path)
        radius_codebooks = collect_radius_codebooks_from_offsets(
            train_split.offsets,
            spec=tokenizer_spec,
            max_points_for_bins=min(args.max_points_for_bins, tokenizer_spec.max_points_for_bins),
            seed=args.seed,
        )
        text_vocab = TextVocab(train_split.texts)
        stroke_vocab = StrokeVocab(tokenizer_spec)
        stroke_tokenizer = PolarOffsetTokenizer(tokenizer_spec, radius_codebooks, stroke_vocab)
        split_map: Dict[str, ProcessedWordSplit] = {"train": train_split, "val": val_split}
        if test_split is not None:
            split_map["test"] = test_split

        tokenizer_artifacts_dir = out_dir / "tokenizer"
        if is_main_process(rank):
            print(f"Train samples: {len(train_split.texts)} | Val samples: {len(val_split.texts)}")
            if test_split is not None:
                print(f"Test samples: {len(test_split.texts)}")
            print(f"Tokenizer variant: {tokenizer_spec.name} ({tokenizer_spec.scheme})")
            print(
                f"Radius codebooks: {tokenizer_spec.radius_codebook_mode} | "
                f"decode representative: {tokenizer_spec.radius_decode_mode}"
            )
            print(f"Checkpoint out dir: {out_dir.as_posix()}")
            print(f"Sample out dir: {sample_dir.as_posix()}")
            print(f"Text vocab size: {len(text_vocab)}")
            print(f"Stroke vocab size: {stroke_vocab.vocab_size}")
            if args.use_writer_conditioning:
                print(
                    f"Writer conditioning: enabled={args.use_writer_conditioning} "
                    f"mode={args.writer_conditioning_mode} embed_dim={args.writer_embed_dim} "
                    f"num_writer_embeddings={args.num_writer_embeddings}"
                )
                assert writer_map_path is not None and panel_writers_path is not None
                print(f"Writer map path: {writer_map_path.as_posix()}")
                print(f"Panel writers path: {panel_writers_path.as_posix()}")
            if args.use_style_cluster_conditioning:
                print(
                    f"Style-cluster conditioning: enabled={args.use_style_cluster_conditioning} "
                    f"mode={args.cluster_conditioning_mode} embed_dim={args.style_cluster_embed_dim} "
                    f"num_style_clusters={args.num_style_clusters}"
                )
                assert style_cluster_map_path is not None and panel_clusters_path is not None
                print(f"Style-cluster map path: {style_cluster_map_path.as_posix()}")
                print(f"Panel clusters path: {panel_clusters_path.as_posix()}")
            if tokenizer_config_path is not None:
                print(f"Tokenizer config path: {tokenizer_config_path.as_posix()}")
            print(f"Evaluation word panel: {', '.join(eval_words)}")
            print(f"Decoding modes: {', '.join(mode.name for mode in decoding_modes)}")
            if panel_writers:
                print(
                    "Panel writers: "
                    + ", ".join(f"{item['writer_id']}@{item['writer_index']}" for item in panel_writers)
                )
            if panel_clusters:
                print(
                    "Panel clusters: "
                    + ", ".join(f"{item['label']}@{item['cluster_id']}" for item in panel_clusters)
                )
            if resume_path is not None:
                print(f"Resume checkpoint: {resume_path.as_posix()}")

        tokenizer_diagnostics = None
        if not args.skip_tokenizer_diagnostics and is_main_process(rank):
            diagnostics_dir = out_dir / "tokenizer_diagnostics"
            tokenizer_diagnostics = run_tokenizer_diagnostics(
                split_map=split_map,
                tokenizer_spec=tokenizer_spec,
                stroke_tokenizer=stroke_tokenizer,
                out_dir=diagnostics_dir,
                per_source_split=args.tokenizer_diagnostics_per_source_split,
            )
            print(
                f"[tokenizer diagnostics] mean_point_mse={tokenizer_diagnostics['mean_point_mse']:.6f} "
                f"mean_draw_accuracy={tokenizer_diagnostics['mean_draw_accuracy']:.6f} "
                f"draw_mse={tokenizer_diagnostics['aggregate_draw_flag_metrics']['draw_1']['mean_offset_mse']:.6f} "
                f"relocate_mse={tokenizer_diagnostics['aggregate_draw_flag_metrics']['draw_0']['mean_offset_mse']:.6f}"
            )
        if ddp and not args.skip_tokenizer_diagnostics:
            torch.distributed.barrier()

        train_ds = ProcessedWordDataset(
            split=train_split,
            split_name="train",
            text_vocab=text_vocab,
            stroke_tokenizer=stroke_tokenizer,
            max_text_len=args.max_text_len,
            max_stroke_tokens=args.max_stroke_tokens,
            writer_to_index=dataset_writer_to_index,
            writer_unknown_index=dataset_writer_unknown_index,
            style_cluster_by_writer=dataset_style_cluster_by_writer,
            default_style_cluster_id=dataset_default_style_cluster_id,
            writer_unseen_policy=dataset_writer_unseen_policy,
            train_downsample_keep_min=args.train_downsample_keep_min,
            train_downsample_keep_max=args.train_downsample_keep_max,
        )
        val_ds = ProcessedWordDataset(
            split=val_split,
            split_name="val",
            text_vocab=text_vocab,
            stroke_tokenizer=stroke_tokenizer,
            max_text_len=args.max_text_len,
            max_stroke_tokens=args.max_stroke_tokens,
            writer_to_index=dataset_writer_to_index,
            writer_unknown_index=dataset_writer_unknown_index,
            style_cluster_by_writer=dataset_style_cluster_by_writer,
            default_style_cluster_id=dataset_default_style_cluster_id,
            writer_unseen_policy=dataset_writer_unseen_policy,
        )
        test_ds = (
            ProcessedWordDataset(
                split=test_split,
                split_name="test",
                text_vocab=text_vocab,
                stroke_tokenizer=stroke_tokenizer,
                max_text_len=args.max_text_len,
                max_stroke_tokens=args.max_stroke_tokens,
                writer_to_index=dataset_writer_to_index,
                writer_unknown_index=dataset_writer_unknown_index,
                style_cluster_by_writer=dataset_style_cluster_by_writer,
                default_style_cluster_id=dataset_default_style_cluster_id,
                writer_unseen_policy=dataset_writer_unseen_policy,
            )
            if test_split is not None
            else None
        )

        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed) if ddp else None
        val_sampler = DistributedSampler(val_ds, shuffle=False, seed=args.seed) if ddp else None
        test_sampler = DistributedSampler(test_ds, shuffle=False, seed=args.seed) if ddp and test_ds is not None else None

        collate_fn = partial(
            collate_batch,
            text_pad_id=text_vocab.pad_id,
            stroke_pad_id=stroke_vocab.pad_id,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(args.seed + rank * 10_000)
        worker_init_fn = build_worker_init_fn(args.seed, rank)
        pin_memory = bool(device.type == "cuda" and args.num_workers > 0)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=worker_init_fn,
            generator=loader_generator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
            persistent_workers=(args.num_workers > 0),
            worker_init_fn=worker_init_fn,
        )
        test_loader = (
            DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=test_sampler,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
                persistent_workers=(args.num_workers > 0),
                worker_init_fn=worker_init_fn,
            )
            if test_ds is not None
            else None
        )

        raw_model = build_model(args, text_vocab, stroke_vocab).to(device)
        checkpoint_payload = None
        if resume_path is not None:
            checkpoint_payload = load_checkpoint_payload(resume_path)
            restore_model_weights(raw_model, checkpoint_payload)

        runtime_seed = args.seed + rank
        seed_everything(runtime_seed)

        model: nn.Module = raw_model
        if ddp:
            print(f"[rank {rank}] entering DDP init", flush=True)
            ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {"device_ids": None}
            model = DDP(raw_model, **ddp_kwargs)
            print(f"[rank {rank}] finished DDP init", flush=True)
        elif (
            device.type == "cuda"
            and str(getattr(args, "multi_gpu_mode", "single")) == "dataparallel"
            and torch.cuda.device_count() > 1
        ):
            model = nn.DataParallel(raw_model)
            if is_main_process(rank):
                print(f"[dataparallel] enabled across {torch.cuda.device_count()} GPUs")

        optimizer = build_optimizer(model, args)
        steps_per_epoch = len(train_loader)
        total_steps = max(1, steps_per_epoch * args.epochs)
        scheduler = build_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
            scheduler_type=args.scheduler_type,
            step_lr_every=args.step_lr_every,
            lr_decay=args.lr_decay,
        )
        scaler = build_scaler(args, device)
        state = TrainState()
        eval_history: List[dict] = []
        panel_history: List[dict] = []
        eos_history: List[dict] = []
        if checkpoint_payload is not None and resume_path is not None:
            state, eval_history, panel_history, eos_history = restore_train_state(checkpoint_payload, resume_path)
            restore_optimizer_scheduler_scaler(
                checkpoint_payload=checkpoint_payload,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                reset_optimizer=args.resume_reset_optimizer,
                reset_scheduler=args.resume_reset_scheduler,
                reset_scaler=args.resume_reset_scaler,
            )
            if is_main_process(rank):
                print(
                    f"[resume] restored step={state.step} completed_epochs={state.completed_epochs} "
                    f"best_val_epoch={state.best_val_epoch} best_val_loss={state.best_val_loss}"
                )
        test_metrics: Optional[dict] = None

        if is_main_process(rank):
            tokenizer_artifacts_dir.mkdir(parents=True, exist_ok=True)
            (tokenizer_artifacts_dir / "tokenizer_spec.json").write_text(
                json.dumps(tokenizer_spec_to_dict(tokenizer_spec), indent=2)
            )
            (tokenizer_artifacts_dir / "radius_codebooks.json").write_text(
                json.dumps(radius_codebooks.to_dict(), indent=2)
            )
            np.save(tokenizer_artifacts_dir / "radius_edges.npy", radius_codebooks.legacy_radius_edges())
            if args.use_writer_conditioning:
                assert writer_vocab is not None and writer_map_path is not None and panel_writers_path is not None
                write_writer_conditioning_artifacts(
                    out_dir=out_dir,
                    sample_dir=sample_dir,
                    args=args,
                    writer_vocab=writer_vocab,
                    writer_map_path=writer_map_path,
                    panel_writers_path=panel_writers_path,
                    panel_writers=panel_writers,
                )
            if args.use_style_cluster_conditioning:
                assert style_cluster_map is not None and style_cluster_map_path is not None and panel_clusters_path is not None
                write_style_cluster_conditioning_artifacts(
                    out_dir=out_dir,
                    sample_dir=sample_dir,
                    args=args,
                    style_cluster_map=style_cluster_map,
                    style_cluster_map_path=style_cluster_map_path,
                    panel_clusters_path=panel_clusters_path,
                    panel_clusters=panel_clusters,
                )
            raw_model_for_stats = model.module if hasattr(model, "module") else model
            num_params = sum(param.numel() for param in raw_model_for_stats.parameters())
            print(f"Model parameters: {num_params:,}")
            print(
                f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps} | "
                f"Warmup steps: {min(args.warmup_steps, max(total_steps - 1, 0))}"
            )
            print(
                f"Scheduler: {args.scheduler_type} | "
                f"step_lr_every={args.step_lr_every} | lr_decay={args.lr_decay} | "
                f"min_lr_ratio={args.min_lr_ratio}"
            )
            print(
                f"AdamW betas=({args.adam_beta1:.3f}, {args.adam_beta2:.3f}) "
                f"eps={args.adam_eps:.1e} | train_label_smoothing={args.train_label_smoothing:.3f}"
            )
            print(
                f"Train downsample keep range: "
                f"{args.train_downsample_keep_min:.3f}-{args.train_downsample_keep_max:.3f}"
            )
            print(f"Effective global batch size: {args.batch_size * world_size}")

        amp_device = "cuda" if device.type == "cuda" else "cpu"
        running_loss = 0.0
        running_tokens = 0
        start_time = time.time()
        start_epoch = int(state.completed_epochs)
        first_train_step_logged = False
        first_batch_fetch_logged = False
        first_forward_logged = False
        first_backward_logged = False

        if start_epoch >= args.epochs and is_main_process(rank):
            print(f"[resume] checkpoint already reached completed_epochs={start_epoch}, target epochs={args.epochs}")

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            iter_train = train_loader
            if is_main_process(rank):
                iter_train = maybe_tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=True)

            for batch in iter_train:
                if not first_batch_fetch_logged:
                    print(f"[rank {rank}] fetched first batch", flush=True)
                    first_batch_fetch_logged = True
                state.step += 1
                text_ids = batch["text_ids"].to(device, non_blocking=True)
                stroke_in_ids = batch["stroke_in_ids"].to(device, non_blocking=True)
                stroke_tgt_ids = batch["stroke_tgt_ids"].to(device, non_blocking=True)
                writer_ids = batch["writer_ids"].to(device, non_blocking=True)
                style_cluster_ids = batch["style_cluster_ids"].to(device, non_blocking=True)
                text_pad_mask = batch["text_pad_mask"].to(device, non_blocking=True)
                stroke_pad_mask = batch["stroke_pad_mask"].to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=amp_device, enabled=args.amp and device.type == "cuda"):
                    logits = model(
                        text_ids,
                        stroke_in_ids,
                        text_pad_mask,
                        stroke_pad_mask,
                        writer_ids=writer_ids,
                        style_cluster_ids=style_cluster_ids,
                    )
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        stroke_tgt_ids.reshape(-1),
                        ignore_index=stroke_vocab.pad_id,
                        label_smoothing=float(args.train_label_smoothing),
                        reduction="sum",
                    )
                    valid_tokens = stroke_tgt_ids.ne(stroke_vocab.pad_id).sum()
                    loss = loss / valid_tokens.clamp_min(1)
                if not first_forward_logged:
                    print(f"[rank {rank}] finished first forward", flush=True)
                    first_forward_logged = True

                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if not first_backward_logged:
                        print(f"[rank {rank}] finished first backward", flush=True)
                        first_backward_logged = True
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if not first_backward_logged:
                        print(f"[rank {rank}] finished first backward", flush=True)
                        first_backward_logged = True
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

                scheduler.step()

                batch_valid_tokens = int(valid_tokens.item())
                running_loss += float(loss.item()) * batch_valid_tokens
                running_tokens += batch_valid_tokens

                if is_main_process(rank) and not first_train_step_logged:
                    print(
                        f"[train] first optimization step complete "
                        f"step={state.step} batch_tokens={batch_valid_tokens} "
                        f"loss={float(loss.item()):.6f}"
                    )
                    first_train_step_logged = True

                if is_main_process(rank) and hasattr(iter_train, "set_postfix"):
                    iter_train.set_postfix(loss=f"{float(loss.item()):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

                if is_main_process(rank) and args.log_every > 0 and state.step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    avg_loss = running_loss / max(running_tokens, 1)
                    print(
                        f"epoch={epoch + 1}/{args.epochs} step={state.step} "
                        f"train_nll={avg_loss:.6f} ppl={math.exp(min(avg_loss, 20)):.3f} "
                        f"lr={optimizer.param_groups[0]['lr']:.3e} elapsed_s={elapsed:.1f}"
                    )
                    running_loss = 0.0
                    running_tokens = 0
                    start_time = time.time()

                if args.eval_every > 0 and state.step % args.eval_every == 0:
                    record_validation_result(
                        result=run_eval_if_needed(
                            ddp=ddp,
                            rank=rank,
                            device=device,
                            model=model,
                            loader=val_loader,
                            split_name="eval",
                            pad_id=stroke_vocab.pad_id,
                            use_amp=args.amp and device.type == "cuda",
                        ),
                        epoch=epoch + 1,
                        trigger="step",
                        state=state,
                        rank=rank,
                        out_dir=out_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        args=args,
                        text_vocab=text_vocab,
                        stroke_vocab=stroke_vocab,
                        tokenizer_spec=tokenizer_spec,
                        radius_codebooks=radius_codebooks,
                        eval_history=eval_history,
                        panel_history=panel_history,
                        eos_history=eos_history,
                    )

                if is_main_process(rank) and args.save_every > 0 and state.step % args.save_every == 0:
                    save_checkpoint(
                        out_dir,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        state,
                        args,
                        text_vocab,
                        stroke_vocab,
                        tokenizer_spec,
                        radius_codebooks,
                        eval_history,
                        panel_history,
                        eos_history,
                        filename=f"step_{state.step}.pt",
                    )
                    print(f"[checkpoint] saved step_{state.step}.pt")

            state.completed_epochs = epoch + 1
            current_epoch_checkpoint_path = out_dir / f"epoch_{epoch + 1}.pt"
            epoch_val_result: Optional[Tuple[float, int]] = None

            if args.eval_every_epochs > 0 and (epoch + 1) % args.eval_every_epochs == 0:
                epoch_val_result = run_eval_if_needed(
                    ddp=ddp,
                    rank=rank,
                    device=device,
                    model=model,
                    loader=val_loader,
                    split_name="val",
                    pad_id=stroke_vocab.pad_id,
                    use_amp=args.amp and device.type == "cuda",
                )
                record_validation_result(
                    result=epoch_val_result,
                    epoch=epoch + 1,
                    trigger="epoch",
                    state=state,
                    rank=rank,
                    out_dir=out_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                    text_vocab=text_vocab,
                    stroke_vocab=stroke_vocab,
                    tokenizer_spec=tokenizer_spec,
                    radius_codebooks=radius_codebooks,
                    eval_history=eval_history,
                    panel_history=panel_history,
                    eos_history=eos_history,
                )

            if args.panel_every_epochs > 0 and (epoch + 1) % args.panel_every_epochs == 0:
                if epoch_val_result is None:
                    epoch_val_result = run_eval_if_needed(
                        ddp=ddp,
                        rank=rank,
                        device=device,
                        model=model,
                        loader=val_loader,
                        split_name="val",
                        pad_id=stroke_vocab.pad_id,
                        use_amp=args.amp and device.type == "cuda",
                    )
                eos_result = compute_eos_diagnostics(
                    ddp=ddp,
                    rank=rank,
                    device=device,
                    model=model,
                    loader=val_loader,
                    pad_id=stroke_vocab.pad_id,
                    eos_id=stroke_vocab.eos_id,
                    use_amp=args.amp and device.type == "cuda",
                    epoch=epoch + 1,
                    global_step=state.step,
                    checkpoint_path=str(current_epoch_checkpoint_path.resolve()),
                    split_name="val",
                )
                if eos_result is not None and is_main_process(rank):
                    eos_history.append(eos_result)
                    eos_summary = write_eos_diagnostics_artifacts(out_dir=out_dir, sample_dir=sample_dir, eos_history=eos_history)
                    print(
                        f"[eos] wrote per-epoch diagnostics to {out_dir.as_posix()} "
                        f"(trend={eos_summary.get('eos_trend')})"
                    )
                evaluate_panel_if_main(
                    ddp=ddp,
                    rank=rank,
                    device=device,
                    model=model,
                    args=args,
                    state=state,
                    sample_dir=sample_dir,
                    current_epoch=epoch + 1,
                    current_checkpoint_path=current_epoch_checkpoint_path,
                    text_vocab=text_vocab,
                    stroke_vocab=stroke_vocab,
                    stroke_tokenizer=stroke_tokenizer,
                    decoding_modes=decoding_modes,
                    eval_words=eval_words,
                    panel_writers=panel_writers,
                    panel_clusters=panel_clusters,
                    panel_history=panel_history,
                )

            if is_main_process(rank):
                save_checkpoint(
                    out_dir,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    state,
                    args,
                    text_vocab,
                    stroke_vocab,
                    tokenizer_spec,
                    radius_codebooks,
                    eval_history,
                    panel_history,
                    eos_history,
                    filename=f"epoch_{epoch + 1}.pt",
                )
                print(f"[checkpoint] saved epoch_{epoch + 1}.pt")

                write_run_summary(
                    out_dir=out_dir,
                    sample_dir=sample_dir,
                    args=args,
                    tokenizer_spec=tokenizer_spec,
                    steps_per_epoch=steps_per_epoch,
                    state=state,
                    eval_history=eval_history,
                    panel_history=panel_history,
                    eos_history=eos_history,
                    test_metrics=test_metrics,
                    tokenizer_diagnostics=tokenizer_diagnostics,
                )

        if is_main_process(rank):
            print("Training finished.")

        if test_loader is not None:
            test_result = run_eval_if_needed(
                ddp=ddp,
                rank=rank,
                device=device,
                model=model,
                loader=test_loader,
                split_name="test",
                pad_id=stroke_vocab.pad_id,
                use_amp=args.amp and device.type == "cuda",
            )
            if test_result is not None and is_main_process(rank):
                test_loss_sum, test_token_count = test_result
                test_nll = test_loss_sum / max(test_token_count, 1)
                test_metrics = {
                    "nll": test_nll,
                    "ppl": math.exp(min(test_nll, 20)),
                    "token_count": test_token_count,
                }

        if is_main_process(rank):
            if eos_history:
                write_eos_diagnostics_artifacts(out_dir=out_dir, sample_dir=sample_dir, eos_history=eos_history)
            write_run_summary(
                out_dir=out_dir,
                sample_dir=sample_dir,
                args=args,
                tokenizer_spec=tokenizer_spec,
                steps_per_epoch=steps_per_epoch,
                state=state,
                eval_history=eval_history,
                panel_history=panel_history,
                eos_history=eos_history,
                test_metrics=test_metrics,
                tokenizer_diagnostics=tokenizer_diagnostics,
            )

    finally:
        cleanup_ddp()
