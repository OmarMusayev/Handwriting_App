from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .checkpoint import TrainState, load_checkpoint_payload, restore_model_weights
from .data import ProcessedWordSplit, TextVocab
from .generation import (
    DecodingMode,
    build_sample_seed,
    build_sample_stem,
    compute_generated_sample_stats,
    generate_tokens,
    offsets_to_absolute_points,
    plot_overlay_points,
    plot_points,
    save_panel_grid,
    save_panel_sample_artifact,
    summarize_panel_sample_group,
)
from .model import build_model
from .reporting import write_panel_epoch_summary
from .tokenizers import PolarOffsetTokenizer, StrokeVocab, TokenizerSpec, tokenizer_spec_to_dict
from .utils import is_main_process, maybe_import_matplotlib


def _wait_for_file(path: Path, *, description: str, timeout_s: float = 3600.0, poll_interval_s: float = 2.0) -> None:
    start = time.monotonic()
    while not path.exists():
        elapsed = time.monotonic() - start
        if elapsed > timeout_s:
            raise TimeoutError(f"Timed out waiting for {description}: {path.as_posix()}")
        time.sleep(poll_interval_s)


@torch.no_grad()
def evaluate(
    model,
    loader: DataLoader,
    device: torch.device,
    pad_id: int,
    use_amp: bool,
) -> Tuple[float, int]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    for batch in loader:
        text_ids = batch["text_ids"].to(device, non_blocking=True)
        stroke_in_ids = batch["stroke_in_ids"].to(device, non_blocking=True)
        stroke_tgt_ids = batch["stroke_tgt_ids"].to(device, non_blocking=True)
        writer_ids = batch["writer_ids"].to(device, non_blocking=True)
        style_cluster_ids = batch["style_cluster_ids"].to(device, non_blocking=True)
        text_pad_mask = batch["text_pad_mask"].to(device, non_blocking=True)
        stroke_pad_mask = batch["stroke_pad_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=amp_device, enabled=use_amp):
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
                ignore_index=pad_id,
                reduction="sum",
            )

        valid = int(stroke_tgt_ids.ne(pad_id).sum().item())
        total_loss += float(loss.item())
        total_tokens += valid

    model.train()
    return total_loss, total_tokens


def _collect_local_eos_records(
    *,
    logits: torch.Tensor,
    stroke_tgt_ids: torch.Tensor,
    pad_id: int,
    eos_id: int,
    end_window: int,
) -> List[dict]:
    probs = torch.softmax(logits.float(), dim=-1)
    valid_lengths = stroke_tgt_ids.ne(pad_id).sum(dim=1)
    records: List[dict] = []

    for batch_index in range(stroke_tgt_ids.shape[0]):
        seq_len = int(valid_lengths[batch_index].item())
        if seq_len <= 0:
            continue
        final_index = seq_len - 1
        final_logits = logits[batch_index, final_index].float()
        final_probs = probs[batch_index, final_index]
        eos_prob = float(final_probs[eos_id].item())
        eos_rank = int((final_logits > final_logits[eos_id]).sum().item()) + 1
        best_token = int(torch.argmax(final_logits).item())

        non_eos_probs = final_probs.clone()
        non_eos_probs[eos_id] = 0.0
        best_non_eos_prob = float(non_eos_probs.max().item())

        window_start = max(0, seq_len - int(end_window))
        window_probs = probs[batch_index, window_start:seq_len, eos_id]
        window_mean = float(window_probs.mean().item())
        window_max = float(window_probs.max().item())
        window_min = float(window_probs.min().item())

        records.append(
            {
                "eos_prob_at_true_end": eos_prob,
                "eos_rank_at_true_end": eos_rank,
                "eos_top1_at_true_end": bool(best_token == eos_id),
                "eos_in_top3_at_true_end": bool(eos_rank <= 3),
                "eos_in_top5_at_true_end": bool(eos_rank <= 5),
                "end_window_mean_eos_prob": window_mean,
                "end_window_max_eos_prob": window_max,
                "end_window_min_eos_prob": window_min,
                "end_window_all_below_0p1": bool(window_max < 0.1),
                "eos_margin_at_true_end": float(eos_prob - best_non_eos_prob),
                "best_non_eos_prob_at_true_end": best_non_eos_prob,
                "eos_prob_ge_0p5": bool(eos_prob >= 0.5),
                "eos_prob_ge_0p2": bool(eos_prob >= 0.2),
                "eos_prob_lt_0p1": bool(eos_prob < 0.1),
            }
        )
    return records


def _merge_eos_records(ddp: bool, local_records: List[dict]) -> List[dict]:
    if not ddp:
        return local_records
    gathered: List[Optional[List[dict]]] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, local_records)
    merged: List[dict] = []
    for shard in gathered:
        if shard:
            merged.extend(shard)
    return merged


def _summarize_eos_records(
    *,
    records: Sequence[dict],
    epoch: int,
    global_step: int,
    checkpoint_path: Optional[str],
    split_name: str,
    val_result: Optional[Tuple[float, int]],
    end_window: int,
) -> dict:
    if not records:
        return {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "checkpoint_path": checkpoint_path,
            "split": split_name,
            "sample_count": 0,
            "val_nll": None,
            "val_ppl": None,
            "end_window_size": int(end_window),
            "eos_top1_rate_at_true_end": None,
            "mean_eos_prob_at_true_end": None,
            "median_eos_prob_at_true_end": None,
            "min_eos_prob_at_true_end": None,
            "max_eos_prob_at_true_end": None,
            "mean_eos_rank_at_true_end": None,
            "median_eos_rank_at_true_end": None,
            "mean_end_window_eos_prob": None,
            "max_end_window_eos_prob": None,
            "min_end_window_eos_prob": None,
            "mean_eos_margin_at_true_end": None,
            "mean_best_non_eos_prob_at_true_end": None,
            "eos_in_top3_rate_at_true_end": None,
            "eos_in_top5_rate_at_true_end": None,
            "eos_prob_ge_0p5_rate_at_true_end": None,
            "eos_prob_ge_0p2_rate_at_true_end": None,
            "eos_prob_lt_0p1_rate_at_true_end": None,
            "end_window_all_below_0p1_rate": None,
        }

    eos_probs = np.asarray([row["eos_prob_at_true_end"] for row in records], dtype=np.float32)
    eos_ranks = np.asarray([row["eos_rank_at_true_end"] for row in records], dtype=np.float32)
    window_means = np.asarray([row["end_window_mean_eos_prob"] for row in records], dtype=np.float32)
    window_maxes = np.asarray([row["end_window_max_eos_prob"] for row in records], dtype=np.float32)
    window_mins = np.asarray([row["end_window_min_eos_prob"] for row in records], dtype=np.float32)
    eos_margins = np.asarray([row["eos_margin_at_true_end"] for row in records], dtype=np.float32)
    best_non_eos_probs = np.asarray([row["best_non_eos_prob_at_true_end"] for row in records], dtype=np.float32)
    val_nll = None
    val_ppl = None
    if val_result is not None:
        val_loss_sum, val_token_count = val_result
        val_nll = float(val_loss_sum / max(val_token_count, 1))
        val_ppl = float(math.exp(min(val_nll, 20.0)))

    return {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "checkpoint_path": checkpoint_path,
        "split": split_name,
        "sample_count": int(len(records)),
        "val_nll": val_nll,
        "val_ppl": val_ppl,
        "end_window_size": int(end_window),
        "eos_top1_rate_at_true_end": float(np.mean([row["eos_top1_at_true_end"] for row in records])),
        "mean_eos_prob_at_true_end": float(np.mean(eos_probs)),
        "median_eos_prob_at_true_end": float(np.median(eos_probs)),
        "min_eos_prob_at_true_end": float(np.min(eos_probs)),
        "max_eos_prob_at_true_end": float(np.max(eos_probs)),
        "mean_eos_rank_at_true_end": float(np.mean(eos_ranks)),
        "median_eos_rank_at_true_end": float(np.median(eos_ranks)),
        "mean_end_window_eos_prob": float(np.mean(window_means)),
        "max_end_window_eos_prob": float(np.max(window_maxes)),
        "min_end_window_eos_prob": float(np.min(window_mins)),
        "mean_eos_margin_at_true_end": float(np.mean(eos_margins)),
        "mean_best_non_eos_prob_at_true_end": float(np.mean(best_non_eos_probs)),
        "eos_in_top3_rate_at_true_end": float(np.mean([row["eos_in_top3_at_true_end"] for row in records])),
        "eos_in_top5_rate_at_true_end": float(np.mean([row["eos_in_top5_at_true_end"] for row in records])),
        "eos_prob_ge_0p5_rate_at_true_end": float(np.mean([row["eos_prob_ge_0p5"] for row in records])),
        "eos_prob_ge_0p2_rate_at_true_end": float(np.mean([row["eos_prob_ge_0p2"] for row in records])),
        "eos_prob_lt_0p1_rate_at_true_end": float(np.mean([row["eos_prob_lt_0p1"] for row in records])),
        "end_window_all_below_0p1_rate": float(np.mean([row["end_window_all_below_0p1"] for row in records])),
    }


@torch.no_grad()
def compute_eos_diagnostics(
    *,
    ddp: bool,
    rank: int,
    device: torch.device,
    model,
    loader: Optional[DataLoader],
    pad_id: int,
    eos_id: int,
    use_amp: bool,
    epoch: int,
    global_step: int,
    checkpoint_path: Optional[str],
    split_name: str = "val",
    end_window: int = 4,
) -> Optional[dict]:
    if loader is None:
        return None
    if ddp:
        dist.barrier()

    model.eval()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    local_records: List[dict] = []
    loss_sum = 0.0
    token_count = 0

    for batch in loader:
        text_ids = batch["text_ids"].to(device, non_blocking=True)
        stroke_in_ids = batch["stroke_in_ids"].to(device, non_blocking=True)
        stroke_tgt_ids = batch["stroke_tgt_ids"].to(device, non_blocking=True)
        writer_ids = batch["writer_ids"].to(device, non_blocking=True)
        style_cluster_ids = batch["style_cluster_ids"].to(device, non_blocking=True)
        text_pad_mask = batch["text_pad_mask"].to(device, non_blocking=True)
        stroke_pad_mask = batch["stroke_pad_mask"].to(device, non_blocking=True)

        with torch.autocast(device_type=amp_device, enabled=use_amp):
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
                ignore_index=pad_id,
                reduction="sum",
            )
        valid_tokens = int(stroke_tgt_ids.ne(pad_id).sum().item())
        loss_sum += float(loss.item())
        token_count += valid_tokens
        local_records.extend(
            _collect_local_eos_records(
                logits=logits,
                stroke_tgt_ids=stroke_tgt_ids,
                pad_id=pad_id,
                eos_id=eos_id,
                end_window=end_window,
            )
        )

    model.train()
    if ddp:
        loss_sum, token_count = maybe_reduce_eval_stats(ddp, device, loss_sum, token_count)
    records = _merge_eos_records(ddp, local_records)
    if ddp:
        dist.barrier()

    if not is_main_process(rank):
        return None

    summary = _summarize_eos_records(
        records=records,
        epoch=epoch,
        global_step=global_step,
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        val_result=(loss_sum, token_count),
        end_window=end_window,
    )
    print(
        f"[eos] epoch={epoch} step={global_step} sample_count={summary['sample_count']} "
        f"top1={summary['eos_top1_rate_at_true_end']:.4f} "
        f"mean_prob={summary['mean_eos_prob_at_true_end']:.4f} "
        f"mean_rank={summary['mean_eos_rank_at_true_end']:.2f}"
    )
    return summary


def maybe_reduce_eval_stats(ddp: bool, device: torch.device, loss_sum: float, token_count: int) -> Tuple[float, int]:
    if not ddp:
        return loss_sum, token_count

    stats = torch.tensor([loss_sum, float(token_count)], dtype=torch.float64, device=device)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    return float(stats[0].item()), int(stats[1].item())


def run_eval_if_needed(
    *,
    ddp: bool,
    rank: int,
    device: torch.device,
    model,
    loader: Optional[DataLoader],
    split_name: str,
    pad_id: int,
    use_amp: bool,
) -> Optional[Tuple[float, int]]:
    if loader is None:
        return None
    if ddp:
        dist.barrier()
    loss_sum, token_count = evaluate(model, loader, device, pad_id, use_amp)
    loss_sum, token_count = maybe_reduce_eval_stats(ddp, device, loss_sum, token_count)
    if ddp:
        dist.barrier()
    if is_main_process(rank):
        avg_loss = loss_sum / max(token_count, 1)
        print(f"[{split_name}] nll={avg_loss:.6f} ppl={math.exp(min(avg_loss, 20)):.3f}")
    return loss_sum, token_count


def run_panel_checkpoint_generation(
    *,
    model,
    panel_epoch: int,
    checkpoint_label: str,
    checkpoint_epoch: Optional[int],
    checkpoint_step: Optional[int],
    checkpoint_path: Optional[str],
    sample_dir: Path,
    words: Sequence[str],
    panel_conditions: Sequence[dict],
    conditioning_kind: str,
    decoding_modes: Sequence[DecodingMode],
    text_vocab: TextVocab,
    stroke_vocab: StrokeVocab,
    stroke_tokenizer: PolarOffsetTokenizer,
    args,
    device: torch.device,
) -> dict:
    epoch_dir = sample_dir / f"epoch_{panel_epoch:02d}" / checkpoint_label
    resolved_panel_conditions = list(panel_conditions) if panel_conditions else [
        {"label": "default", "condition_id": None}
    ]
    samples_by_key: Dict[Tuple[str, str, str], dict] = {}
    mode_entries: List[dict] = []
    mode_samples_by_name: Dict[str, List[dict]] = {mode.name: [] for mode in decoding_modes}
    condition_entries: List[dict] = []

    for condition_entry in resolved_panel_conditions:
        condition_label = str(
            condition_entry.get("label")
            or condition_entry.get("writer_id")
            or condition_entry.get("cluster_id")
            or "default"
        )
        writer_index = condition_entry.get("writer_index")
        style_cluster_id = condition_entry.get("cluster_id")
        condition_dir = epoch_dir if (writer_index is None and style_cluster_id is None) else (epoch_dir / condition_label)
        condition_mode_entries: List[dict] = []

        for mode in decoding_modes:
            mode_dir = condition_dir / mode.name
            mode_samples: List[dict] = []
            for word in words:
                sample_seed = build_sample_seed(
                    args,
                    panel_epoch,
                    checkpoint_label,
                    mode.name,
                    word,
                    condition_label if (writer_index is not None or style_cluster_id is not None) else None,
                )
                token_ids = generate_tokens(
                    model,
                    text=word,
                    text_vocab=text_vocab,
                    stroke_vocab=stroke_vocab,
                    max_text_len=args.max_text_len,
                    max_gen_tokens=args.sample_max_gen_tokens,
                    device=device,
                    temperature=mode.temperature,
                    top_k=mode.top_k,
                    greedy=mode.greedy,
                    sample_seed=sample_seed,
                    writer_id=int(writer_index) if writer_index is not None else None,
                    style_cluster_id=int(style_cluster_id) if style_cluster_id is not None else None,
                )
                offsets = stroke_tokenizer.decode_tokens_to_offsets(token_ids)
                points = offsets_to_absolute_points(offsets)
                stats = compute_generated_sample_stats(
                    token_ids=token_ids,
                    offsets=offsets,
                    points=points,
                    stroke_vocab=stroke_vocab,
                    max_gen_tokens=args.sample_max_gen_tokens,
                )
                file_stem = build_sample_stem(
                    panel_epoch,
                    checkpoint_label,
                    mode.name,
                    word,
                    condition_label if (writer_index is not None or style_cluster_id is not None) else None,
                )
                condition_metadata = {
                    "type": conditioning_kind,
                    "label": condition_label,
                    "writer_id": None if writer_index is None else str(condition_entry.get("writer_id")),
                    "writer_index": None if writer_index is None else int(writer_index),
                    "cluster_id": None if style_cluster_id is None else int(style_cluster_id),
                    "train_count": condition_entry.get("train_count"),
                    "train_writer_count": condition_entry.get("train_writer_count"),
                    "train_sample_count": condition_entry.get("train_sample_count"),
                    "representative_writer_ids": condition_entry.get("representative_writer_ids"),
                }
                metadata = {
                    "epoch": int(panel_epoch),
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_epoch": int(checkpoint_epoch) if checkpoint_epoch is not None else None,
                    "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else None,
                    "checkpoint_path": checkpoint_path,
                    "word": word,
                    "conditioning": condition_metadata,
                    "decoding_mode": {
                        "name": mode.name,
                        "temperature": mode.temperature,
                        "top_k": mode.top_k,
                        "greedy": mode.greedy,
                    },
                    "seed": int(sample_seed),
                    "experiment_name": args.experiment_name,
                    "stats": stats,
                }
                saved_paths = save_panel_sample_artifact(
                    output_dir=mode_dir,
                    file_stem=file_stem,
                    metadata=metadata,
                    token_ids=token_ids,
                    offsets=offsets,
                    points=points,
                )
                item = {
                    "word": word,
                    "conditioning": condition_metadata,
                    "file_stem": file_stem,
                    "seed": int(sample_seed),
                    "checkpoint_label": checkpoint_label,
                    "checkpoint_epoch": int(checkpoint_epoch) if checkpoint_epoch is not None else None,
                    "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else None,
                    "checkpoint_path": checkpoint_path,
                    "decoding_mode": metadata["decoding_mode"],
                    "stats": stats,
                    "json_path": saved_paths["json_path"],
                    "png_path": saved_paths["png_path"],
                    "points": points,
                }
                mode_samples.append(item)
                mode_samples_by_name[mode.name].append(item)
                samples_by_key[(condition_label, mode.name, word)] = item

            condition_mode_entries.append(
                {
                    "name": mode.name,
                    "temperature": mode.temperature,
                    "top_k": mode.top_k,
                    "greedy": mode.greedy,
                    "samples": [{key: value for key, value in sample.items() if key != "points"} for sample in mode_samples],
                    "summary": summarize_panel_sample_group(mode_samples),
                }
            )

        condition_entries.append(
            {
                "type": conditioning_kind,
                "writer_id": None if writer_index is None else str(condition_entry.get("writer_id")),
                "writer_index": None if writer_index is None else int(writer_index),
                "cluster_id": None if style_cluster_id is None else int(style_cluster_id),
                "label": condition_label,
                "train_count": condition_entry.get("train_count"),
                "train_writer_count": condition_entry.get("train_writer_count"),
                "train_sample_count": condition_entry.get("train_sample_count"),
                "representative_writer_ids": condition_entry.get("representative_writer_ids"),
                "decoding_modes": condition_mode_entries,
            }
        )

    for mode in decoding_modes:
        mode_samples = mode_samples_by_name[mode.name]
        mode_entries.append(
            {
                "name": mode.name,
                "temperature": mode.temperature,
                "top_k": mode.top_k,
                "greedy": mode.greedy,
                "samples": [{key: value for key, value in sample.items() if key != "points"} for sample in mode_samples],
                "summary": summarize_panel_sample_group(mode_samples),
            }
        )

    grid_path = save_panel_grid(
        path=epoch_dir / f"grid__{checkpoint_label}.png",
        epoch=panel_epoch,
        checkpoint_label=checkpoint_label,
        checkpoint_epoch=checkpoint_epoch,
        words=words,
        decoding_modes=decoding_modes,
        condition_entries=resolved_panel_conditions,
        samples_by_key=samples_by_key,
    )
    return {
        "label": checkpoint_label,
        "checkpoint_epoch": int(checkpoint_epoch) if checkpoint_epoch is not None else None,
        "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else None,
        "checkpoint_path": checkpoint_path,
        "grid_path": grid_path,
        "conditioning_kind": conditioning_kind,
        "panel_conditions": condition_entries,
        "decoding_modes": mode_entries,
    }


def evaluate_panel_if_main(
    *,
    ddp: bool,
    rank: int,
    device: torch.device,
    model,
    args,
    state: TrainState,
    sample_dir: Path,
    current_epoch: int,
    current_checkpoint_path: Path,
    text_vocab: TextVocab,
    stroke_vocab: StrokeVocab,
    stroke_tokenizer: PolarOffsetTokenizer,
    decoding_modes: Sequence[DecodingMode],
    eval_words: Sequence[str],
    panel_writers: Sequence[dict],
    panel_clusters: Sequence[dict],
    panel_history: List[dict],
) -> None:
    epoch_dir = sample_dir / f"epoch_{current_epoch:02d}"
    epoch_summary_path = epoch_dir / "summary.json"

    if is_main_process(rank):
        raw_model = model.module if hasattr(model, "module") else model
        conditioning_kind = "style_cluster" if bool(getattr(args, "use_style_cluster_conditioning", False)) else (
            "writer" if bool(getattr(args, "use_writer_conditioning", False)) else "default"
        )
        panel_conditions = list(panel_clusters) if conditioning_kind == "style_cluster" else list(panel_writers)
        summary = {
            "epoch": int(current_epoch),
            "seed": int(args.seed),
            "writer_conditioning": {
                "enabled": bool(getattr(args, "use_writer_conditioning", False)),
                "panel_writers": list(panel_writers),
            },
            "style_cluster_conditioning": {
                "enabled": bool(getattr(args, "use_style_cluster_conditioning", False)),
                "panel_clusters": list(panel_clusters),
            },
            "best_checkpoint": {
                "label": f"best_epoch_{state.best_val_epoch}" if state.best_val_epoch is not None else None,
                "epoch": int(state.best_val_epoch) if state.best_val_epoch is not None else None,
                "step": int(state.best_val_step) if state.best_val_step is not None else None,
                "path": state.best_checkpoint_path,
                "nll": float(state.best_val_loss) if math.isfinite(state.best_val_loss) else None,
                "ppl": float(state.best_val_ppl) if state.best_val_ppl is not None else None,
            },
            "evaluated_checkpoints": [],
        }

        current_result = run_panel_checkpoint_generation(
            model=raw_model,
            panel_epoch=current_epoch,
            checkpoint_label="current",
            checkpoint_epoch=current_epoch,
            checkpoint_step=state.step,
            checkpoint_path=str(current_checkpoint_path),
            sample_dir=sample_dir,
            words=eval_words,
            panel_conditions=panel_conditions,
            conditioning_kind=conditioning_kind,
            decoding_modes=decoding_modes,
            text_vocab=text_vocab,
            stroke_vocab=stroke_vocab,
            stroke_tokenizer=stroke_tokenizer,
            args=args,
            device=device,
        )
        summary["evaluated_checkpoints"].append(current_result)

        best_path = Path(state.best_checkpoint_path).expanduser() if state.best_checkpoint_path else None
        best_is_distinct = (
            best_path is not None
            and state.best_val_epoch is not None
            and state.best_val_epoch != current_epoch
            and best_path.exists()
        )
        if best_is_distinct:
            best_payload = load_checkpoint_payload(best_path)
            best_model = build_model(args, text_vocab, stroke_vocab).to(device)
            restore_model_weights(best_model, best_payload)
            best_model.eval()
            best_result = run_panel_checkpoint_generation(
                model=best_model,
                panel_epoch=current_epoch,
                checkpoint_label=f"best_epoch_{state.best_val_epoch}",
                checkpoint_epoch=state.best_val_epoch,
                checkpoint_step=state.best_val_step,
                checkpoint_path=str(best_path.resolve()),
                sample_dir=sample_dir,
                words=eval_words,
                panel_conditions=panel_conditions,
                conditioning_kind=conditioning_kind,
                decoding_modes=decoding_modes,
                text_vocab=text_vocab,
                stroke_vocab=stroke_vocab,
                stroke_tokenizer=stroke_tokenizer,
                args=args,
                device=device,
            )
            summary["evaluated_checkpoints"].append(best_result)
        elif state.best_val_epoch == current_epoch:
            summary["current_is_best_so_far"] = True

        write_panel_epoch_summary(epoch_dir, summary)
        panel_history.append(summary)
        print(f"[panel] saved evaluation artifacts for epoch {current_epoch} to {epoch_dir.as_posix()}")
    elif ddp:
        _wait_for_file(
            epoch_summary_path,
            description=f"panel summary for epoch {current_epoch}",
        )


def summarize_segment_subset(
    sq_errors: np.ndarray,
    l2_errors: np.ndarray,
    radii: np.ndarray,
) -> dict:
    count = int(sq_errors.shape[0])
    if count == 0:
        return {
            "num_segments": 0,
            "mean_offset_mse": None,
            "mean_l2_error": None,
            "mean_radius": None,
            "median_radius": None,
        }
    return {
        "num_segments": count,
        "mean_offset_mse": float(np.mean(sq_errors)),
        "mean_l2_error": float(np.mean(l2_errors)),
        "mean_radius": float(np.mean(radii)),
        "median_radius": float(np.median(radii)),
    }


def summarize_segment_errors(
    offsets: np.ndarray,
    reconstructed_offsets: np.ndarray,
    percentile_edges: np.ndarray,
    percentile_spans: Sequence[Tuple[int, int]],
) -> dict:
    offset_count = min(offsets.shape[0], reconstructed_offsets.shape[0])
    if offset_count == 0:
        empty_subset = summarize_segment_subset(
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
        return {
            "draw_flag_metrics": {"draw_0": empty_subset, "draw_1": empty_subset},
            "segment_length_percentile_buckets": [],
            "segment_sq_errors": np.zeros((0,), dtype=np.float32),
            "segment_l2_errors": np.zeros((0,), dtype=np.float32),
            "segment_radii": np.zeros((0,), dtype=np.float32),
            "segment_draw_flags": np.zeros((0,), dtype=np.int64),
        }

    orig = np.asarray(offsets[:offset_count], dtype=np.float32)
    recon = np.asarray(reconstructed_offsets[:offset_count], dtype=np.float32)
    delta = orig[:, :2] - recon[:, :2]
    sq_errors = np.mean(delta * delta, axis=1)
    l2_errors = np.sqrt(np.sum(delta * delta, axis=1))
    radii = np.sqrt(orig[:, 0] * orig[:, 0] + orig[:, 1] * orig[:, 1])
    draw_flags = np.clip(np.rint(orig[:, 2]), 0, 1).astype(np.int64)

    draw_flag_metrics = {}
    for flag in (0, 1):
        mask = draw_flags == flag
        draw_flag_metrics[f"draw_{flag}"] = summarize_segment_subset(
            sq_errors[mask],
            l2_errors[mask],
            radii[mask],
        )

    buckets = []
    for bucket_index, (lower_pct, upper_pct) in enumerate(percentile_spans):
        lower_radius = float(percentile_edges[bucket_index])
        upper_radius = float(percentile_edges[bucket_index + 1])
        if bucket_index == len(percentile_spans) - 1:
            mask = (radii >= lower_radius) & (radii <= upper_radius)
        else:
            mask = (radii >= lower_radius) & (radii < upper_radius)
        bucket = {
            "label": f"p{lower_pct:02d}_{upper_pct:02d}",
            "lower_percentile": int(lower_pct),
            "upper_percentile": int(upper_pct),
            "lower_radius": lower_radius,
            "upper_radius": upper_radius,
            "num_segments": int(np.sum(mask)),
            "mean_offset_mse": float(np.mean(sq_errors[mask])) if np.any(mask) else None,
            "mean_l2_error": float(np.mean(l2_errors[mask])) if np.any(mask) else None,
            "draw_flag_metrics": {},
        }
        for flag in (0, 1):
            flag_mask = mask & (draw_flags == flag)
            bucket["draw_flag_metrics"][f"draw_{flag}"] = summarize_segment_subset(
                sq_errors[flag_mask],
                l2_errors[flag_mask],
                radii[flag_mask],
            )
        buckets.append(bucket)

    return {
        "draw_flag_metrics": draw_flag_metrics,
        "segment_length_percentile_buckets": buckets,
        "segment_sq_errors": sq_errors,
        "segment_l2_errors": l2_errors,
        "segment_radii": radii,
        "segment_draw_flags": draw_flags,
    }


def select_tokenizer_diagnostic_examples(
    split_map: Dict[str, ProcessedWordSplit],
    per_source_split: int,
    min_offsets: int = 8,
) -> List[dict]:
    selected: List[dict] = []
    for split_name in ["train", "val", "test"]:
        split = split_map.get(split_name)
        if split is None:
            continue
        for source in ["deepwriting", "iamondb"]:
            matches = []
            for index, (sample_id, text, offsets, sample_source) in enumerate(
                zip(split.sample_ids, split.texts, split.offsets, split.sources)
            ):
                if str(sample_source) != source:
                    continue
                arr = np.asarray(offsets, dtype=np.float32)
                if arr.ndim != 2 or arr.shape[0] < min_offsets:
                    continue
                matches.append(
                    {
                        "split": split_name,
                        "source": source,
                        "index": index,
                        "sample_id": str(sample_id),
                        "text": str(text),
                        "offsets": arr,
                    }
                )
                if len(matches) >= per_source_split:
                    break
            selected.extend(matches)
    return selected


def run_tokenizer_diagnostics(
    *,
    split_map: Dict[str, ProcessedWordSplit],
    tokenizer_spec: TokenizerSpec,
    stroke_tokenizer: PolarOffsetTokenizer,
    out_dir: Path,
    per_source_split: int,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt = maybe_import_matplotlib()

    selected = select_tokenizer_diagnostic_examples(split_map, per_source_split=per_source_split)
    percentile_spans: List[Tuple[int, int]] = [(0, 25), (25, 50), (50, 75), (75, 90), (90, 100)]
    all_selected_radii = []
    for item in selected:
        offsets = np.asarray(item["offsets"], dtype=np.float32)
        if offsets.ndim != 2 or offsets.shape[0] == 0:
            continue
        all_selected_radii.append(np.sqrt(offsets[:, 0] * offsets[:, 0] + offsets[:, 1] * offsets[:, 1]))
    if all_selected_radii:
        percentile_edges = np.percentile(
            np.concatenate(all_selected_radii),
            [span[0] for span in percentile_spans] + [percentile_spans[-1][1]],
        ).astype(np.float32)
    else:
        percentile_edges = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    manifest = []
    point_mses: List[float] = []
    offset_mses: List[float] = []
    draw_accuracies: List[float] = []
    global_sq_errors: List[np.ndarray] = []
    global_l2_errors: List[np.ndarray] = []
    global_radii: List[np.ndarray] = []
    global_draw_flags: List[np.ndarray] = []

    for item in selected:
        offsets = item["offsets"]
        token_ids = stroke_tokenizer.encode_offsets(offsets)
        reconstructed_offsets = stroke_tokenizer.decode_tokens_to_offsets(token_ids)
        orig_points = offsets_to_absolute_points(offsets)
        recon_points = offsets_to_absolute_points(reconstructed_offsets)

        point_count = min(orig_points.shape[0], recon_points.shape[0])
        offset_count = min(offsets.shape[0], reconstructed_offsets.shape[0])
        point_mse = (
            float(np.mean((orig_points[:point_count, :2] - recon_points[:point_count, :2]) ** 2))
            if point_count
            else 0.0
        )
        offset_mse = (
            float(np.mean((offsets[:offset_count, :2] - reconstructed_offsets[:offset_count, :2]) ** 2))
            if offset_count
            else 0.0
        )
        draw_accuracy = (
            float(np.mean(offsets[:offset_count, 2] == reconstructed_offsets[:offset_count, 2]))
            if offset_count
            else 1.0
        )
        point_mses.append(point_mse)
        offset_mses.append(offset_mse)
        draw_accuracies.append(draw_accuracy)
        segment_summary = summarize_segment_errors(
            offsets,
            reconstructed_offsets,
            percentile_edges=percentile_edges,
            percentile_spans=percentile_spans,
        )
        global_sq_errors.append(segment_summary.pop("segment_sq_errors"))
        global_l2_errors.append(segment_summary.pop("segment_l2_errors"))
        global_radii.append(segment_summary.pop("segment_radii"))
        global_draw_flags.append(segment_summary.pop("segment_draw_flags"))

        plot_path = out_dir / f"{item['split']}_{item['source']}_{item['sample_id'].replace(':', '_')}.png"
        if plt is not None:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4))
            plot_points(axes[0], orig_points, f"Original / {item['text']}")
            plot_points(axes[1], recon_points, f"Tokenized -> detokenized")
            plot_overlay_points(axes[2], orig_points, recon_points, "Overlay")
            fig.suptitle(f"{tokenizer_spec.name} / {item['split']} / {item['source']}", fontsize=11)
            fig.tight_layout()
            fig.savefig(plot_path, dpi=180, bbox_inches="tight")
            plt.close(fig)

        manifest.append(
            {
                "sample_id": item["sample_id"],
                "text": item["text"],
                "split": item["split"],
                "source": item["source"],
                "plot_path": plot_path.as_posix(),
                "num_offsets": int(offsets.shape[0]),
                "num_tokens": len(token_ids),
                "point_mse": point_mse,
                "offset_mse": offset_mse,
                "draw_accuracy": draw_accuracy,
                "segment_error_metrics": segment_summary,
            }
        )

    all_sq = np.concatenate(global_sq_errors) if global_sq_errors else np.zeros((0,), dtype=np.float32)
    all_l2 = np.concatenate(global_l2_errors) if global_l2_errors else np.zeros((0,), dtype=np.float32)
    all_radii = np.concatenate(global_radii) if global_radii else np.zeros((0,), dtype=np.float32)
    all_draw = np.concatenate(global_draw_flags) if global_draw_flags else np.zeros((0,), dtype=np.int64)
    aggregate_draw_metrics = {}
    for flag in (0, 1):
        mask = all_draw == flag
        aggregate_draw_metrics[f"draw_{flag}"] = summarize_segment_subset(
            all_sq[mask],
            all_l2[mask],
            all_radii[mask],
        )
    aggregate_buckets = []
    for bucket_index, (lower_pct, upper_pct) in enumerate(percentile_spans):
        lower_radius = float(percentile_edges[bucket_index])
        upper_radius = float(percentile_edges[bucket_index + 1])
        if bucket_index == len(percentile_spans) - 1:
            mask = (all_radii >= lower_radius) & (all_radii <= upper_radius)
        else:
            mask = (all_radii >= lower_radius) & (all_radii < upper_radius)
        bucket = {
            "label": f"p{lower_pct:02d}_{upper_pct:02d}",
            "lower_percentile": int(lower_pct),
            "upper_percentile": int(upper_pct),
            "lower_radius": lower_radius,
            "upper_radius": upper_radius,
            "num_segments": int(np.sum(mask)),
            "mean_offset_mse": float(np.mean(all_sq[mask])) if np.any(mask) else None,
            "mean_l2_error": float(np.mean(all_l2[mask])) if np.any(mask) else None,
            "draw_flag_metrics": {},
        }
        for flag in (0, 1):
            flag_mask = mask & (all_draw == flag)
            bucket["draw_flag_metrics"][f"draw_{flag}"] = summarize_segment_subset(
                all_sq[flag_mask],
                all_l2[flag_mask],
                all_radii[flag_mask],
            )
        aggregate_buckets.append(bucket)

    summary = {
        "tokenizer_spec": tokenizer_spec_to_dict(tokenizer_spec),
        "radius_codebooks": stroke_tokenizer.radius_codebooks.to_dict(),
        "radius_representative_value": tokenizer_spec.radius_decode_mode,
        "mean_point_mse": float(np.mean(point_mses)) if point_mses else None,
        "mean_offset_mse": float(np.mean(offset_mses)) if offset_mses else None,
        "mean_draw_accuracy": float(np.mean(draw_accuracies)) if draw_accuracies else None,
        "aggregate_draw_flag_metrics": aggregate_draw_metrics,
        "segment_length_percentile_buckets": aggregate_buckets,
        "num_examples": len(manifest),
        "examples": manifest,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
