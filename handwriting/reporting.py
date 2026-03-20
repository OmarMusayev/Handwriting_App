from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np

from .checkpoint import TrainState
from .tokenizers import TokenizerSpec, tokenizer_spec_to_dict


def write_writer_conditioning_artifacts(
    *,
    out_dir: Path,
    sample_dir: Path,
    args,
    writer_vocab,
    writer_map_path: Path,
    panel_writers_path: Path,
    panel_writers: List[dict],
) -> dict:
    summary = {
        "enabled": bool(getattr(args, "use_writer_conditioning", False)),
        "writer_embed_dim": int(getattr(args, "writer_embed_dim", 0)),
        "writer_conditioning_mode": str(getattr(args, "writer_conditioning_mode", "disabled")),
        "writer_unseen_policy": str(getattr(args, "writer_unseen_policy", "error")),
        "num_writer_embeddings": int(writer_vocab.num_embeddings),
        "num_train_writers": int(writer_vocab.num_train_writers),
        "unknown_writer_id": writer_vocab.unknown_writer_id,
        "unknown_writer_index": int(writer_vocab.unknown_writer_index),
        "writer_map_path": writer_map_path.as_posix(),
        "panel_writers_path": panel_writers_path.as_posix(),
        "panel_writers": panel_writers,
        "split_stats": writer_vocab.split_stats,
    }
    (out_dir / "writer_conditioning_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Writer Conditioning Summary",
        "",
        f"- Enabled: `{summary['enabled']}`",
        f"- Conditioning mode: `{summary['writer_conditioning_mode']}`",
        f"- Writer embedding dim: `{summary['writer_embed_dim']}`",
        f"- Writer unseen policy: `{summary['writer_unseen_policy']}`",
        f"- Number of writer embeddings: `{summary['num_writer_embeddings']}`",
        f"- Number of train writers: `{summary['num_train_writers']}`",
        f"- Unknown writer token: `{summary['unknown_writer_id']}` at index `{summary['unknown_writer_index']}`",
        f"- Writer map: `{summary['writer_map_path']}`",
        f"- Panel writers config: `{summary['panel_writers_path']}`",
        "",
        "## Panel Writers",
        "",
    ]
    for writer in panel_writers:
        lines.append(
            f"- `{writer.get('writer_id')}` -> index `{writer.get('writer_index')}` "
            f"(train_count=`{writer.get('train_count')}`)"
        )
    lines.extend(["", "## Split Coverage", ""])
    for split_name, stats in summary["split_stats"].items():
        lines.append(
            f"- `{split_name}`: sample_count=`{stats.get('sample_count')}` "
            f"writer_count=`{stats.get('writer_count')}` "
            f"seen=`{stats.get('seen_writer_count')}` "
            f"unseen=`{stats.get('unseen_writer_count')}`"
        )
        unseen_ids = stats.get("unseen_writer_ids") or []
        if unseen_ids:
            lines.append(f"  unseen writer ids: `{', '.join(unseen_ids[:20])}`")
    md_text = "\n".join(lines)
    (out_dir / "writer_conditioning_summary.md").write_text(md_text)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "writer_conditioning_summary.md").write_text(md_text)
    return summary


def write_style_cluster_conditioning_artifacts(
    *,
    out_dir: Path,
    sample_dir: Path,
    args,
    style_cluster_map,
    style_cluster_map_path: Path,
    panel_clusters_path: Path,
    panel_clusters: List[dict],
) -> dict:
    summary = {
        "enabled": bool(getattr(args, "use_style_cluster_conditioning", False)),
        "style_cluster_embed_dim": int(getattr(args, "style_cluster_embed_dim", 0)),
        "cluster_conditioning_mode": str(getattr(args, "cluster_conditioning_mode", "disabled")),
        "num_style_clusters": int(style_cluster_map.num_clusters),
        "chosen_k": int(style_cluster_map.chosen_k),
        "train_writer_count": int(len(style_cluster_map.train_writer_ids)),
        "style_cluster_map_path": style_cluster_map_path.as_posix(),
        "panel_clusters_path": panel_clusters_path.as_posix(),
        "panel_clusters": panel_clusters,
        "feature_columns": style_cluster_map.feature_columns,
        "split_stats": style_cluster_map.split_stats,
        "cluster_stats": style_cluster_map.cluster_stats,
        "cluster_fit_summary_path": style_cluster_map.fit_summary_path,
    }
    (out_dir / "style_cluster_conditioning_summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        "# Style Cluster Conditioning Summary",
        "",
        f"- Enabled: `{summary['enabled']}`",
        f"- Conditioning mode: `{summary['cluster_conditioning_mode']}`",
        f"- Style-cluster embedding dim: `{summary['style_cluster_embed_dim']}`",
        f"- Number of style clusters: `{summary['num_style_clusters']}`",
        f"- Chosen k: `{summary['chosen_k']}`",
        f"- Train writer count: `{summary['train_writer_count']}`",
        f"- Style-cluster map: `{summary['style_cluster_map_path']}`",
        f"- Panel clusters config: `{summary['panel_clusters_path']}`",
        f"- Cluster fit summary: `{summary['cluster_fit_summary_path']}`",
        "",
        "## Representative Panel Clusters",
        "",
    ]
    for cluster in panel_clusters:
        lines.append(
            f"- `{cluster.get('label')}` -> cluster_id `{cluster.get('cluster_id')}` "
            f"(train_writers=`{cluster.get('train_writer_count')}`, "
            f"train_samples=`{cluster.get('train_sample_count')}`)"
        )
    lines.extend(["", "## Split Cluster Coverage", ""])
    for split_name, stats in summary["split_stats"].items():
        lines.append(
            f"- `{split_name}`: sample_count=`{stats.get('sample_count')}` "
            f"writer_count=`{stats.get('writer_count')}` "
            f"sample_count_by_cluster=`{stats.get('sample_count_by_cluster')}` "
            f"writer_count_by_cluster=`{stats.get('writer_count_by_cluster')}`"
        )
    md_text = "\n".join(lines)
    (out_dir / "style_cluster_conditioning_summary.md").write_text(md_text)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "style_cluster_conditioning_summary.md").write_text(md_text)
    return summary


def write_panel_epoch_summary(epoch_dir: Path, summary: dict) -> None:
    epoch_dir.mkdir(parents=True, exist_ok=True)
    (epoch_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = [
        f"# Evaluation Epoch {summary['epoch']:02d}",
        "",
        f"- Current epoch: `{summary['epoch']}`",
        f"- Best checkpoint up to this point: `{summary['best_checkpoint']['label']}`",
        f"- Best checkpoint epoch: `{summary['best_checkpoint']['epoch']}`",
        f"- Best checkpoint step: `{summary['best_checkpoint']['step']}`",
        "",
    ]
    writer_conditioning = summary.get("writer_conditioning") or {}
    panel_writers = writer_conditioning.get("panel_writers") or []
    style_cluster_conditioning = summary.get("style_cluster_conditioning") or {}
    panel_clusters = style_cluster_conditioning.get("panel_clusters") or []
    if writer_conditioning:
        lines.extend(
            [
                f"- Writer conditioning enabled: `{writer_conditioning.get('enabled')}`",
                f"- Panel writers: `{', '.join(str(item.get('writer_id')) for item in panel_writers) if panel_writers else 'default'}`",
                "",
            ]
        )
    if style_cluster_conditioning:
        lines.extend(
            [
                f"- Style-cluster conditioning enabled: `{style_cluster_conditioning.get('enabled')}`",
                f"- Panel clusters: `{', '.join(str(item.get('label')) for item in panel_clusters) if panel_clusters else 'default'}`",
                "",
            ]
        )
    for checkpoint in summary["evaluated_checkpoints"]:
        lines.append(f"## {checkpoint['label']}")
        lines.append("")
        lines.append(f"- Checkpoint epoch: `{checkpoint['checkpoint_epoch']}`")
        lines.append(f"- Checkpoint step: `{checkpoint['checkpoint_step']}`")
        lines.append(f"- Grid: `{checkpoint['grid_path']}`")
        lines.append("")
        for mode in checkpoint["decoding_modes"]:
            lines.append(
                f"- `{mode['name']}`: eos_completion_rate={mode['summary']['eos_completion_rate']} "
                f"mean_generated_offsets={mode['summary']['mean_generated_offsets']}"
            )
        lines.append("")
    (epoch_dir / "summary.md").write_text("\n".join(lines))


def summarize_eos_history(eos_history: List[dict]) -> dict:
    if not eos_history:
        return {
            "num_evaluated_epochs": 0,
            "best_epoch_by_eos_top1_rate": None,
            "best_epoch_by_mean_eos_prob": None,
            "latest_epoch": None,
            "eos_trend": "no_data",
            "eos_bottleneck_hint": "no_data",
        }

    ordered = sorted(eos_history, key=lambda item: (int(item.get("epoch", 0)), int(item.get("global_step", 0))))
    best_top1 = max(
        ordered,
        key=lambda item: float(item.get("eos_top1_rate_at_true_end") if item.get("eos_top1_rate_at_true_end") is not None else -1.0),
    )
    best_prob = max(
        ordered,
        key=lambda item: float(item.get("mean_eos_prob_at_true_end") if item.get("mean_eos_prob_at_true_end") is not None else -1.0),
    )
    first = ordered[0]
    last = ordered[-1]
    first_prob = float(first.get("mean_eos_prob_at_true_end") or 0.0)
    last_prob = float(last.get("mean_eos_prob_at_true_end") or 0.0)
    first_top1 = float(first.get("eos_top1_rate_at_true_end") or 0.0)
    last_top1 = float(last.get("eos_top1_rate_at_true_end") or 0.0)

    if last_prob >= first_prob + 0.05 or last_top1 >= first_top1 + 0.10:
        eos_trend = "strengthening"
    elif last_prob <= first_prob - 0.05 or last_top1 <= first_top1 - 0.10:
        eos_trend = "weakening"
    else:
        eos_trend = "roughly_flat"

    if last_top1 < 0.50 or last_prob < 0.30:
        bottleneck_hint = "EOS likely remains a material bottleneck"
    elif last_top1 > 0.85 and last_prob > 0.60:
        bottleneck_hint = "EOS looks reasonably strong; endings are less likely to be the main bottleneck"
    else:
        bottleneck_hint = "EOS is mixed; it may contribute but is not decisively dominant"

    return {
        "num_evaluated_epochs": int(len(ordered)),
        "best_epoch_by_eos_top1_rate": {
            "epoch": int(best_top1["epoch"]),
            "global_step": int(best_top1["global_step"]) if best_top1.get("global_step") is not None else None,
            "value": float(best_top1["eos_top1_rate_at_true_end"]),
            "checkpoint_path": best_top1.get("checkpoint_path"),
        },
        "best_epoch_by_mean_eos_prob": {
            "epoch": int(best_prob["epoch"]),
            "global_step": int(best_prob["global_step"]) if best_prob.get("global_step") is not None else None,
            "value": float(best_prob["mean_eos_prob_at_true_end"]),
            "checkpoint_path": best_prob.get("checkpoint_path"),
        },
        "latest_epoch": {
            "epoch": int(last["epoch"]),
            "global_step": int(last["global_step"]) if last.get("global_step") is not None else None,
            "eos_top1_rate_at_true_end": last.get("eos_top1_rate_at_true_end"),
            "mean_eos_prob_at_true_end": last.get("mean_eos_prob_at_true_end"),
            "mean_eos_rank_at_true_end": last.get("mean_eos_rank_at_true_end"),
            "checkpoint_path": last.get("checkpoint_path"),
            "val_nll": last.get("val_nll"),
        },
        "eos_trend": eos_trend,
        "eos_bottleneck_hint": bottleneck_hint,
    }


def write_eos_diagnostics_artifacts(out_dir: Path, sample_dir: Path, eos_history: List[dict]) -> dict:
    summary = summarize_eos_history(eos_history)
    jsonl_path = out_dir / "per_epoch_eos_diagnostics.jsonl"
    with jsonl_path.open("w") as handle:
        for row in sorted(eos_history, key=lambda item: (int(item.get("epoch", 0)), int(item.get("global_step", 0)))):
            handle.write(json.dumps(row) + "\n")

    summary_path = out_dir / "eos_diagnostics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    lines = [
        "# EOS Diagnostics Summary",
        "",
        f"- Evaluated epochs: `{summary['num_evaluated_epochs']}`",
        f"- EOS trend: `{summary['eos_trend']}`",
        f"- Bottleneck hint: `{summary['eos_bottleneck_hint']}`",
        "",
    ]
    best_top1 = summary.get("best_epoch_by_eos_top1_rate") or {}
    if best_top1:
        lines.extend(
            [
                "## Best EOS Top-1 Epoch",
                "",
                f"- Epoch: `{best_top1.get('epoch')}`",
                f"- Step: `{best_top1.get('global_step')}`",
                f"- EOS top-1 rate at true end: `{best_top1.get('value')}`",
                f"- Checkpoint: `{best_top1.get('checkpoint_path')}`",
                "",
            ]
        )
    best_prob = summary.get("best_epoch_by_mean_eos_prob") or {}
    if best_prob:
        lines.extend(
            [
                "## Best Mean EOS-Probability Epoch",
                "",
                f"- Epoch: `{best_prob.get('epoch')}`",
                f"- Step: `{best_prob.get('global_step')}`",
                f"- Mean EOS probability at true end: `{best_prob.get('value')}`",
                f"- Checkpoint: `{best_prob.get('checkpoint_path')}`",
                "",
            ]
        )
    latest = summary.get("latest_epoch") or {}
    if latest:
        lines.extend(
            [
                "## Latest Evaluated Epoch",
                "",
                f"- Epoch: `{latest.get('epoch')}`",
                f"- Step: `{latest.get('global_step')}`",
                f"- Validation NLL: `{latest.get('val_nll')}`",
                f"- EOS top-1 rate at true end: `{latest.get('eos_top1_rate_at_true_end')}`",
                f"- Mean EOS probability at true end: `{latest.get('mean_eos_prob_at_true_end')}`",
                f"- Mean EOS rank at true end: `{latest.get('mean_eos_rank_at_true_end')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Per-Epoch Rows",
            "",
        ]
    )
    if not eos_history:
        lines.append("- No EOS diagnostics were recorded.")
    else:
        for row in sorted(eos_history, key=lambda item: (int(item.get("epoch", 0)), int(item.get("global_step", 0)))):
            lines.append(
                f"- Epoch `{row['epoch']}` step `{row.get('global_step')}`: "
                f"top1={row.get('eos_top1_rate_at_true_end')} "
                f"mean_prob={row.get('mean_eos_prob_at_true_end')} "
                f"mean_rank={row.get('mean_eos_rank_at_true_end')} "
                f"val_nll={row.get('val_nll')}"
            )

    md_text = "\n".join(lines)
    (out_dir / "eos_diagnostics_summary.md").write_text(md_text)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "eos_diagnostics_summary.md").write_text(md_text)
    return summary


def write_run_summary(
    *,
    out_dir: Path,
    sample_dir: Path,
    args,
    tokenizer_spec: TokenizerSpec,
    steps_per_epoch: int,
    state: TrainState,
    eval_history: List[dict],
    panel_history: List[dict],
    eos_history: List[dict],
    test_metrics: Optional[dict],
    tokenizer_diagnostics: Optional[dict],
) -> None:
    val_entries = [entry for entry in eval_history if entry.get("split") == "val"]
    best_entry = min(val_entries, key=lambda entry: entry["nll"]) if val_entries else None
    best_step = int(best_entry["step"]) if best_entry else state.best_val_step
    best_epoch = int(best_entry["epoch"]) if best_entry and best_entry.get("epoch") is not None else state.best_val_epoch

    panel_proxy_ranking = []
    val_by_epoch = {int(entry["epoch"]): entry for entry in val_entries if entry.get("epoch") is not None}
    for panel in panel_history:
        current_epoch = int(panel["epoch"])
        current_val = val_by_epoch.get(current_epoch, {})
        panel_proxy_ranking.append(
            {
                "panel_epoch": current_epoch,
                "current_epoch_val_nll": current_val.get("nll"),
                "best_checkpoint_epoch": panel.get("best_checkpoint", {}).get("epoch"),
                "best_checkpoint_val_nll": panel.get("best_checkpoint", {}).get("nll"),
                "current_grid": next(
                    (checkpoint["grid_path"] for checkpoint in panel["evaluated_checkpoints"] if checkpoint["label"] == "current"),
                    None,
                ),
                "best_grid": next(
                    (checkpoint["grid_path"] for checkpoint in panel["evaluated_checkpoints"] if checkpoint["label"] != "current"),
                    None,
                ),
            }
        )
    panel_proxy_ranking.sort(
        key=lambda item: float(item["current_epoch_val_nll"])
        if item["current_epoch_val_nll"] is not None
        else float("inf")
    )

    decoding_mode_summary = {}
    for panel in panel_history:
        for checkpoint in panel.get("evaluated_checkpoints", []):
            for mode in checkpoint.get("decoding_modes", []):
                decoding_mode_summary.setdefault(mode["name"], []).append(mode["summary"])
    aggregated_decoding_modes = {}
    for mode_name, summaries in decoding_mode_summary.items():
        eos_rates = [item["eos_completion_rate"] for item in summaries if item["eos_completion_rate"] is not None]
        mean_offsets = [item["mean_generated_offsets"] for item in summaries if item["mean_generated_offsets"] is not None]
        aggregated_decoding_modes[mode_name] = {
            "num_panels": int(len(summaries)),
            "mean_eos_completion_rate": float(np.mean(eos_rates)) if eos_rates else None,
            "mean_generated_offsets": float(np.mean(mean_offsets)) if mean_offsets else None,
        }

    training_regime = "insufficient_validation_history"
    larger_model_guidance = "unclear"
    if val_entries:
        final_val = val_entries[-1]
        if best_entry is not None and best_epoch is not None and best_epoch >= max(1, args.epochs - 5):
            training_regime = "still_improving_or_not_fully_converged"
            larger_model_guidance = "plausible_if_samples_improve_with_depth"
        elif best_entry is not None and final_val["nll"] > best_entry["nll"] * 1.02:
            training_regime = "regressed_after_best_checkpoint"
            larger_model_guidance = "not_before_fixing_optimization_or_regularization"
        else:
            training_regime = "stable_plateau"
            larger_model_guidance = "reasonable_next_comparison"

    eos_summary = summarize_eos_history(eos_history)

    summary = {
        "args": vars(args),
        "tokenizer_spec": tokenizer_spec_to_dict(tokenizer_spec),
        "steps_per_epoch": steps_per_epoch,
        "writer_conditioning": {
            "enabled": bool(getattr(args, "use_writer_conditioning", False)),
            "writer_embed_dim": int(getattr(args, "writer_embed_dim", 0)),
            "writer_conditioning_mode": str(getattr(args, "writer_conditioning_mode", "disabled")),
            "writer_unseen_policy": str(getattr(args, "writer_unseen_policy", "error")),
            "num_writer_embeddings": int(getattr(args, "num_writer_embeddings", 1)),
            "writer_map_path": getattr(args, "writer_map_path", None),
            "panel_writers_path": getattr(args, "panel_writers_path", None),
            "panel_writers": getattr(args, "panel_writers", []),
        },
        "style_cluster_conditioning": {
            "enabled": bool(getattr(args, "use_style_cluster_conditioning", False)),
            "style_cluster_embed_dim": int(getattr(args, "style_cluster_embed_dim", 0)),
            "cluster_conditioning_mode": str(getattr(args, "cluster_conditioning_mode", "disabled")),
            "num_style_clusters": int(getattr(args, "num_style_clusters", 1)),
            "style_cluster_map_path": getattr(args, "style_cluster_map_path", None),
            "panel_clusters_path": getattr(args, "panel_clusters_path", None),
            "panel_clusters": getattr(args, "panel_clusters", []),
        },
        "best_val_nll": state.best_val_loss,
        "best_val_ppl": math.exp(min(state.best_val_loss, 20)) if math.isfinite(state.best_val_loss) else None,
        "best_step": best_step,
        "best_epoch": best_epoch,
        "best_checkpoint_path": state.best_checkpoint_path,
        "eval_history": eval_history,
        "panel_history": panel_history,
        "panel_proxy_ranking": panel_proxy_ranking,
        "decoding_mode_summary": aggregated_decoding_modes,
        "eos_history": eos_history,
        "eos_summary": eos_summary,
        "training_regime": training_regime,
        "larger_model_guidance": larger_model_guidance,
        "test_metrics": test_metrics,
        "tokenizer_diagnostics": tokenizer_diagnostics,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    report_lines = [
        f"# {args.experiment_name}",
        "",
        f"- Seed: `{args.seed}`",
        f"- Best validation epoch: `{best_epoch}`",
        f"- Best validation NLL: `{summary['best_val_nll']}`",
        f"- Best validation PPL: `{summary['best_val_ppl']}`",
        f"- Best checkpoint path: `{state.best_checkpoint_path}`",
        f"- Training regime heuristic: `{training_regime}`",
        f"- Larger model heuristic: `{larger_model_guidance}`",
        f"- Writer conditioning enabled: `{summary['writer_conditioning']['enabled']}`",
        f"- Writer conditioning mode: `{summary['writer_conditioning']['writer_conditioning_mode']}`",
        f"- Writer embeddings: `{summary['writer_conditioning']['num_writer_embeddings']}`",
        f"- Style-cluster conditioning enabled: `{summary['style_cluster_conditioning']['enabled']}`",
        f"- Style-cluster conditioning mode: `{summary['style_cluster_conditioning']['cluster_conditioning_mode']}`",
        f"- Style clusters: `{summary['style_cluster_conditioning']['num_style_clusters']}`",
        f"- EOS trend: `{eos_summary.get('eos_trend')}`",
        f"- EOS bottleneck hint: `{eos_summary.get('eos_bottleneck_hint')}`",
        "",
        "## Fixed Evaluation Panel",
        "",
        f"- Words: {', '.join(args.eval_words)}",
        f"- Decoding modes: {', '.join(mode['name'] for mode in args.decoding_modes)}",
        f"- Panel writers: {', '.join(item['writer_id'] for item in getattr(args, 'panel_writers', [])) if getattr(args, 'panel_writers', []) else 'default'}",
        f"- Panel clusters: {', '.join(item['label'] for item in getattr(args, 'panel_clusters', [])) if getattr(args, 'panel_clusters', []) else 'default'}",
        "",
        "## Current vs Best At Panel Epochs",
        "",
    ]
    if not panel_proxy_ranking:
        report_lines.append("- No panel evaluations were run.")
    else:
        for item in panel_proxy_ranking:
            report_lines.append(
                f"- Epoch `{item['panel_epoch']}`: current_val_nll={item['current_epoch_val_nll']} "
                f"best_epoch={item['best_checkpoint_epoch']} best_val_nll={item['best_checkpoint_val_nll']} "
                f"current_grid=`{item['current_grid']}` best_grid=`{item['best_grid']}`"
            )
    report_lines.extend(
        [
            "",
            "## Decoding Mode Structural Summary",
            "",
        ]
    )
    if not aggregated_decoding_modes:
        report_lines.append("- No decoding mode summaries were recorded.")
    else:
        for mode_name, mode_summary in aggregated_decoding_modes.items():
            report_lines.append(
                f"- `{mode_name}`: mean_eos_completion_rate={mode_summary['mean_eos_completion_rate']} "
                f"mean_generated_offsets={mode_summary['mean_generated_offsets']}"
            )
    report_lines.extend(
        [
            "",
            "## EOS Diagnostics",
            "",
            f"- Evaluated epochs: `{eos_summary.get('num_evaluated_epochs')}`",
            f"- Best EOS top-1 epoch: `{(eos_summary.get('best_epoch_by_eos_top1_rate') or {}).get('epoch')}`",
            f"- Best mean EOS-prob epoch: `{(eos_summary.get('best_epoch_by_mean_eos_prob') or {}).get('epoch')}`",
            f"- EOS trend: `{eos_summary.get('eos_trend')}`",
            f"- Bottleneck hint: `{eos_summary.get('eos_bottleneck_hint')}`",
        ]
    )
    report_lines.extend(
        [
            "",
            "## Proxy-Strong Panel Epochs",
            "",
        ]
    )
    if not panel_proxy_ranking:
        report_lines.append("- No ranked panel epochs available.")
    else:
        for item in panel_proxy_ranking[:3]:
            report_lines.append(
                f"- Epoch `{item['panel_epoch']}` with current_val_nll={item['current_epoch_val_nll']} "
                f"(grid `{item['current_grid']}`)"
            )
    (sample_dir / "comparison_report.json").write_text(json.dumps(summary, indent=2))
    (sample_dir / "comparison_report.md").write_text("\n".join(report_lines))
