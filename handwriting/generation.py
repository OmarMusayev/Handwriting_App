from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import TextVocab
from .seed import stable_seed_from_components
from .tokenizers import PolarOffsetTokenizer, StrokeVocab
from .utils import maybe_import_matplotlib, slugify_text


@dataclass(frozen=True)
class DecodingMode:
    name: str
    temperature: float
    top_k: int
    greedy: bool = False


@torch.no_grad()
def generate_tokens(
    model,
    text: str,
    text_vocab: TextVocab,
    stroke_vocab: StrokeVocab,
    max_text_len: int,
    max_gen_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 0,
    greedy: bool = False,
    sample_seed: Optional[int] = None,
    writer_id: Optional[int] = None,
    style_cluster_id: Optional[int] = None,
) -> list[int]:
    model.eval()
    text_ids = text_vocab.encode(text, add_bos_eos=True)[:max_text_len]
    if text_ids[-1] != text_vocab.eos_id:
        text_ids[-1] = text_vocab.eos_id
    text_ids_t = torch.tensor(text_ids, dtype=torch.long, device=device).unsqueeze(0)
    text_pad_mask = text_ids_t.eq(text_vocab.pad_id)
    writer_ids_t = None
    if writer_id is not None:
        writer_ids_t = torch.tensor([int(writer_id)], dtype=torch.long, device=device)
    style_cluster_ids_t = None
    if style_cluster_id is not None:
        style_cluster_ids_t = torch.tensor([int(style_cluster_id)], dtype=torch.long, device=device)
    generator = None
    if sample_seed is not None and not greedy:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sample_seed))

    generated = [stroke_vocab.bos_id]
    for _ in range(max_gen_tokens):
        stroke_in = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
        stroke_pad_mask = stroke_in.eq(stroke_vocab.pad_id)
        logits = model(
            text_ids_t,
            stroke_in,
            text_pad_mask,
            stroke_pad_mask,
            writer_ids=writer_ids_t,
            style_cluster_ids=style_cluster_ids_t,
        )
        next_logits = logits[0, -1]
        if greedy:
            next_token = int(torch.argmax(next_logits).item())
        else:
            next_logits = (next_logits / max(temperature, 1e-5)).float().cpu()
            if top_k > 0:
                values, indices = torch.topk(next_logits, k=min(top_k, next_logits.numel()))
                probs = torch.softmax(values, dim=-1)
                sampled_index = torch.multinomial(probs, num_samples=1, generator=generator)
                next_token = int(indices[sampled_index].item())
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = int(torch.multinomial(probs, num_samples=1, generator=generator).item())
        generated.append(next_token)
        if next_token == stroke_vocab.eos_id:
            break

    model.train()
    return generated


def offsets_to_absolute_points(offsets: np.ndarray) -> np.ndarray:
    if offsets.size == 0:
        return np.zeros((1, 3), dtype=np.float32)
    points = np.zeros((offsets.shape[0] + 1, 3), dtype=np.float32)
    points[1:, :2] = np.cumsum(offsets[:, :2].astype(np.float32), axis=0)
    points[1:, 2] = offsets[:, 2].astype(np.float32)
    return points


def save_sample_json(path: Path, text: str, token_ids: Sequence[int], offsets: np.ndarray, points: np.ndarray) -> None:
    payload = {
        "text": text,
        "token_ids": [int(value) for value in token_ids],
        "offsets": offsets.tolist(),
        "points": points.tolist(),
    }
    path.write_text(json.dumps(payload, indent=2))


def save_sample_plot(path: Path, points: np.ndarray) -> bool:
    plt = maybe_import_matplotlib()
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(10, 4))
    for index in range(1, points.shape[0]):
        if points[index, 2] < 0.5:
            continue
        segment = points[index - 1 : index + 1]
        if segment.shape[0] == 2:
            ax.plot(segment[:, 0], -segment[:, 1], linewidth=1.2, color="black")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def save_generated_sample(
    *,
    sample_dir: Path,
    sample_label: str,
    sample_text: str,
    sample_tokens: Sequence[int],
    offsets: np.ndarray,
    points: np.ndarray,
) -> bool:
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_json = sample_dir / f"{sample_label}.json"
    save_sample_json(sample_json, sample_text, sample_tokens, offsets, points)
    sample_png = sample_dir / f"{sample_label}.png"
    plotted = save_sample_plot(sample_png, points)
    print(f"[sample] saved {sample_json.as_posix()}" + (f" and {sample_png.as_posix()}" if plotted else ""))
    return plotted


def plot_points(
    ax,
    points: np.ndarray,
    title: str,
    *,
    color: str = "tab:blue",
    alpha: float = 1.0,
    linewidth: float = 1.2,
) -> None:
    for index in range(1, points.shape[0]):
        if points[index, 2] < 0.5:
            continue
        segment = points[index - 1 : index + 1]
        if segment.shape[0] == 2:
            ax.plot(segment[:, 0], -segment[:, 1], linewidth=linewidth, color=color, alpha=alpha)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def plot_overlay_points(ax, orig_points: np.ndarray, recon_points: np.ndarray, title: str) -> None:
    plot_points(ax, orig_points, title="", color="tab:blue", alpha=0.85)
    plot_points(ax, recon_points, title="", color="tab:red", alpha=0.55)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def compute_generated_sample_stats(
    *,
    token_ids: Sequence[int],
    offsets: np.ndarray,
    points: np.ndarray,
    stroke_vocab: StrokeVocab,
    max_gen_tokens: int,
) -> dict:
    offsets = np.asarray(offsets, dtype=np.float32)
    points = np.asarray(points, dtype=np.float32)
    draw_mask = offsets[:, 2] >= 0.5 if offsets.size else np.zeros((0,), dtype=bool)
    nondraw_mask = ~draw_mask if offsets.size else np.zeros((0,), dtype=bool)
    segment_lengths = (
        np.sqrt(offsets[:, 0] * offsets[:, 0] + offsets[:, 1] * offsets[:, 1])
        if offsets.size
        else np.zeros((0,), dtype=np.float32)
    )
    x_coords = points[:, 0] if points.size else np.zeros((0,), dtype=np.float32)
    y_coords = points[:, 1] if points.size else np.zeros((0,), dtype=np.float32)
    bbox_width = float(np.max(x_coords) - np.min(x_coords)) if x_coords.size else 0.0
    bbox_height = float(np.max(y_coords) - np.min(y_coords)) if y_coords.size else 0.0
    return {
        "generated_tokens": int(len(token_ids)),
        "generated_offsets": int(offsets.shape[0]),
        "draw_segments": int(np.sum(draw_mask)),
        "nondraw_segments": int(np.sum(nondraw_mask)),
        "draw_path_length": float(np.sum(segment_lengths[draw_mask])) if segment_lengths.size else 0.0,
        "nondraw_path_length": float(np.sum(segment_lengths[nondraw_mask])) if segment_lengths.size else 0.0,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "terminated_with_eos": bool(token_ids and token_ids[-1] == stroke_vocab.eos_id),
        "hit_generation_limit": bool(token_ids and token_ids[-1] != stroke_vocab.eos_id and len(token_ids) > max_gen_tokens),
    }


def summarize_panel_sample_group(samples: Sequence[dict]) -> dict:
    if not samples:
        return {
            "num_samples": 0,
            "mean_generated_offsets": None,
            "mean_draw_segments": None,
            "mean_bbox_width": None,
            "mean_bbox_height": None,
            "mean_draw_path_length": None,
            "eos_completion_rate": None,
        }
    return {
        "num_samples": int(len(samples)),
        "mean_generated_offsets": float(np.mean([item["stats"]["generated_offsets"] for item in samples])),
        "mean_draw_segments": float(np.mean([item["stats"]["draw_segments"] for item in samples])),
        "mean_bbox_width": float(np.mean([item["stats"]["bbox_width"] for item in samples])),
        "mean_bbox_height": float(np.mean([item["stats"]["bbox_height"] for item in samples])),
        "mean_draw_path_length": float(np.mean([item["stats"]["draw_path_length"] for item in samples])),
        "eos_completion_rate": float(np.mean([1.0 if item["stats"]["terminated_with_eos"] else 0.0 for item in samples])),
    }


def save_panel_sample_artifact(
    *,
    output_dir: Path,
    file_stem: str,
    metadata: dict,
    token_ids: Sequence[int],
    offsets: np.ndarray,
    points: np.ndarray,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{file_stem}.json"
    png_path = output_dir / f"{file_stem}.png"
    payload = dict(metadata)
    payload["token_ids"] = [int(value) for value in token_ids]
    payload["offsets"] = offsets.tolist()
    payload["points"] = points.tolist()
    json_path.write_text(json.dumps(payload, indent=2))
    plotted = save_sample_plot(png_path, points)
    return {
        "json_path": json_path.as_posix(),
        "png_path": png_path.as_posix() if plotted else None,
    }


def save_panel_grid(
    *,
    path: Path,
    epoch: int,
    checkpoint_label: str,
    checkpoint_epoch: Optional[int],
    words: Sequence[str],
    decoding_modes: Sequence[DecodingMode],
    condition_entries: Sequence[dict],
    samples_by_key: Dict[Tuple[str, str, str], dict],
) -> Optional[str]:
    plt = maybe_import_matplotlib()
    if plt is None:
        return None
    rows = max(1, len(decoding_modes) * max(1, len(condition_entries)))
    cols = len(words)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.8), squeeze=False)
    row_index = 0
    for condition_entry in condition_entries:
        condition_label = str(
            condition_entry.get("label")
            or condition_entry.get("writer_id")
            or condition_entry.get("cluster_id")
            or "default"
        )
        for mode in decoding_modes:
            for col_index, word in enumerate(words):
                ax = axes[row_index][col_index]
                item = samples_by_key.get((condition_label, mode.name, word))
                if item is not None:
                    plot_points(ax, item["points"], title="")
                    ax.set_title(f"{word}\n{condition_label} / {mode.name}", fontsize=9)
                else:
                    ax.axis("off")
            row_index += 1
    suffix = f"checkpoint epoch {checkpoint_epoch}" if checkpoint_epoch is not None else "checkpoint epoch unknown"
    fig.suptitle(f"Epoch {epoch:02d} / {checkpoint_label} / {suffix}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.as_posix()


def build_sample_seed(
    args,
    panel_epoch: int,
    checkpoint_label: str,
    mode_name: str,
    word: str,
    condition_label: Optional[str] = None,
) -> int:
    return stable_seed_from_components(
        args.seed,
        panel_epoch,
        checkpoint_label,
        condition_label or "default",
        mode_name,
        word,
    )


def build_sample_stem(
    panel_epoch: int,
    checkpoint_label: str,
    mode_name: str,
    word: str,
    condition_label: Optional[str] = None,
) -> str:
    word_slug = slugify_text(word)
    if condition_label:
        condition_slug = slugify_text(condition_label)
        return f"sample_epoch_{panel_epoch:02d}__{checkpoint_label}__{condition_slug}__{mode_name}__{word_slug}"
    return f"sample_epoch_{panel_epoch:02d}__{checkpoint_label}__{mode_name}__{word_slug}"
