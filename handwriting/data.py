from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizers import PolarOffsetTokenizer


class TextVocab:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"

    def __init__(self, texts: Sequence[str]):
        chars = set()
        for text in texts:
            for ch in str(text):
                chars.add(ch)

        ordered = [self.PAD, self.BOS, self.EOS, self.UNK] + sorted(chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(ordered)}
        self.itos: List[str] = ordered
        self.pad_id = self.stoi[self.PAD]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]
        self.unk_id = self.stoi[self.UNK]

    def encode(self, text: str, add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in str(text)]
        if add_bos_eos:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def __len__(self) -> int:
        return len(self.itos)


@dataclass
class ProcessedWordSplit:
    sample_ids: np.ndarray
    texts: np.ndarray
    offsets: np.ndarray
    writer_ids: np.ndarray
    sources: np.ndarray


def slice_loaded_split(split: ProcessedWordSplit, max_samples: Optional[int]) -> ProcessedWordSplit:
    if max_samples is None or max_samples <= 0:
        return split
    count = min(int(max_samples), len(split.sample_ids))
    return ProcessedWordSplit(
        sample_ids=split.sample_ids[:count],
        texts=split.texts[:count],
        offsets=split.offsets[:count],
        writer_ids=split.writer_ids[:count],
        sources=split.sources[:count],
    )


def load_processed_npz(path, max_samples: Optional[int] = None) -> ProcessedWordSplit:
    data = np.load(path, allow_pickle=True)
    required = ["sample_id", "text", "offsets"]
    for key in required:
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {path}")

    writer_ids = data["writer_id"] if "writer_id" in data else np.array(["unknown"] * len(data["sample_id"]), dtype=object)
    sources = data["source"] if "source" in data else np.array(["unknown"] * len(data["sample_id"]), dtype=object)
    split = ProcessedWordSplit(
        sample_ids=np.asarray(data["sample_id"], dtype=object),
        texts=np.asarray(data["text"], dtype=object),
        offsets=np.asarray(data["offsets"], dtype=object),
        writer_ids=np.asarray(writer_ids, dtype=object),
        sources=np.asarray(sources, dtype=object),
    )
    path_str = str(path)
    if "processed_iam_" in path_str:
        unique_sources = {str(value) for value in split.sources.tolist()}
        if unique_sources != {"iamondb"}:
            raise ValueError(
                f"IAM bundle expected only source='iamondb' in {path}, found sources={sorted(unique_sources)}"
            )
    return slice_loaded_split(split, max_samples=max_samples)


class ProcessedWordDataset(Dataset):
    def __init__(
        self,
        split: ProcessedWordSplit,
        split_name: str,
        text_vocab: TextVocab,
        stroke_tokenizer: PolarOffsetTokenizer,
        max_text_len: int,
        max_stroke_tokens: int,
        writer_to_index: Dict[str, int],
        writer_unknown_index: int,
        style_cluster_by_writer: Optional[Dict[str, int]] = None,
        default_style_cluster_id: int = 0,
        writer_unseen_policy: str = "error",
        train_downsample_keep_min: float = 1.0,
        train_downsample_keep_max: float = 1.0,
    ):
        self.split_name = str(split_name)
        self.sample_ids = split.sample_ids
        self.texts = split.texts
        self.offsets = split.offsets
        self.writer_ids = split.writer_ids
        self.text_vocab = text_vocab
        self.stroke_tokenizer = stroke_tokenizer
        self.max_text_len = int(max_text_len)
        self.max_stroke_tokens = int(max_stroke_tokens)
        self.writer_to_index = {str(key): int(value) for key, value in writer_to_index.items()}
        self.writer_unknown_index = int(writer_unknown_index)
        self.style_cluster_by_writer = (
            {str(key): int(value) for key, value in style_cluster_by_writer.items()}
            if style_cluster_by_writer is not None
            else {}
        )
        self.default_style_cluster_id = int(default_style_cluster_id)
        self.writer_unseen_policy = str(writer_unseen_policy)
        self.train_downsample_keep_min = float(train_downsample_keep_min)
        self.train_downsample_keep_max = float(train_downsample_keep_max)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        offsets = np.asarray(self.offsets[idx], dtype=np.float32)
        offsets = maybe_randomly_downsample_offsets(
            offsets,
            keep_min=self.train_downsample_keep_min,
            keep_max=self.train_downsample_keep_max,
        )
        raw_writer_id = str(self.writer_ids[idx])
        writer_index = self.writer_to_index.get(raw_writer_id)
        if writer_index is None:
            if self.writer_unseen_policy == "map_to_unk":
                writer_index = self.writer_unknown_index
            else:
                raise ValueError(
                    f"Unseen writer_id={raw_writer_id} in split={self.split_name} with writer_unseen_policy={self.writer_unseen_policy}"
                )
        style_cluster_id = self.style_cluster_by_writer.get(raw_writer_id, self.default_style_cluster_id)

        text_ids = self.text_vocab.encode(text, add_bos_eos=True)[: self.max_text_len]
        if text_ids[-1] != self.text_vocab.eos_id:
            text_ids[-1] = self.text_vocab.eos_id

        stroke_ids = self.stroke_tokenizer.encode_offsets(offsets)[: self.max_stroke_tokens]
        if stroke_ids[-1] != self.stroke_tokenizer.stroke_vocab.eos_id:
            stroke_ids[-1] = self.stroke_tokenizer.stroke_vocab.eos_id

        stroke_in_ids = [self.stroke_tokenizer.stroke_vocab.bos_id] + stroke_ids[:-1]
        stroke_tgt_ids = stroke_ids

        return {
            "sample_id": str(self.sample_ids[idx]),
            "writer_id_raw": raw_writer_id,
            "writer_ids": torch.tensor(writer_index, dtype=torch.long),
            "style_cluster_ids": torch.tensor(style_cluster_id, dtype=torch.long),
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "stroke_in_ids": torch.tensor(stroke_in_ids, dtype=torch.long),
            "stroke_tgt_ids": torch.tensor(stroke_tgt_ids, dtype=torch.long),
        }


def maybe_randomly_downsample_offsets(offsets: np.ndarray, *, keep_min: float, keep_max: float) -> np.ndarray:
    """Approximate point downsampling by merging adjacent same-pen-state offsets.

    The paper reference describes aggressive random downsampling. Our data is already in
    offset-segment form, so we cannot safely drop rows outright. Instead, we randomly keep
    a subset of boundary points inside contiguous draw / non-draw runs and merge the
    skipped segments into longer offsets. This preserves the relative-motion representation
    while creating a clean paper-leaning augmentation.
    """
    if offsets.ndim != 2 or offsets.shape[1] < 3 or offsets.shape[0] <= 2:
        return offsets
    keep_min = float(np.clip(keep_min, 0.0, 1.0))
    keep_max = float(np.clip(keep_max, keep_min, 1.0))
    if keep_min >= 0.999 and keep_max >= 0.999:
        return offsets

    keep_ratio = float(np.random.uniform(keep_min, keep_max))
    draw_flag = np.clip(np.rint(offsets[:, 2]).astype(np.int64), 0, 1)
    runs: List[np.ndarray] = []
    run_start = 0
    for index in range(1, offsets.shape[0]):
        if int(draw_flag[index]) != int(draw_flag[run_start]):
            runs.append(offsets[run_start:index])
            run_start = index
    runs.append(offsets[run_start:])

    merged_runs = [_merge_offset_run(run, keep_ratio=keep_ratio) for run in runs]
    if not merged_runs:
        return offsets
    return np.concatenate(merged_runs, axis=0).astype(np.float32, copy=False)


def _merge_offset_run(run_offsets: np.ndarray, *, keep_ratio: float) -> np.ndarray:
    if run_offsets.shape[0] <= 2:
        return np.asarray(run_offsets, dtype=np.float32)

    boundary_count = run_offsets.shape[0] + 1
    keep_mask = np.zeros(boundary_count, dtype=bool)
    keep_mask[0] = True
    keep_mask[-1] = True
    if boundary_count > 2:
        keep_mask[1:-1] = np.random.random(boundary_count - 2) < keep_ratio

    kept_boundaries = np.flatnonzero(keep_mask)
    if kept_boundaries.size < 2:
        kept_boundaries = np.asarray([0, run_offsets.shape[0]], dtype=np.int64)

    draw_flag = float(np.clip(np.rint(run_offsets[0, 2]), 0, 1))
    merged_rows: List[List[float]] = []
    for start_boundary, end_boundary in zip(kept_boundaries[:-1], kept_boundaries[1:]):
        merged_slice = run_offsets[int(start_boundary):int(end_boundary)]
        if merged_slice.size == 0:
            continue
        merged_rows.append(
            [
                float(np.sum(merged_slice[:, 0])),
                float(np.sum(merged_slice[:, 1])),
                draw_flag,
            ]
        )
    return np.asarray(merged_rows, dtype=np.float32)


def collate_batch(batch: List[Dict[str, torch.Tensor]], text_pad_id: int, stroke_pad_id: int) -> Dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_text = max(item["text_ids"].shape[0] for item in batch)
    max_stroke = max(item["stroke_in_ids"].shape[0] for item in batch)

    writer_ids = torch.zeros((batch_size,), dtype=torch.long)
    style_cluster_ids = torch.zeros((batch_size,), dtype=torch.long)
    text_ids = torch.full((batch_size, max_text), text_pad_id, dtype=torch.long)
    stroke_in_ids = torch.full((batch_size, max_stroke), stroke_pad_id, dtype=torch.long)
    stroke_tgt_ids = torch.full((batch_size, max_stroke), stroke_pad_id, dtype=torch.long)
    sample_ids: List[str] = []
    writer_ids_raw: List[str] = []

    for index, item in enumerate(batch):
        writer_ids[index] = item["writer_ids"]
        style_cluster_ids[index] = item["style_cluster_ids"]
        text = item["text_ids"]
        stroke_in = item["stroke_in_ids"]
        stroke_tgt = item["stroke_tgt_ids"]
        text_ids[index, : text.shape[0]] = text
        stroke_in_ids[index, : stroke_in.shape[0]] = stroke_in
        stroke_tgt_ids[index, : stroke_tgt.shape[0]] = stroke_tgt
        sample_ids.append(str(item["sample_id"]))
        writer_ids_raw.append(str(item["writer_id_raw"]))

    return {
        "sample_ids": sample_ids,
        "writer_ids_raw": writer_ids_raw,
        "writer_ids": writer_ids,
        "style_cluster_ids": style_cluster_ids,
        "text_ids": text_ids,
        "stroke_in_ids": stroke_in_ids,
        "stroke_tgt_ids": stroke_tgt_ids,
        "text_pad_mask": text_ids.eq(text_pad_id),
        "stroke_pad_mask": stroke_in_ids.eq(stroke_pad_id),
    }
