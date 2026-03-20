from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from .utils import load_json_dict


UNK_WRITER_TOKEN = "<unk_writer>"


@dataclass(frozen=True)
class WriterVocab:
    writer_to_index: Dict[str, int]
    index_to_writer: List[str]
    train_writer_ids: List[str]
    unknown_writer_id: str
    unknown_writer_index: int
    split_stats: Dict[str, dict]
    train_writer_counts: Dict[str, int]

    @property
    def num_embeddings(self) -> int:
        return len(self.index_to_writer)

    @property
    def num_train_writers(self) -> int:
        return len(self.train_writer_ids)

    def encode(self, raw_writer_id: str, *, allow_unseen: bool) -> int:
        writer_id = str(raw_writer_id)
        if writer_id in self.writer_to_index:
            return int(self.writer_to_index[writer_id])
        if allow_unseen:
            return int(self.unknown_writer_index)
        raise ValueError(f"Unseen writer_id={writer_id}")

    def to_dict(self) -> dict:
        return {
            "unknown_writer_id": self.unknown_writer_id,
            "unknown_writer_index": self.unknown_writer_index,
            "writer_to_index": self.writer_to_index,
            "index_to_writer": self.index_to_writer,
            "train_writer_ids": self.train_writer_ids,
            "train_writer_counts": self.train_writer_counts,
            "split_stats": self.split_stats,
            "num_writer_embeddings": self.num_embeddings,
            "train_writer_count": self.num_train_writers,
        }


def _normalize_writer_ids(writer_ids: Sequence[object]) -> List[str]:
    return [str(value) for value in writer_ids]


def build_writer_artifact_payloads(
    *,
    split_to_writer_ids: Mapping[str, Sequence[object]],
    unknown_writer_id: str = UNK_WRITER_TOKEN,
    representative_count: int = 3,
) -> Tuple[dict, dict]:
    if "train" not in split_to_writer_ids:
        raise ValueError("Writer artifact build requires a train split")

    train_writer_ids = _normalize_writer_ids(split_to_writer_ids["train"])
    train_counts = Counter(train_writer_ids)
    ordered_train_writers = sorted(train_counts)
    writer_to_index = {unknown_writer_id: 0}
    for writer_id in ordered_train_writers:
        writer_to_index[writer_id] = len(writer_to_index)
    index_to_writer = [None] * len(writer_to_index)
    for writer_id, index in writer_to_index.items():
        index_to_writer[int(index)] = writer_id

    train_writer_set = set(ordered_train_writers)
    split_stats: Dict[str, dict] = {}
    for split_name, raw_ids in split_to_writer_ids.items():
        normalized = _normalize_writer_ids(raw_ids)
        writer_set = set(normalized)
        unseen = sorted(writer_set - train_writer_set)
        split_stats[str(split_name)] = {
            "sample_count": int(len(normalized)),
            "writer_count": int(len(writer_set)),
            "seen_writer_count": int(len(writer_set & train_writer_set)),
            "unseen_writer_count": int(len(unseen)),
            "unseen_writer_ids": unseen,
        }

    writer_map_payload = {
        "name": "paper_local_writer_id_map",
        "unknown_writer_id": unknown_writer_id,
        "unknown_writer_index": 0,
        "num_writer_embeddings": int(len(writer_to_index)),
        "train_writer_count": int(len(ordered_train_writers)),
        "writer_to_index": writer_to_index,
        "index_to_writer": index_to_writer,
        "train_writer_ids": ordered_train_writers,
        "train_writer_counts": {writer_id: int(count) for writer_id, count in sorted(train_counts.items())},
        "writers": [
            {
                "writer_id": writer_id,
                "writer_index": int(writer_to_index[writer_id]),
                "train_count": int(train_counts[writer_id]),
            }
            for writer_id in ordered_train_writers
        ],
        "split_stats": split_stats,
    }

    representative_count = max(1, int(representative_count))
    representative_writers = [
        {
            "writer_id": writer_id,
            "writer_index": int(writer_to_index[writer_id]),
            "train_count": int(count),
        }
        for writer_id, count in sorted(train_counts.items(), key=lambda item: (-item[1], item[0]))[:representative_count]
    ]
    panel_payload = {
        "name": "default_panel_writers",
        "source_split": "train",
        "selection": "top_train_counts",
        "writers": representative_writers,
    }
    return writer_map_payload, panel_payload


def load_writer_vocab(path: Path) -> WriterVocab:
    payload = load_json_dict(path)
    writer_to_index = {str(key): int(value) for key, value in payload["writer_to_index"].items()}
    index_to_writer = [str(value) for value in payload["index_to_writer"]]
    train_writer_ids = [str(value) for value in payload.get("train_writer_ids", [])]
    train_writer_counts = {
        str(key): int(value) for key, value in payload.get("train_writer_counts", {}).items()
    }
    split_stats = payload.get("split_stats", {})
    return WriterVocab(
        writer_to_index=writer_to_index,
        index_to_writer=index_to_writer,
        train_writer_ids=train_writer_ids,
        unknown_writer_id=str(payload.get("unknown_writer_id", UNK_WRITER_TOKEN)),
        unknown_writer_index=int(payload.get("unknown_writer_index", 0)),
        split_stats={str(key): dict(value) for key, value in split_stats.items()},
        train_writer_counts=train_writer_counts,
    )


def load_panel_writers(path: Path, writer_vocab: WriterVocab) -> List[dict]:
    payload = load_json_dict(path)
    writers = payload.get("writers", [])
    if not isinstance(writers, list) or not writers:
        raise ValueError(f"Expected a non-empty writers list in {path}")
    resolved: List[dict] = []
    for item in writers:
        if not isinstance(item, dict):
            raise ValueError(f"Panel writer entries must be objects in {path}")
        raw_writer_id = str(item.get("writer_id") or "").strip()
        if not raw_writer_id:
            raise ValueError(f"Panel writer entry missing writer_id in {path}")
        if raw_writer_id not in writer_vocab.writer_to_index:
            raise ValueError(f"Panel writer_id={raw_writer_id} is not present in the train writer map")
        resolved.append(
            {
                "writer_id": raw_writer_id,
                "writer_index": int(writer_vocab.writer_to_index[raw_writer_id]),
                "train_count": int(item.get("train_count", writer_vocab.train_writer_counts.get(raw_writer_id, 0))),
                "label": f"writer_{raw_writer_id}",
            }
        )
    return resolved
