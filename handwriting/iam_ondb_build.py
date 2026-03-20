from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from xml.etree import ElementTree as ET

import numpy as np

from .dot_cross_audit import run_dot_cross_timing_audit
from .dot_cross_operational import build_operational_rows, render_operational_summary_markdown, summarize_operational_rows


SOURCE_NAME = "iamondb"
SOURCE_LABEL = "IAM-OnDB"
OPERATIONAL_RULE_DESCRIPTION = (
    "immediate=>local, nearby=>delayed, delayed=>delayed, ambiguous=>unresolved"
)


@dataclass
class CanonicalSample:
    sample_id: str
    source: str
    writer_id: str
    raw_user_id: Optional[str]
    word_form: str
    text: str
    points_abs: np.ndarray
    timestamps: np.ndarray
    events: np.ndarray
    word_segments: List[dict]
    char_segments: List[dict]
    json_path: str
    sample_key: str
    source_file_id: str
    is_word_segmentation_valid: Optional[bool]
    is_char_segmentation_valid: Optional[bool]
    is_sentence_misspelled: Optional[bool]
    misspelled_words_idx: List[int]
    split: Optional[str] = None


@dataclass
class DerivedWordExample:
    sample_id: str
    canonical_sample_id: str
    source: str
    writer_id: str
    text: str
    offsets: np.ndarray
    word_index: int
    point_count: int
    draw_segment_count: int
    relocation_count: int
    split: Optional[str] = None


def build_parser(bundle_root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the IAM-OnDB all-writers local-only dataset bundle from the available local sources."
    )
    parser.add_argument("--data-root", type=Path, default=bundle_root.parent / "DATA")
    parser.add_argument("--trajectory-json-root", type=Path, default=None)
    parser.add_argument("--out-bundle-root", type=Path, default=bundle_root)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--audit-max-examples-per-category", type=int, default=0)
    parser.add_argument("--audit-limit-samples", type=int, default=None)
    return parser


def default_trajectory_json_root(bundle_root: Path) -> Path:
    candidates = [
        bundle_root.parent / "Version1A_local_only" / "extended_dataset" / "Iamondb Dataset",
        bundle_root.parent / "Version1A" / "extended_dataset" / "Iamondb Dataset",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find the IAM trajectory JSON export. "
        "Expected something like Version1A_local_only/extended_dataset/Iamondb Dataset."
    )


def normalize_boolish(value) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def normalize_text(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def normalize_int_list(values) -> List[int]:
    if not isinstance(values, list):
        return []
    out: List[int] = []
    for value in values:
        try:
            out.append(int(value))
        except Exception:
            continue
    return out


def normalize_ranges(raw_ranges) -> List[List[int]]:
    if not isinstance(raw_ranges, list):
        return []
    ranges: List[List[int]] = []
    for group in raw_ranges:
        if not isinstance(group, list):
            continue
        values = normalize_int_list(group)
        if values:
            ranges.append(values)
    return ranges


def make_sample_id(writer_id: str, word_form: str, file_stem: str, sample_key: str) -> str:
    return f"{SOURCE_NAME}:{writer_id}:{word_form}:{file_stem}:{sample_key}"


def load_points(sample: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    raw_points = sample.get("word_stroke")
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        return None, None, None, "no_points"

    rows: List[List[float]] = []
    for point in raw_points:
        if not isinstance(point, dict):
            return None, None, None, "bad_point_entry"
        try:
            rows.append(
                [
                    float(point["x"]),
                    float(point["y"]),
                    float(point["ts"]),
                    float(point["ev"]),
                ]
            )
        except Exception:
            return None, None, None, "point_parse_error"

    data = np.asarray(rows, dtype=np.float64)
    if data.shape[0] < 2:
        return None, None, None, "too_few_points"

    points_abs = data[:, :2].astype(np.float32)
    timestamps = data[:, 2].astype(np.float64)
    events = np.clip(np.rint(data[:, 3]), 0, 2).astype(np.uint8)
    return points_abs, timestamps, events, None


def normalize_segment_text(word_entry: dict) -> Optional[str]:
    for key in ("recognized_label", "ocr_label"):
        text = normalize_text(word_entry.get(key))
        if text:
            return text
    chars = word_entry.get("chars") or []
    letters = [normalize_text(char.get("char")) for char in chars if isinstance(char, dict)]
    merged = "".join(letter for letter in letters if letter)
    return merged or None


def normalize_word_and_char_segments(sample: dict) -> Tuple[List[dict], List[dict]]:
    word_segments: List[dict] = []
    char_segments: List[dict] = []
    raw_words = sample.get("wholeword_segments")
    if not isinstance(raw_words, list):
        return word_segments, char_segments

    for word_index, raw_word in enumerate(raw_words):
        if not isinstance(raw_word, dict):
            continue
        word_ranges = normalize_ranges(raw_word.get("ranges"))
        raw_chars = raw_word.get("chars")
        chars: List[dict] = []
        if isinstance(raw_chars, list):
            for char_index, raw_char in enumerate(raw_chars):
                if not isinstance(raw_char, dict):
                    continue
                char_entry = {
                    "word_index": word_index,
                    "char_index": char_index,
                    "char": normalize_text(raw_char.get("char")),
                    "recognition_is_correct": normalize_boolish(raw_char.get("recognition_is_correct")),
                    "ranges": normalize_ranges(raw_char.get("ranges")),
                    "char_image_path": normalize_text(raw_char.get("char_image_path")),
                }
                chars.append(char_entry)
                char_segments.append(char_entry)

        word_entry = {
            "word_index": word_index,
            "text": normalize_segment_text(raw_word),
            "recognized_label": normalize_text(raw_word.get("recognized_label")),
            "ocr_label": normalize_text(raw_word.get("ocr_label")),
            "recognition_is_correct": normalize_boolish(raw_word.get("recognition_is_correct")),
            "ranges": word_ranges,
            "chars": chars,
            "word_image_path": normalize_text(raw_word.get("word_image_path")),
        }
        word_segments.append(word_entry)

    return word_segments, char_segments


def sample_to_canonical(path: Path, sample_key: str, sample: dict) -> Tuple[Optional[CanonicalSample], Optional[str]]:
    text = normalize_text(sample.get("word_ascii"))
    if not text:
        return None, "no_text"

    points_abs, timestamps, events, point_error = load_points(sample)
    if point_error is not None:
        return None, point_error

    writer_id = path.parents[1].name
    raw_user_id = normalize_text(sample.get("user_id"))
    word_form = normalize_text(sample.get("word_form")) or path.parent.name
    word_segments, char_segments = normalize_word_and_char_segments(sample)

    canonical = CanonicalSample(
        sample_id=make_sample_id(writer_id, word_form, path.stem, sample_key),
        source=SOURCE_NAME,
        writer_id=writer_id,
        raw_user_id=raw_user_id,
        word_form=word_form,
        text=text,
        points_abs=points_abs,
        timestamps=timestamps,
        events=events,
        word_segments=word_segments,
        char_segments=char_segments,
        json_path=path.as_posix(),
        sample_key=sample_key,
        source_file_id=path.stem,
        is_word_segmentation_valid=normalize_boolish(sample.get("is_word_segmentation_valid")),
        is_char_segmentation_valid=normalize_boolish(sample.get("is_char_segmentation_valid")),
        is_sentence_misspelled=normalize_boolish(sample.get("is_sentence_misspelled")),
        misspelled_words_idx=normalize_int_list(sample.get("misspelled_words_idx")),
    )
    return canonical, None


def extract_word_point_sequence(
    sample: CanonicalSample,
    word_segment: dict,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    if not word_segment.get("ranges"):
        return None, None, None, None, "no_ranges"

    point_count = sample.points_abs.shape[0]
    indices: List[int] = []
    for group in word_segment["ranges"]:
        if any(index < 0 or index >= point_count for index in group):
            return None, None, None, None, "range_out_of_bounds"
        indices.extend(int(index) for index in group)

    if len(indices) < 2:
        return None, None, None, None, "word_too_few_points"

    index_array = np.asarray(indices, dtype=np.int64)
    return (
        sample.points_abs[index_array],
        sample.timestamps[index_array],
        sample.events[index_array],
        index_array,
        None,
    )


def events_to_draw_flags(events: np.ndarray) -> np.ndarray:
    if events.shape[0] < 2:
        return np.zeros((0,), dtype=np.float32)
    return np.isin(events[1:].astype(np.int64), [1, 2]).astype(np.float32)


def points_to_offsets(points_abs: np.ndarray, events: np.ndarray) -> np.ndarray:
    if points_abs.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    diffs = points_abs[1:] - points_abs[:-1]
    draw_flags = events_to_draw_flags(events).reshape(-1, 1)
    return np.concatenate([diffs.astype(np.float32), draw_flags], axis=1)


def canonical_to_word_examples(sample: CanonicalSample, invalid_counts: Counter) -> List[DerivedWordExample]:
    examples: List[DerivedWordExample] = []
    if sample.is_word_segmentation_valid is not True:
        invalid_counts["sample_invalid_word_segmentation"] += 1
        return examples
    if not sample.word_segments:
        invalid_counts["sample_missing_word_segments"] += 1
        return examples
    if sample.is_sentence_misspelled is True and not sample.misspelled_words_idx:
        invalid_counts["sample_misspelled_without_word_indices"] += 1
        return examples

    misspelled_word_indices = set(sample.misspelled_words_idx)
    for word_segment in sample.word_segments:
        word_index = int(word_segment["word_index"])
        if word_index in misspelled_word_indices:
            invalid_counts["word_marked_misspelled"] += 1
            continue
        if word_segment.get("recognition_is_correct") is False:
            invalid_counts["word_recognition_incorrect"] += 1
            continue

        text = normalize_text(word_segment.get("text"))
        if not text:
            invalid_counts["word_empty_text"] += 1
            continue

        word_points, _, word_events, _, word_error = extract_word_point_sequence(sample, word_segment)
        if word_error is not None or word_points is None or word_events is None:
            invalid_counts[word_error or "word_extract_error"] += 1
            continue

        offsets = points_to_offsets(word_points, word_events)
        if offsets.shape[0] == 0:
            invalid_counts["word_no_offsets"] += 1
            continue

        draw_flags = offsets[:, 2]
        draw_segment_count = int(np.count_nonzero(draw_flags >= 0.5))
        relocation_count = int(offsets.shape[0] - draw_segment_count)
        examples.append(
            DerivedWordExample(
                sample_id=f"{sample.sample_id}:word{word_index}",
                canonical_sample_id=sample.sample_id,
                source=sample.source,
                writer_id=sample.writer_id,
                text=text,
                offsets=offsets,
                word_index=word_index,
                point_count=int(word_points.shape[0]),
                draw_segment_count=draw_segment_count,
                relocation_count=relocation_count,
            )
        )
    return examples


def allocate_split_counts(count: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Dict[str, int]:
    if count <= 1:
        return {"train": count, "val": 0, "test": 0}
    if count == 2:
        return {"train": 1, "val": 1, "test": 0}
    if count == 3:
        return {"train": 1, "val": 1, "test": 1}

    raw = {
        "train": count * train_ratio,
        "val": count * val_ratio,
        "test": count * test_ratio,
    }
    counts = {split: int(math.floor(value)) for split, value in raw.items()}
    remainder = count - sum(counts.values())
    split_order = ["train", "val", "test"]
    ranked = sorted(split_order, key=lambda split: (raw[split] - counts[split], -split_order.index(split)), reverse=True)
    for split in ranked:
        if remainder <= 0:
            break
        counts[split] += 1
        remainder -= 1

    for split in ("val", "test"):
        if counts[split] == 0:
            donor = "train" if counts["train"] > 1 else ("val" if split == "test" and counts["val"] > 1 else None)
            if donor is not None:
                counts[donor] -= 1
                counts[split] += 1

    if counts["train"] == 0:
        donor = "val" if counts["val"] >= counts["test"] else "test"
        counts[donor] -= 1
        counts["train"] += 1

    delta = count - sum(counts.values())
    counts["train"] += delta
    return counts


def assign_sample_level_splits(
    canonical_samples: Sequence[CanonicalSample],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, str]:
    rng = np.random.default_rng(seed)
    writer_to_samples: Dict[str, List[CanonicalSample]] = defaultdict(list)
    for sample in canonical_samples:
        writer_to_samples[str(sample.writer_id)].append(sample)

    mapping: Dict[str, str] = {}
    for writer_id in sorted(writer_to_samples):
        samples = sorted(writer_to_samples[writer_id], key=lambda item: item.sample_id)
        permuted_indices = rng.permutation(len(samples))
        shuffled = [samples[int(index)] for index in permuted_indices]
        counts = allocate_split_counts(len(shuffled), train_ratio, val_ratio, test_ratio)
        start = 0
        for split in ("train", "val", "test"):
            end = start + counts[split]
            for sample in shuffled[start:end]:
                mapping[sample.sample_id] = split
            start = end
    return mapping


def save_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def save_canonical_outputs(canonical_samples: Sequence[CanonicalSample], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "canonical_samples.npz",
        sample_id=np.array([sample.sample_id for sample in canonical_samples], dtype=object),
        source=np.array([sample.source for sample in canonical_samples], dtype=object),
        writer_id=np.array([sample.writer_id for sample in canonical_samples], dtype=object),
        raw_user_id=np.array([sample.raw_user_id for sample in canonical_samples], dtype=object),
        word_form=np.array([sample.word_form for sample in canonical_samples], dtype=object),
        text=np.array([sample.text for sample in canonical_samples], dtype=object),
        split=np.array([sample.split for sample in canonical_samples], dtype=object),
        points_abs=np.array([sample.points_abs for sample in canonical_samples], dtype=object),
        timestamps=np.array([sample.timestamps for sample in canonical_samples], dtype=object),
        events=np.array([sample.events for sample in canonical_samples], dtype=object),
    )
    rows = []
    for index, sample in enumerate(canonical_samples):
        rows.append(
            {
                "index": index,
                "sample_id": sample.sample_id,
                "source": sample.source,
                "writer_id": sample.writer_id,
                "raw_user_id": sample.raw_user_id,
                "word_form": sample.word_form,
                "text": sample.text,
                "split": sample.split,
                "json_path": sample.json_path,
                "sample_key": sample.sample_key,
                "source_file_id": sample.source_file_id,
                "point_count": int(sample.points_abs.shape[0]),
                "is_word_segmentation_valid": sample.is_word_segmentation_valid,
                "is_char_segmentation_valid": sample.is_char_segmentation_valid,
                "is_sentence_misspelled": sample.is_sentence_misspelled,
                "misspelled_words_idx": sample.misspelled_words_idx,
                "word_segments": sample.word_segments,
                "char_segments": sample.char_segments,
            }
        )
    save_jsonl(out_dir / "canonical_samples.jsonl", rows)


def save_derived_outputs(examples: Sequence[DerivedWordExample], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_subset(path_prefix: Path, subset: Sequence[DerivedWordExample]) -> None:
        np.savez_compressed(
            path_prefix.with_suffix(".npz"),
            sample_id=np.array([example.sample_id for example in subset], dtype=object),
            canonical_sample_id=np.array([example.canonical_sample_id for example in subset], dtype=object),
            source=np.array([example.source for example in subset], dtype=object),
            writer_id=np.array([example.writer_id for example in subset], dtype=object),
            text=np.array([example.text for example in subset], dtype=object),
            split=np.array([example.split for example in subset], dtype=object),
            word_index=np.array([example.word_index for example in subset], dtype=np.int32),
            point_count=np.array([example.point_count for example in subset], dtype=np.int32),
            draw_segment_count=np.array([example.draw_segment_count for example in subset], dtype=np.int32),
            relocation_count=np.array([example.relocation_count for example in subset], dtype=np.int32),
            offsets=np.array([example.offsets for example in subset], dtype=object),
        )
        manifest_rows = []
        for index, example in enumerate(subset):
            manifest_rows.append(
                {
                    "index": index,
                    "sample_id": example.sample_id,
                    "canonical_sample_id": example.canonical_sample_id,
                    "source": example.source,
                    "writer_id": example.writer_id,
                    "text": example.text,
                    "split": example.split,
                    "word_index": example.word_index,
                    "point_count": example.point_count,
                    "num_segments": int(example.offsets.shape[0]),
                    "draw_segment_count": example.draw_segment_count,
                    "relocation_count": example.relocation_count,
                }
            )
        save_jsonl(path_prefix.with_suffix(".jsonl"), manifest_rows)

    save_subset(out_dir / "all_examples", examples)
    for split in ("train", "val", "test"):
        save_subset(out_dir / split, [example for example in examples if example.split == split])


def inspect_data_xml_folder(data_root: Path) -> dict:
    xml_paths = sorted(data_root.rglob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No XML files found under {data_root}")

    tag_counts = Counter()
    point_parent_counts = Counter()
    point_attribute_keys = Counter()
    line_count = 0
    word_count = 0
    cmp_count = 0
    contour_point_counts: List[int] = []
    example = {}

    for path in xml_paths:
        root = ET.parse(path).getroot()
        if not example:
            handwritten = root.find("handwritten-part")
            first_line = handwritten.find("line") if handwritten is not None else None
            first_word = first_line.find("word") if first_line is not None else None
            example = {
                "path": path.as_posix(),
                "root_tag": root.tag,
                "root_attributes": dict(root.attrib),
                "first_line_attributes": dict(first_line.attrib) if first_line is not None else None,
                "first_word_attributes": dict(first_word.attrib) if first_word is not None else None,
            }
        for elem in root.iter():
            tag_counts[elem.tag] += 1
        handwritten = root.find("handwritten-part")
        if handwritten is not None:
            for line in handwritten.findall("line"):
                line_count += 1
                word_count += len(line.findall("word"))
                cmp_count += len(line.findall(".//cmp"))
                for contour_tag in ("upper-contour", "lower-contour"):
                    contour = line.find(contour_tag)
                    if contour is None:
                        continue
                    points = contour.findall("point")
                    contour_point_counts.append(len(points))
                    point_parent_counts[contour_tag] += len(points)
                    for point in points:
                        for key in point.attrib:
                            point_attribute_keys[key] += 1

    trajectory_like_tags = sorted(
        tag
        for tag in tag_counts
        if tag.lower() in {"strokeset", "stroke", "trace", "channel", "sample", "whiteboarddescription"}
    )
    trajectory_like_attrs = sorted(key for key in point_attribute_keys if key.lower() in {"ts", "ev", "time", "pressure"})
    return {
        "data_root": data_root.as_posix(),
        "xml_file_count": int(len(xml_paths)),
        "tag_counts": dict(sorted(tag_counts.items())),
        "line_count": int(line_count),
        "word_count": int(word_count),
        "cmp_count": int(cmp_count),
        "point_parent_counts": dict(sorted(point_parent_counts.items())),
        "point_attribute_keys": dict(sorted(point_attribute_keys.items())),
        "contour_point_count_stats": {
            "min": int(min(contour_point_counts)) if contour_point_counts else None,
            "max": int(max(contour_point_counts)) if contour_point_counts else None,
            "mean": float(np.mean(contour_point_counts)) if contour_point_counts else None,
        },
        "trajectory_like_tags_found": trajectory_like_tags,
        "trajectory_like_point_attributes_found": trajectory_like_attrs,
        "contains_online_trajectory_points": bool(trajectory_like_tags or trajectory_like_attrs),
        "example": example,
        "interpretation": {
            "verdict": "DATA xmls are offline form/segmentation metadata, not online trajectory files",
            "reason": (
                "The XMLs contain form/line/word/cmp geometry plus upper/lower contour points only. "
                "They do not contain stroke-set style trajectory tags or point timestamps/pen events."
            ),
        },
    }


def filter_json_source_files(root: Path) -> List[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def build_canonical_and_word_level(
    *,
    trajectory_json_root: Path,
    canonical_out_dir: Path,
    word_level_out_dir: Path,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict:
    json_files = filter_json_source_files(trajectory_json_root)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {trajectory_json_root}")

    canonical_samples: List[CanonicalSample] = []
    derived_examples: List[DerivedWordExample] = []
    canonical_invalid_counts: Counter = Counter()
    derived_invalid_counts: Counter = Counter()
    sample_entries_seen = 0

    for file_index, path in enumerate(json_files, start=1):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            canonical_invalid_counts["file_not_dict"] += 1
            continue
        for sample_key, raw_sample in data.items():
            sample_entries_seen += 1
            if not isinstance(raw_sample, dict):
                canonical_invalid_counts["sample_not_dict"] += 1
                continue
            canonical, error = sample_to_canonical(path, sample_key, raw_sample)
            if canonical is None or error is not None:
                canonical_invalid_counts[error or "unknown_sample_error"] += 1
                continue
            canonical_samples.append(canonical)
            derived_examples.extend(canonical_to_word_examples(canonical, derived_invalid_counts))

        if file_index % 200 == 0 or file_index == len(json_files):
            print(
                f"[iam-build] files={file_index}/{len(json_files)} "
                f"sample_entries={sample_entries_seen} canonical={len(canonical_samples)} "
                f"derived_words={len(derived_examples)}"
            )

    split_by_sample_id = assign_sample_level_splits(
        canonical_samples,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    canonical_by_id = {sample.sample_id: sample for sample in canonical_samples}
    for sample in canonical_samples:
        sample.split = split_by_sample_id[sample.sample_id]
    for example in derived_examples:
        example.split = canonical_by_id[example.canonical_sample_id].split

    save_canonical_outputs(canonical_samples, canonical_out_dir)
    save_derived_outputs(derived_examples, word_level_out_dir)

    canonical_split_sizes = Counter(sample.split for sample in canonical_samples)
    derived_split_sizes = Counter(example.split for example in derived_examples)
    canonical_writer_coverage = {
        split: len({sample.writer_id for sample in canonical_samples if sample.split == split})
        for split in ("train", "val", "test")
    }
    derived_writer_coverage = {
        split: len({example.writer_id for example in derived_examples if example.split == split})
        for split in ("train", "val", "test")
    }

    return {
        "trajectory_json_root": trajectory_json_root.as_posix(),
        "trajectory_json_file_count": int(len(json_files)),
        "sample_entries_seen": int(sample_entries_seen),
        "canonical_samples_total": int(len(canonical_samples)),
        "canonical_invalid_counts": {str(key): int(value) for key, value in sorted(canonical_invalid_counts.items())},
        "canonical_char_valid_count": int(sum(1 for sample in canonical_samples if sample.is_char_segmentation_valid is True)),
        "canonical_word_valid_count": int(sum(1 for sample in canonical_samples if sample.is_word_segmentation_valid is True)),
        "canonical_split_sizes": {str(key): int(value) for key, value in sorted(canonical_split_sizes.items())},
        "canonical_writer_coverage": canonical_writer_coverage,
        "derived_word_examples_total": int(len(derived_examples)),
        "derived_invalid_counts": {str(key): int(value) for key, value in sorted(derived_invalid_counts.items())},
        "derived_split_sizes": {str(key): int(value) for key, value in sorted(derived_split_sizes.items())},
        "derived_writer_coverage": derived_writer_coverage,
        "split_policy": {
            "type": "writer_stratified_sample_level_split",
            "description": (
                "Canonical samples are shuffled within each writer with a fixed seed, then split approximately "
                "80/10/10 while keeping small-writer edge cases sane. Derived word examples inherit the parent sample split."
            ),
            "seed": int(seed),
            "ratios": {"train": float(train_ratio), "val": float(val_ratio), "test": float(test_ratio)},
        },
        "canonical_out_dir": canonical_out_dir.as_posix(),
        "word_level_out_dir": word_level_out_dir.as_posix(),
    }


def write_reports_json_md(reports_dir: Path, stem: str, payload: dict, markdown: str) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    (reports_dir / f"{stem}.md").write_text(markdown)


def render_dataset_build_summary_md(summary: dict) -> str:
    xml_info = summary["data_xml_inspection"]
    canonical = summary["canonical_build"]
    lines = [
        "# IAM-OnDB Dataset Build Summary",
        "",
        "## Raw DATA Inspection",
        "",
        f"- DATA root: `{xml_info['data_root']}`",
        f"- XML files found: `{xml_info['xml_file_count']}`",
        f"- Contains online trajectory points: `{xml_info['contains_online_trajectory_points']}`",
        f"- Interpretation: `{xml_info['interpretation']['verdict']}`",
        f"- Reason: `{xml_info['interpretation']['reason']}`",
        "",
        "## Trajectory Source Used",
        "",
        f"- Source used for canonical trajectory build: `{canonical['trajectory_json_root']}`",
        f"- JSON files used: `{canonical['trajectory_json_file_count']}`",
        f"- Sample entries scanned: `{canonical['sample_entries_seen']}`",
        "",
        "## Canonical Build",
        "",
        f"- Canonical samples built: `{canonical['canonical_samples_total']}`",
        f"- Char-valid canonical samples: `{canonical['canonical_char_valid_count']}`",
        f"- Word-valid canonical samples: `{canonical['canonical_word_valid_count']}`",
        f"- Canonical split sizes: `{canonical['canonical_split_sizes']}`",
        f"- Canonical writer coverage: `{canonical['canonical_writer_coverage']}`",
        f"- Canonical invalid counts: `{canonical['canonical_invalid_counts']}`",
        "",
        "## Word-Level Build",
        "",
        f"- Derived word examples: `{canonical['derived_word_examples_total']}`",
        f"- Derived split sizes: `{canonical['derived_split_sizes']}`",
        f"- Derived writer coverage: `{canonical['derived_writer_coverage']}`",
        f"- Derived invalid counts: `{canonical['derived_invalid_counts']}`",
        "",
        "## Split Policy",
        "",
        f"- Type: `{canonical['split_policy']['type']}`",
        f"- Description: `{canonical['split_policy']['description']}`",
        f"- Seed: `{canonical['split_policy']['seed']}`",
        "",
    ]
    return "\n".join(lines)


def write_jsonl_rows(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_operational_status_map(path: Path) -> Dict[Tuple[str, int], dict]:
    grouped: Dict[Tuple[str, int], dict] = defaultdict(
        lambda: {
            "local": 0,
            "delayed": 0,
            "ambiguous": 0,
            "chars": [],
            "raw_categories": [],
        }
    )
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        key = (str(row["sample_id"]), int(row["word_index"]))
        op_label = str(row["operational_label"])
        grouped[key][op_label] += 1
        grouped[key]["chars"].append(str(row["char"]))
        grouped[key]["raw_categories"].append(str(row["category"]))
    return dict(grouped)


def decide_word_reason(status: Optional[dict]) -> Tuple[bool, str]:
    if status is None:
        return True, "keep_no_audited_ijt_occurrences"
    has_delayed = int(status.get("delayed", 0)) > 0
    has_ambiguous = int(status.get("ambiguous", 0)) > 0
    if has_delayed and has_ambiguous:
        return False, "drop_contains_delayed_and_ambiguous_occurrence"
    if has_delayed:
        return False, "drop_contains_delayed_occurrence"
    if has_ambiguous:
        return False, "drop_contains_ambiguous_occurrence"
    return True, "keep_all_audited_ijt_occurrences_local"


def filter_split(npz_path: Path, status_map: Dict[Tuple[str, int], dict], out_path: Path) -> dict:
    npz = np.load(npz_path, allow_pickle=True)
    required = ("sample_id", "canonical_sample_id", "word_index", "text", "split")
    for key in required:
        if key not in npz:
            raise KeyError(f"Missing '{key}' in {npz_path}")

    sample_ids = np.asarray(npz["sample_id"], dtype=object)
    canonical_ids = np.asarray(npz["canonical_sample_id"], dtype=object)
    word_indices = np.asarray(npz["word_index"], dtype=np.int64)
    texts = np.asarray(npz["text"], dtype=object)
    split_names = np.asarray(npz["split"], dtype=object)
    keep_mask = np.zeros(sample_ids.shape[0], dtype=bool)
    manifest_rows = []
    reason_counts = Counter()

    for index in range(sample_ids.shape[0]):
        key = (str(canonical_ids[index]), int(word_indices[index]))
        status = status_map.get(key)
        keep, reason = decide_word_reason(status)
        keep_mask[index] = keep
        reason_counts[reason] += 1
        manifest_rows.append(
            {
                "sample_id": str(sample_ids[index]),
                "canonical_sample_id": str(canonical_ids[index]),
                "word_index": int(word_indices[index]),
                "text": str(texts[index]),
                "split": str(split_names[index]),
                "decision": "keep" if keep else "drop",
                "reason": reason,
                "local_occurrence_count": int((status or {}).get("local", 0)),
                "delayed_occurrence_count": int((status or {}).get("delayed", 0)),
                "ambiguous_occurrence_count": int((status or {}).get("ambiguous", 0)),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_arrays = {key: npz[key][keep_mask] for key in npz.files}
    np.savez_compressed(out_path, **filtered_arrays)
    npz.close()

    kept = int(np.count_nonzero(keep_mask))
    original = int(sample_ids.shape[0])
    dropped = int(original - kept)
    return {
        "split": str(split_names[0]) if split_names.size else out_path.stem,
        "source_path": npz_path.as_posix(),
        "out_path": out_path.as_posix(),
        "original_count": original,
        "kept_count": kept,
        "dropped_count": dropped,
        "kept_fraction": float(kept / original) if original else None,
        "reason_counts": {str(key): int(value) for key, value in sorted(reason_counts.items())},
        "manifest_rows": manifest_rows,
    }


def render_filter_summary_md(summary: dict) -> str:
    lines = [
        "# IAM Operational Local-Only Filter Report",
        "",
        "Filtering rule:",
        "",
        "- keep words with no audited `i/j/t` occurrence",
        "- keep words whose audited `i/j/t` occurrences are all operational `local`",
        "- drop words containing operational `delayed` occurrences",
        "- drop words containing operational `ambiguous` occurrences",
        "",
        "## Overall",
        "",
        f"- Original words: `{summary['overall']['original_count']}`",
        f"- Kept words: `{summary['overall']['kept_count']}`",
        f"- Dropped words: `{summary['overall']['dropped_count']}`",
        f"- Kept fraction: `{summary['overall']['kept_fraction']}`",
        f"- Reason counts: `{summary['overall']['reason_counts']}`",
        "",
    ]
    for split_name in ("train", "val", "test"):
        split_summary = summary["by_split"][split_name]
        lines.extend(
            [
                f"## `{split_name}`",
                "",
                f"- Original words: `{split_summary['original_count']}`",
                f"- Kept words: `{split_summary['kept_count']}`",
                f"- Dropped words: `{split_summary['dropped_count']}`",
                f"- Kept fraction: `{split_summary['kept_fraction']}`",
                f"- Reason counts: `{split_summary['reason_counts']}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_operational_filter_outputs(
    *,
    word_level_dir: Path,
    occurrences_with_operational_label_path: Path,
    out_dir: Path,
) -> dict:
    status_map = load_operational_status_map(occurrences_with_operational_label_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_reports = {}
    all_manifest_rows: List[dict] = []
    for split_name in ("train", "val", "test"):
        report = filter_split(word_level_dir / f"{split_name}.npz", status_map, out_dir / f"{split_name}.npz")
        split_reports[split_name] = {key: value for key, value in report.items() if key != "manifest_rows"}
        all_manifest_rows.extend(report["manifest_rows"])

    save_jsonl(out_dir / "filter_manifest.jsonl", all_manifest_rows)
    overall_original = int(sum(report["original_count"] for report in split_reports.values()))
    overall_kept = int(sum(report["kept_count"] for report in split_reports.values()))
    overall_dropped = int(sum(report["dropped_count"] for report in split_reports.values()))
    overall_reasons = Counter()
    for row in all_manifest_rows:
        overall_reasons[str(row["reason"])] += 1

    summary = {
        "rule": OPERATIONAL_RULE_DESCRIPTION,
        "occurrences_with_operational_label_path": occurrences_with_operational_label_path.as_posix(),
        "overall": {
            "original_count": overall_original,
            "kept_count": overall_kept,
            "dropped_count": overall_dropped,
            "kept_fraction": float(overall_kept / overall_original) if overall_original else None,
            "reason_counts": {str(key): int(value) for key, value in sorted(overall_reasons.items())},
        },
        "by_split": split_reports,
        "out_dir": out_dir.as_posix(),
    }
    (out_dir / "filtering_report.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "filtering_report.md").write_text(render_filter_summary_md(summary))
    return summary


def maybe_copy_report(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(src.read_text())


def main(argv: Optional[Sequence[str]] = None) -> int:
    bundle_root = Path(__file__).resolve().parents[1]
    parser = build_parser(bundle_root)
    args = parser.parse_args(argv)

    if not np.isclose(args.train_ratio + args.val_ratio + args.test_ratio, 1.0):
        raise ValueError("train/val/test ratios must sum to 1.0")

    out_bundle_root = args.out_bundle_root.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    trajectory_json_root = (
        args.trajectory_json_root.expanduser().resolve()
        if args.trajectory_json_root is not None
        else default_trajectory_json_root(out_bundle_root)
    )
    reports_dir = out_bundle_root / "reports"
    canonical_out_dir = out_bundle_root / "Data" / "canonical_iam_raw"
    word_level_out_dir = out_bundle_root / "Data" / "processed_iam_all_writers"
    audit_out_dir = out_bundle_root / "audits" / "iam_dot_cross_timing"
    filter_inputs_dir = out_bundle_root / "Data" / "filter_inputs"
    filtered_out_dir = out_bundle_root / "Data" / "processed_iam_local_only"

    data_xml_inspection = inspect_data_xml_folder(data_root)
    canonical_build = build_canonical_and_word_level(
        trajectory_json_root=trajectory_json_root,
        canonical_out_dir=canonical_out_dir,
        word_level_out_dir=word_level_out_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    dataset_summary = {
        "source_label": SOURCE_LABEL,
        "data_xml_inspection": data_xml_inspection,
        "canonical_build": canonical_build,
    }
    write_reports_json_md(
        reports_dir,
        "iam_dataset_build_summary",
        dataset_summary,
        render_dataset_build_summary_md(dataset_summary),
    )

    audit_summary = run_dot_cross_timing_audit(
        canonical_jsonl=canonical_out_dir / "canonical_samples.jsonl",
        canonical_npz=canonical_out_dir / "canonical_samples.npz",
        out_dir=audit_out_dir,
        source_filter=SOURCE_NAME,
        source_label=SOURCE_LABEL,
        max_examples_per_category=int(args.audit_max_examples_per_category),
        limit_samples=args.audit_limit_samples,
    )
    maybe_copy_report(audit_out_dir / "summary.json", reports_dir / "iam_dot_cross_audit_summary.json")
    maybe_copy_report(audit_out_dir / "summary.md", reports_dir / "iam_dot_cross_audit_summary.md")

    occurrences_path = audit_out_dir / "occurrences.jsonl"
    operational_rows = build_operational_rows(json.loads(line) for line in occurrences_path.open(encoding="utf-8"))
    occurrences_with_operational_label_path = filter_inputs_dir / "occurrences_with_operational_label.jsonl"
    write_jsonl_rows(occurrences_with_operational_label_path, operational_rows)

    operational_summary = summarize_operational_rows(operational_rows)
    filter_inputs_dir.mkdir(parents=True, exist_ok=True)
    (filter_inputs_dir / "operational_summary.json").write_text(json.dumps(operational_summary, indent=2))
    (filter_inputs_dir / "operational_summary.md").write_text(
        render_operational_summary_markdown(operational_summary, None)
    )

    filter_summary = build_operational_filter_outputs(
        word_level_dir=word_level_out_dir,
        occurrences_with_operational_label_path=occurrences_with_operational_label_path,
        out_dir=filtered_out_dir,
    )
    maybe_copy_report(filtered_out_dir / "filtering_report.json", reports_dir / "iam_operational_filter_summary.json")
    maybe_copy_report(filtered_out_dir / "filtering_report.md", reports_dir / "iam_operational_filter_summary.md")

    print(f"[iam-build] dataset summary: {(reports_dir / 'iam_dataset_build_summary.json').as_posix()}")
    print(f"[iam-build] audit summary: {(reports_dir / 'iam_dot_cross_audit_summary.json').as_posix()}")
    print(f"[iam-build] filter summary: {(reports_dir / 'iam_operational_filter_summary.json').as_posix()}")
    print(f"[iam-build] final train split: {(filtered_out_dir / 'train.npz').as_posix()}")
    return 0
