from __future__ import annotations

"""Render manual-review panels from real DeepWriting audit occurrences."""

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .dot_cross_audit import (
    DELAY_CATEGORIES,
    DRAW_EVENTS,
    TARGET_CHARS,
    DrawComponent,
    build_draw_components,
    build_global_stroke_ids,
    collect_word_indices,
    compute_i_or_j_pair_score,
    compute_t_pair_score,
    flatten_ranges,
)
from .dot_cross_operational import operational_label_from_audit_label
from .utils import maybe_import_matplotlib, slugify_text


REQUIRED_OCCURRENCE_FIELDS = {
    "sample_id",
    "split",
    "source",
    "writer_id",
    "text",
    "word_index",
    "char_index",
    "char",
    "word",
    "category",
    "ambiguity_reason",
    "component_spans",
    "base_span",
    "mark_span",
    "point_index_gap",
    "stroke_index_gap",
    "timestamp_gap",
    "intervening_characters",
    "pair_resolution",
    "row_index",
}


@dataclass
class CanonicalStore:
    rows: List[dict]
    arrays: Dict[str, np.ndarray]
    sample_id_to_row_index: Dict[str, int]


@dataclass
class ReviewCase:
    occurrence: dict
    occurrence_id: str
    row_index: int
    row: dict
    points_abs: np.ndarray
    timestamps: np.ndarray
    events: np.ndarray
    word_indices: np.ndarray
    char_indices: np.ndarray
    word_components: List[DrawComponent]
    target_components: List[DrawComponent]
    base_component: Optional[DrawComponent]
    mark_component: Optional[DrawComponent]
    suspicion_score: Optional[float]


def build_occurrence_id(occurrence: dict) -> str:
    return (
        f"{slugify_text(str(occurrence['sample_id']))}"
        f"__w{int(occurrence['word_index'])}"
        f"__c{int(occurrence['char_index'])}"
        f"__{str(occurrence['char']).lower()}"
    )


def validate_occurrence_row(row: dict) -> None:
    missing = sorted(REQUIRED_OCCURRENCE_FIELDS - set(row.keys()))
    if missing:
        raise KeyError(f"Occurrence row is missing required fields: {missing}")


def load_occurrences(path: Path) -> List[dict]:
    """Load occurrence rows produced by the existing audit and attach stable occurrence IDs."""
    occurrences: List[dict] = []
    for line_number, line in enumerate(path.open(), start=1):
        row = json.loads(line)
        validate_occurrence_row(row)
        row["occurrence_id"] = build_occurrence_id(row)
        occurrences.append(row)
    if not occurrences:
        raise ValueError(f"No occurrences found in {path}")
    return occurrences


def load_canonical_store(canonical_jsonl: Path, canonical_npz: Path) -> CanonicalStore:
    """Load canonical DeepWriting geometry plus the aligned segmentation manifest."""
    rows = [json.loads(line) for line in canonical_jsonl.open()]
    npz = np.load(canonical_npz, allow_pickle=True)
    required_npz = ("sample_id", "points_abs", "timestamps", "events")
    for key in required_npz:
        if key not in npz:
            raise KeyError(f"Missing '{key}' in {canonical_npz}")
    arrays = {
        "sample_id": np.asarray(npz["sample_id"], dtype=object),
        "points_abs": np.asarray(npz["points_abs"], dtype=object),
        "timestamps": np.asarray(npz["timestamps"], dtype=object),
        "events": np.asarray(npz["events"], dtype=object),
    }
    npz.close()
    if len(rows) != len(arrays["sample_id"]):
        raise ValueError("canonical JSONL and NPZ have different lengths")
    sample_id_to_row_index = {str(sample_id): index for index, sample_id in enumerate(arrays["sample_id"].tolist())}
    return CanonicalStore(rows=rows, arrays=arrays, sample_id_to_row_index=sample_id_to_row_index)


def split_csv_arg(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    values = [part.strip() for part in str(raw).split(",")]
    return [value for value in values if value]


def occurrence_matches_substring(row: dict, pattern: Optional[str]) -> bool:
    if not pattern:
        return True
    pattern_lower = pattern.lower()
    return pattern_lower in str(row.get("word", "")).lower() or pattern_lower in str(row.get("text", "")).lower()


def compute_ambiguous_delay_suspicion_from_occurrence(occurrence: dict) -> Optional[float]:
    if str(occurrence.get("category")) != "ambiguous":
        return None

    score = 0.0
    if str(occurrence.get("ambiguity_reason") or "") == "low_confidence_pair":
        pair_resolution = occurrence.get("pair_resolution") or {}
        best_score = float(pair_resolution.get("best_score") or 0.0)
        second_score = float(pair_resolution.get("second_score") or 0.0)
        score += 4.0 + best_score - max(second_score, 0.0)

    spans = [dict(span) for span in occurrence.get("component_spans") or [] if not bool(span.get("is_orphan"))]
    spans.sort(key=lambda span: int(span["start_index"]))
    if len(spans) >= 2:
        base_like = max(spans, key=lambda span: float(span.get("path_length", 0.0)) + 0.5 * float(span.get("height", 0.0)))
        base_path = max(float(base_like.get("path_length", 1.0)), 1.0)
        base_end = int(base_like["end_index"])
        for span in spans:
            if span is base_like:
                continue
            start_index = int(span["start_index"])
            if start_index <= base_end:
                continue
            gap = start_index - base_end
            size_ratio = float(span.get("path_length", 0.0)) / base_path
            if size_ratio <= 0.45:
                score += 1.5
            score += min(gap / 40.0, 4.0)
            score += min(float(span.get("height", 0.0)) / max(float(base_like.get("height", 1.0)), 1.0), 1.0)
            break

    score += 0.5 * max(int(occurrence.get("draw_component_count") or 0) - 1, 0)
    return score


def filter_occurrences(
    occurrences: Sequence[dict],
    *,
    char_filter: str,
    label_filter: str,
    word_filter: Optional[str],
    writer_filter: Optional[str],
    ambiguous_reason: Optional[str],
) -> List[dict]:
    filtered: List[dict] = []
    for row in occurrences:
        if str(row.get("source")) != "deepwriting":
            continue
        if char_filter != "all" and str(row.get("char")).lower() != char_filter:
            continue
        if label_filter != "all" and str(row.get("category")) != label_filter:
            continue
        if writer_filter and str(row.get("writer_id")) != writer_filter:
            continue
        if ambiguous_reason and str(row.get("ambiguity_reason") or "") != ambiguous_reason:
            continue
        if not occurrence_matches_substring(row, word_filter):
            continue
        filtered.append(row)
    return filtered


def select_by_ids(occurrences: Sequence[dict], ids: Sequence[str], count: int) -> List[dict]:
    occurrence_map = {str(row["occurrence_id"]): row for row in occurrences}
    sample_map: Dict[str, List[dict]] = defaultdict(list)
    for row in occurrences:
        sample_map[str(row["sample_id"])].append(row)

    selected: List[dict] = []
    seen = set()
    for token in ids:
        matches: List[dict] = []
        if token in occurrence_map:
            matches = [occurrence_map[token]]
        elif token in sample_map:
            matches = list(sample_map[token])
        else:
            raise ValueError(f"--ids token '{token}' did not match any occurrence_id or sample_id")
        for match in matches:
            occurrence_id = str(match["occurrence_id"])
            if occurrence_id in seen:
                continue
            selected.append(match)
            seen.add(occurrence_id)
            if count > 0 and len(selected) >= count:
                return selected
    return selected


def stratified_sample(occurrences: Sequence[dict], count: int, rng: random.Random, *, char_filter: str, label_filter: str) -> List[dict]:
    if count <= 0 or count >= len(occurrences):
        return list(occurrences)

    groups: Dict[Tuple[str, ...], List[dict]] = defaultdict(list)
    for row in occurrences:
        key_parts: List[str] = []
        if char_filter == "all":
            key_parts.append(str(row["char"]))
        if label_filter == "all":
            key_parts.append(str(row["category"]))
        group_key = tuple(key_parts) if key_parts else ("all",)
        groups[group_key].append(row)

    for group_rows in groups.values():
        rng.shuffle(group_rows)

    selected: List[dict] = []
    active_keys = list(sorted(groups.keys()))
    while active_keys and len(selected) < count:
        next_active: List[Tuple[str, ...]] = []
        for key in active_keys:
            group_rows = groups[key]
            if not group_rows:
                continue
            selected.append(group_rows.pop())
            if group_rows:
                next_active.append(key)
            if len(selected) >= count:
                break
        active_keys = next_active
    return selected


def review_priority_sample(
    occurrences: Sequence[dict],
    count: int,
    rng: random.Random,
    *,
    rank_ambiguous_by_suspicion: bool,
) -> List[dict]:
    """Prefer ambiguous cases while reserving a smaller share for committed sanity checks."""
    if count <= 0 or count >= len(occurrences):
        rows = list(occurrences)
        rng.shuffle(rows)
        return rows

    grouped: Dict[str, List[dict]] = {"ambiguous": [], "delayed": [], "local": []}
    for row in occurrences:
        grouped[operational_label_from_audit_label(str(row["category"]))].append(row)

    if rank_ambiguous_by_suspicion:
        grouped["ambiguous"].sort(
            key=lambda row: compute_ambiguous_delay_suspicion_from_occurrence(row) or float("-inf"),
            reverse=True,
        )
    else:
        rng.shuffle(grouped["ambiguous"])
    rng.shuffle(grouped["delayed"])
    rng.shuffle(grouped["local"])

    desired = {
        "ambiguous": max(int(round(count * 0.70)), 1),
        "delayed": max(int(round(count * 0.15)), 1 if count >= 5 else 0),
        "local": max(int(round(count * 0.15)), 1 if count >= 5 else 0),
    }
    total_desired = sum(desired.values())
    while total_desired > count:
        for label in ("ambiguous", "delayed", "local"):
            minimum = 1 if label == "ambiguous" and count > 0 else 0
            if desired[label] > minimum and total_desired > count:
                desired[label] -= 1
                total_desired -= 1
    while total_desired < count:
        desired["ambiguous"] += 1
        total_desired += 1

    selected: List[dict] = []
    seen = set()
    for label in ("ambiguous", "delayed", "local"):
        take = min(desired[label], len(grouped[label]))
        for row in grouped[label][:take]:
            occurrence_id = str(row["occurrence_id"])
            if occurrence_id in seen:
                continue
            selected.append(row)
            seen.add(occurrence_id)

    if len(selected) < count:
        remaining: List[dict] = []
        for label in ("ambiguous", "delayed", "local"):
            remaining.extend(row for row in grouped[label] if str(row["occurrence_id"]) not in seen)
        if rank_ambiguous_by_suspicion:
            delayed_rows = [row for row in remaining if operational_label_from_audit_label(str(row["category"])) == "delayed"]
            local_rows = [row for row in remaining if operational_label_from_audit_label(str(row["category"])) == "local"]
            rng.shuffle(delayed_rows)
            rng.shuffle(local_rows)
            remaining = delayed_rows + local_rows
        else:
            rng.shuffle(remaining)
        for row in remaining:
            occurrence_id = str(row["occurrence_id"])
            if occurrence_id in seen:
                continue
            selected.append(row)
            seen.add(occurrence_id)
            if len(selected) >= count:
                break
    return selected[:count]


def sample_occurrences(
    occurrences: Sequence[dict],
    *,
    count: int,
    seed: int,
    sample_mode: str,
    char_filter: str,
    label_filter: str,
    ids: Sequence[str],
    rank_ambiguous_by_suspicion: bool,
) -> List[dict]:
    rng = random.Random(seed)
    if ids:
        return select_by_ids(occurrences, ids, count=count)

    rows = list(occurrences)
    if rank_ambiguous_by_suspicion:
        ambiguous_rows = [row for row in rows if str(row.get("category")) == "ambiguous"]
        non_ambiguous_rows = [row for row in rows if str(row.get("category")) != "ambiguous"]
        ambiguous_rows.sort(
            key=lambda row: compute_ambiguous_delay_suspicion_from_occurrence(row) or float("-inf"),
            reverse=True,
        )
        rows = non_ambiguous_rows + ambiguous_rows if label_filter == "all" else ambiguous_rows if label_filter == "ambiguous" else rows

    if count <= 0 or count >= len(rows):
        return rows

    if sample_mode == "review_priority":
        return review_priority_sample(
            rows,
            count,
            rng,
            rank_ambiguous_by_suspicion=rank_ambiguous_by_suspicion,
        )

    if sample_mode == "stratified":
        return stratified_sample(rows, count, rng, char_filter=char_filter, label_filter=label_filter)

    if rank_ambiguous_by_suspicion and label_filter == "ambiguous":
        return rows[:count]

    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:count]


def find_matching_char_entry(row: dict, occurrence: dict) -> dict:
    for entry in row["char_segments"]:
        if (
            int(entry["word_index"]) == int(occurrence["word_index"])
            and int(entry["char_index"]) == int(occurrence["char_index"])
            and str(entry.get("char") or "").lower() == str(occurrence["char"]).lower()
        ):
            return dict(entry)
    raise KeyError(
        f"Could not find char entry for occurrence {occurrence['occurrence_id']} in sample {occurrence['sample_id']}"
    )


def match_component_from_span(components: Sequence[DrawComponent], span: Optional[dict]) -> Optional[DrawComponent]:
    if not span:
        return None
    start_index = int(span["start_index"])
    end_index = int(span["end_index"])
    for component in components:
        if component.start_index == start_index and component.end_index == end_index:
            return component
    return None


def build_review_case(occurrence: dict, store: CanonicalStore) -> ReviewCase:
    """Resolve one occurrence row back to its real canonical word trajectory and components."""
    row_index = int(occurrence.get("row_index", store.sample_id_to_row_index[str(occurrence["sample_id"])]))
    row = store.rows[row_index]
    if str(row["sample_id"]) != str(occurrence["sample_id"]):
        raise ValueError(
            f"Occurrence {occurrence['occurrence_id']} row_index mismatch: {row['sample_id']} != {occurrence['sample_id']}"
        )
    if str(row["source"]) != "deepwriting":
        raise ValueError(f"Occurrence {occurrence['occurrence_id']} is not DeepWriting")

    points_abs = np.asarray(store.arrays["points_abs"][row_index], dtype=np.float32)
    timestamps = np.asarray(store.arrays["timestamps"][row_index], dtype=np.float64)
    events = np.asarray(store.arrays["events"][row_index], dtype=np.int64)
    char_entry = find_matching_char_entry(row, occurrence)
    word_indices = collect_word_indices(row["word_segments"], int(occurrence["word_index"]))
    char_indices = flatten_ranges(char_entry.get("ranges") or [])
    point_to_stroke = build_global_stroke_ids(events)
    word_components = build_draw_components(word_indices, points_abs, timestamps, events, point_to_stroke=point_to_stroke)
    target_components = build_draw_components(char_indices, points_abs, timestamps, events, point_to_stroke=point_to_stroke)
    base_component = match_component_from_span(target_components, occurrence.get("base_span"))
    mark_component = match_component_from_span(target_components, occurrence.get("mark_span"))
    suspicion_score = compute_ambiguous_delay_suspicion_from_occurrence(occurrence)
    return ReviewCase(
        occurrence=occurrence,
        occurrence_id=str(occurrence["occurrence_id"]),
        row_index=row_index,
        row=row,
        points_abs=points_abs,
        timestamps=timestamps,
        events=events,
        word_indices=word_indices,
        char_indices=char_indices,
        word_components=word_components,
        target_components=target_components,
        base_component=base_component,
        mark_component=mark_component,
        suspicion_score=suspicion_score,
    )


def iterate_word_segments(points_abs: np.ndarray, events: np.ndarray, indices: np.ndarray) -> List[dict]:
    sorted_indices = np.asarray(sorted(set(int(value) for value in indices.tolist())), dtype=np.int64)
    segments: List[dict] = []
    for prev, current in zip(sorted_indices[:-1], sorted_indices[1:]):
        if int(current) != int(prev) + 1:
            continue
        segment = np.asarray(points_abs[[int(prev), int(current)]], dtype=np.float32)
        is_draw = int(events[int(prev)]) != 2 and int(events[int(current)]) in DRAW_EVENTS
        segments.append(
            {
                "segment": segment,
                "prev": int(prev),
                "current": int(current),
                "is_draw": bool(is_draw),
            }
        )
    return segments


def component_indices_set(component: Optional[DrawComponent]) -> set[int]:
    if component is None:
        return set()
    return {int(value) for value in component.indices.tolist()}


def infer_target_bbox(case: ReviewCase) -> Tuple[float, float, float, float]:
    focus_indices: List[int] = []
    if case.base_component is not None:
        focus_indices.extend(int(value) for value in case.base_component.indices.tolist())
    if case.mark_component is not None:
        focus_indices.extend(int(value) for value in case.mark_component.indices.tolist())
    if not focus_indices:
        focus_indices.extend(int(value) for value in case.char_indices.tolist())
    if not focus_indices:
        focus_indices.extend(int(value) for value in case.word_indices.tolist())
    points = np.asarray(case.points_abs[np.asarray(sorted(set(focus_indices)), dtype=np.int64)], dtype=np.float32)
    min_x = float(np.min(points[:, 0]))
    max_x = float(np.max(points[:, 0]))
    min_y = float(np.min(points[:, 1]))
    max_y = float(np.max(points[:, 1]))
    pad_x = max((max_x - min_x) * 0.25, 8.0)
    pad_y = max((max_y - min_y) * 0.25, 8.0)
    return min_x - pad_x, max_x + pad_x, min_y - pad_y, max_y + pad_y


def apply_plot_style(ax, title: str) -> None:
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def draw_start_end_markers(ax, points_abs: np.ndarray, indices: np.ndarray) -> None:
    if indices.size == 0:
        return
    start_point = points_abs[int(indices[0])]
    end_point = points_abs[int(indices[-1])]
    ax.scatter([float(start_point[0])], [-float(start_point[1])], color="tab:green", s=25, zorder=5)
    ax.scatter([float(end_point[0])], [-float(end_point[1])], color="tab:red", s=25, zorder=5)


def plot_temporal_gradient(ax, case: ReviewCase, *, title: str, alpha_non_draw: float = 0.35) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    segments = iterate_word_segments(case.points_abs, case.events, case.word_indices)
    draw_segments = [segment for segment in segments if segment["is_draw"]]
    move_segments = [segment for segment in segments if not segment["is_draw"]]
    cmap = plt.get_cmap("viridis")
    draw_total = max(len(draw_segments) - 1, 1)
    for draw_index, item in enumerate(draw_segments):
        color = cmap(draw_index / draw_total)
        segment = item["segment"]
        ax.plot(segment[:, 0], -segment[:, 1], color=color, linewidth=1.6, alpha=0.95)
    for item in move_segments:
        segment = item["segment"]
        ax.plot(segment[:, 0], -segment[:, 1], color="0.8", linewidth=0.8, linestyle="--", alpha=alpha_non_draw)
    draw_start_end_markers(ax, case.points_abs, case.word_indices)
    apply_plot_style(ax, title)


def plot_component_order(ax, case: ReviewCase, *, title: str) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    plot_temporal_gradient(ax, case, title="")
    colors = plt.get_cmap("tab20")
    draw_components = [component for component in case.word_components if not component.is_orphan and component.indices.size >= 2]
    for component_index, component in enumerate(draw_components):
        color = colors(component_index % 20)
        for prev, current in zip(component.indices[:-1], component.indices[1:]):
            if int(current) != int(prev) + 1:
                continue
            if int(case.events[int(prev)]) == 2 or int(case.events[int(current)]) not in DRAW_EVENTS:
                continue
            segment = case.points_abs[[int(prev), int(current)]]
            ax.plot(segment[:, 0], -segment[:, 1], color=color, linewidth=2.4, alpha=0.95)
        ax.text(
            float(component.center_x),
            -float(component.center_y),
            str(component_index),
            fontsize=8,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "lw": 0.8, "alpha": 0.9},
            zorder=6,
        )
    apply_plot_style(ax, title)


def plot_target_focus(ax, case: ReviewCase, *, title: str) -> None:
    base_indices = component_indices_set(case.base_component)
    mark_indices = component_indices_set(case.mark_component)
    char_indices_set = {int(value) for value in case.char_indices.tolist()}
    segments = iterate_word_segments(case.points_abs, case.events, case.word_indices)

    for item in segments:
        segment = item["segment"]
        prev_index = int(item["prev"])
        current_index = int(item["current"])
        if not item["is_draw"]:
            ax.plot(segment[:, 0], -segment[:, 1], color="0.9", linewidth=0.7, linestyle="--", alpha=0.4)
            continue
        if prev_index in base_indices and current_index in base_indices:
            ax.plot(segment[:, 0], -segment[:, 1], color="tab:orange", linewidth=2.8, alpha=0.95, label="base")
        elif prev_index in mark_indices and current_index in mark_indices:
            ax.plot(segment[:, 0], -segment[:, 1], color="tab:red", linewidth=2.8, alpha=0.95, label="mark")
        elif prev_index in char_indices_set and current_index in char_indices_set and not base_indices and not mark_indices:
            ax.plot(segment[:, 0], -segment[:, 1], color="tab:blue", linewidth=2.2, alpha=0.9, label="target_char")
        else:
            ax.plot(segment[:, 0], -segment[:, 1], color="0.7", linewidth=1.0, alpha=0.35)

    note = "audit base/mark unresolved" if case.base_component is None or case.mark_component is None else "base/mark from audit row"
    ax.text(
        0.02,
        0.02,
        note,
        transform=ax.transAxes,
        fontsize=8,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75", "lw": 0.8, "alpha": 0.9},
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=8, frameon=False)
    apply_plot_style(ax, title)


def plot_zoom_panel(ax, case: ReviewCase, *, title: str) -> None:
    plot_target_focus(ax, case, title="")
    min_x, max_x, min_y, max_y = infer_target_bbox(case)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(-max_y, -min_y)
    draw_start_end_markers(ax, case.points_abs, case.char_indices if case.char_indices.size else case.word_indices)
    apply_plot_style(ax, title)


def compact_tile_title(case: ReviewCase) -> str:
    return (
        f"{case.occurrence['word']} | {case.occurrence['char']} | {case.occurrence['category']}\n"
        f"gap={case.occurrence.get('point_index_gap')} chars={case.occurrence.get('intervening_characters')}"
    )


def render_compact_tile(ax, case: ReviewCase) -> None:
    plot_temporal_gradient(ax, case, title="")
    if case.base_component is not None:
        for prev, current in zip(case.base_component.indices[:-1], case.base_component.indices[1:]):
            if int(current) != int(prev) + 1:
                continue
            if int(case.events[int(prev)]) == 2 or int(case.events[int(current)]) not in DRAW_EVENTS:
                continue
            segment = case.points_abs[[int(prev), int(current)]]
            ax.plot(segment[:, 0], -segment[:, 1], color="tab:orange", linewidth=2.0, alpha=0.95)
    if case.mark_component is not None:
        for prev, current in zip(case.mark_component.indices[:-1], case.mark_component.indices[1:]):
            if int(current) != int(prev) + 1:
                continue
            if int(case.events[int(prev)]) == 2 or int(case.events[int(current)]) not in DRAW_EVENTS:
                continue
            segment = case.points_abs[[int(prev), int(current)]]
            ax.plot(segment[:, 0], -segment[:, 1], color="tab:red", linewidth=2.0, alpha=0.95)
    ax.set_title(compact_tile_title(case), fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def save_review_figure(path: Path, case: ReviewCase) -> Optional[str]:
    plt = maybe_import_matplotlib()
    if plt is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_temporal_gradient(axes[0, 0], case, title="A. Full Word / Temporal Gradient")
    plot_component_order(axes[0, 1], case, title="B. Draw Components / Order")
    plot_target_focus(axes[1, 0], case, title="C. Target Focus")
    plot_zoom_panel(axes[1, 1], case, title="D. Zoomed Target Region")

    pair_resolution = case.occurrence.get("pair_resolution") or {}
    confidence_text = (
        f"score={pair_resolution.get('score')}"
        if pair_resolution.get("score") is not None
        else f"best_score={pair_resolution.get('best_score')}"
        if pair_resolution.get("best_score") is not None
        else None
    )
    meta_parts = [
        f"word={case.occurrence['word']}",
        f"char={case.occurrence['char']}",
        f"label={case.occurrence['category']}",
        f"occurrence_id={case.occurrence_id}",
        f"sample_id={case.occurrence['sample_id']}",
        f"writer_id={case.occurrence['writer_id']}",
        f"point_gap={case.occurrence.get('point_index_gap')}",
        f"stroke_gap={case.occurrence.get('stroke_index_gap')}",
        f"time_gap={case.occurrence.get('timestamp_gap')}",
        f"intervening_chars={case.occurrence.get('intervening_characters')}",
    ]
    if case.occurrence.get("ambiguity_reason"):
        meta_parts.append(f"ambiguous_reason={case.occurrence['ambiguity_reason']}")
    if confidence_text:
        meta_parts.append(confidence_text)
    if case.suspicion_score is not None:
        meta_parts.append(f"ambiguous_delay_suspicion={case.suspicion_score:.3f}")
    fig.suptitle(
        f"{case.occurrence['word']} / {case.occurrence['char']} / {case.occurrence['category']}\n"
        + " | ".join(meta_parts),
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.as_posix()


def save_grid(path: Path, title: str, cases: Sequence[ReviewCase]) -> Optional[str]:
    if not cases:
        return None
    plt = maybe_import_matplotlib()
    if plt is None:
        return None
    cols = 4
    rows = int(math.ceil(len(cases) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.2), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, case in zip(axes.reshape(-1), cases):
        render_compact_tile(ax, case)
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.as_posix()


def write_manifest_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_manifest_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manual_annotation_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames = [
        "occurrence_id",
        "predicted_label",
        "human_label",
        "human_notes",
        "likely_true_delayed",
        "body_mark_pair_correct",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "occurrence_id": row["occurrence_id"],
                    "predicted_label": row["audit_label"],
                    "human_label": "",
                    "human_notes": "",
                    "likely_true_delayed": "",
                    "body_mark_pair_correct": "",
                }
            )


def auto_run_name(args) -> str:
    parts = [
        f"char_{args.char}",
        f"label_{args.label}",
        f"mode_{args.sample_mode}",
        f"n_{args.count}",
        f"seed_{args.seed}",
    ]
    if args.word_filter:
        parts.append(f"word_{slugify_text(args.word_filter)}")
    if args.writer_filter:
        parts.append(f"writer_{slugify_text(args.writer_filter)}")
    if args.ambiguous_reason:
        parts.append(f"ambig_{slugify_text(args.ambiguous_reason)}")
    if args.ids:
        parts.append("ids")
    if args.rank_ambiguous_by_suspicion:
        parts.append("rank_suspicious")
    return "__".join(parts)


def build_parser(bundle_root: Path) -> argparse.ArgumentParser:
    project_root = bundle_root.parent
    audit_dir = bundle_root / "audits" / "dot_cross_timing_full"
    canonical_dir = project_root / "Version1A" / "Data" / "processed" / "canonical_raw"
    parser = argparse.ArgumentParser(
        description="Render manual-review panels for real DeepWriting i/j/t dot-cross timing occurrences."
    )
    parser.add_argument("--char", choices=["all", "i", "j", "t"], default="all")
    parser.add_argument("--label", choices=["all", *DELAY_CATEGORIES], default="all")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--sample-mode", choices=["random", "stratified"], default="stratified")
    parser.add_argument("--word-filter", type=str, default=None)
    parser.add_argument("--writer-filter", type=str, default=None)
    parser.add_argument("--ambiguous-reason", type=str, default=None)
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated occurrence_ids or sample_ids.")
    parser.add_argument(
        "--rank-ambiguous-by-suspicion",
        action="store_true",
        help="Sort ambiguous cases by a heuristic delayed-likelihood score before truncation.",
    )
    parser.add_argument("--occurrences-path", type=Path, default=audit_dir / "occurrences.jsonl")
    parser.add_argument("--canonical-jsonl", type=Path, default=canonical_dir / "canonical_samples.jsonl")
    parser.add_argument("--canonical-npz", type=Path, default=canonical_dir / "canonical_samples.npz")
    parser.add_argument("--out-dir", type=Path, default=audit_dir / "manual_review")
    parser.add_argument("--run-name", type=str, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint for manual-review rendering."""
    bundle_root = Path(__file__).resolve().parents[1]
    parser = build_parser(bundle_root)
    args = parser.parse_args(argv)

    occurrences = load_occurrences(args.occurrences_path.expanduser().resolve())
    filtered = filter_occurrences(
        occurrences,
        char_filter=args.char,
        label_filter=args.label,
        word_filter=args.word_filter,
        writer_filter=args.writer_filter,
        ambiguous_reason=args.ambiguous_reason,
    )
    if not filtered:
        raise ValueError("No occurrences matched the requested filters")

    selected_rows = sample_occurrences(
        filtered,
        count=int(args.count),
        seed=int(args.seed),
        sample_mode=str(args.sample_mode),
        char_filter=str(args.char),
        label_filter=str(args.label),
        ids=split_csv_arg(args.ids),
        rank_ambiguous_by_suspicion=bool(args.rank_ambiguous_by_suspicion),
    )
    if not selected_rows:
        raise ValueError("Sampling produced no occurrences")

    store = load_canonical_store(
        args.canonical_jsonl.expanduser().resolve(),
        args.canonical_npz.expanduser().resolve(),
    )

    run_name = args.run_name or auto_run_name(args)
    run_dir = args.out_dir.expanduser().resolve() / run_name
    figures_dir = run_dir / "figures"
    grids_dir = run_dir / "grids"
    run_dir.mkdir(parents=True, exist_ok=True)

    cases = [build_review_case(row, store) for row in selected_rows]

    grid_groups: Dict[str, List[ReviewCase]] = defaultdict(list)
    manifest_rows: List[dict] = []
    for case in cases:
        figure_name = f"{case.occurrence_id}.png"
        figure_path = figures_dir / str(case.occurrence["category"]) / figure_name
        saved_figure = save_review_figure(figure_path, case)
        if not saved_figure:
            raise RuntimeError("Matplotlib is unavailable; cannot render manual review figures")

        label_group = str(case.occurrence["category"])
        char_label_group = f"{case.occurrence['char']}_{case.occurrence['category']}"
        grid_groups[label_group].append(case)
        grid_groups[char_label_group].append(case)
        manifest_rows.append(
            {
                "occurrence_id": case.occurrence_id,
                "sample_id": str(case.occurrence["sample_id"]),
                "word": str(case.occurrence["word"]),
                "char": str(case.occurrence["char"]),
                "audit_label": str(case.occurrence["category"]),
                "writer_id": str(case.occurrence["writer_id"]),
                "split": str(case.occurrence["split"]),
                "ambiguous_reason": str(case.occurrence.get("ambiguity_reason") or ""),
                "figure_path": saved_figure,
                "grid_group": f"{label_group};{char_label_group}",
                "confidence_score": (
                    case.occurrence.get("pair_resolution", {}).get("score")
                    if isinstance(case.occurrence.get("pair_resolution"), dict)
                    else None
                ),
                "ambiguous_delay_suspicion": case.suspicion_score,
                "point_gap": case.occurrence.get("point_index_gap"),
                "stroke_gap": case.occurrence.get("stroke_index_gap"),
                "timestamp_gap": case.occurrence.get("timestamp_gap"),
                "intervening_chars": case.occurrence.get("intervening_characters"),
            }
        )

    grid_manifest: Dict[str, Optional[str]] = {}
    for group_name, group_cases in sorted(grid_groups.items()):
        title = f"Manual Review / {group_name} / n={len(group_cases)}"
        grid_path = grids_dir / f"{group_name}_grid.png"
        grid_manifest[group_name] = save_grid(grid_path, title, group_cases)

    write_manifest_jsonl(run_dir / "manifest.jsonl", manifest_rows)
    write_manifest_csv(run_dir / "manifest.csv", manifest_rows)
    write_manual_annotation_csv(run_dir / "manual_annotation.csv", manifest_rows)

    selection_summary = {
        "args": {
            key: value.as_posix() if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "run_dir": run_dir.as_posix(),
        "num_available_after_filters": len(filtered),
        "num_selected": len(cases),
        "selected_occurrence_ids": [case.occurrence_id for case in cases],
        "grid_manifest": grid_manifest,
    }
    (run_dir / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2))

    print(f"[manual-review] wrote {run_dir.as_posix()}")
    print(f"[manual-review] selected {len(cases)} occurrences from {len(filtered)} filtered candidates")
    return 0
