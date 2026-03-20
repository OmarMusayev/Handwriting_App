from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .utils import maybe_import_matplotlib, slugify_text


DRAW_EVENTS = {1, 2}
TARGET_CHARS = ("i", "j", "t")
DELAY_CATEGORIES = ("immediate", "nearby", "delayed", "ambiguous")


@dataclass(frozen=True)
class DrawComponent:
    indices: np.ndarray
    points: np.ndarray
    start_index: int
    end_index: int
    start_time: float
    end_time: float
    stroke_id: Optional[int]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: float
    height: float
    center_x: float
    center_y: float
    path_length: float
    disp_x: float
    disp_y: float
    is_orphan: bool = False


def flatten_ranges(ranges: Sequence[Sequence[int]]) -> np.ndarray:
    values: List[int] = []
    for group in ranges:
        values.extend(int(index) for index in group)
    if not values:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(sorted(set(values)), dtype=np.int64)


def compute_word_text_map(word_segments: Sequence[dict]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for word_segment in word_segments:
        word_index = int(word_segment["word_index"])
        mapping[word_index] = str(word_segment.get("text") or "")
    return mapping


def build_char_entries(char_segments: Sequence[dict]) -> List[dict]:
    entries: List[dict] = []
    for entry in char_segments:
        indices = flatten_ranges(entry.get("ranges") or [])
        if indices.size == 0:
            continue
        entries.append(
            {
                "word_index": int(entry["word_index"]),
                "char_index": int(entry["char_index"]),
                "char": str(entry.get("char") or ""),
                "start_index": int(indices[0]),
                "end_index": int(indices[-1]),
            }
        )
    entries.sort(key=lambda item: (item["start_index"], item["word_index"], item["char_index"]))
    return entries


def build_global_stroke_ids(events: np.ndarray) -> Dict[int, int]:
    point_to_stroke: Dict[int, int] = {}
    current_stroke = -1
    stroke_is_active = False
    for index in range(1, int(events.shape[0])):
        segment_is_draw = int(events[index - 1]) != 2 and int(events[index]) in DRAW_EVENTS
        if not segment_is_draw:
            stroke_is_active = False
            continue
        if not stroke_is_active:
            current_stroke += 1
            stroke_is_active = True
        point_to_stroke[int(index - 1)] = int(current_stroke)
        point_to_stroke[int(index)] = int(current_stroke)
    return point_to_stroke


def build_draw_components(
    indices: np.ndarray,
    points_abs: np.ndarray,
    timestamps: np.ndarray,
    events: np.ndarray,
    *,
    point_to_stroke: Optional[Dict[int, int]] = None,
) -> List[DrawComponent]:
    sorted_indices = np.asarray(sorted(set(int(value) for value in indices.tolist())), dtype=np.int64)
    if sorted_indices.size == 0:
        return []

    draw_components: List[List[int]] = []
    current: List[int] = []
    prev_index: Optional[int] = None
    used_points: set[int] = set()

    for index in sorted_indices.tolist():
        if prev_index is not None:
            is_consecutive = index == prev_index + 1
            segment_is_draw = is_consecutive and int(events[prev_index]) != 2 and int(events[index]) in DRAW_EVENTS
            if segment_is_draw:
                if not current:
                    current = [prev_index]
                current.append(index)
                used_points.add(prev_index)
                used_points.add(index)
            else:
                if current:
                    draw_components.append(current)
                    current = []
        prev_index = index
    if current:
        draw_components.append(current)

    orphan_points = [int(index) for index in sorted_indices.tolist() if int(index) not in used_points]
    components: List[DrawComponent] = []
    for component_indices in draw_components:
        components.append(
            create_component(
                component_indices,
                points_abs,
                timestamps,
                point_to_stroke=point_to_stroke,
                is_orphan=False,
            )
        )
    for orphan_index in orphan_points:
        components.append(
            create_component(
                [orphan_index],
                points_abs,
                timestamps,
                point_to_stroke=point_to_stroke,
                is_orphan=True,
            )
        )
    components.sort(key=lambda component: (component.start_index, component.end_index))
    return components


def create_component(
    component_indices: Sequence[int],
    points_abs: np.ndarray,
    timestamps: np.ndarray,
    *,
    point_to_stroke: Optional[Dict[int, int]],
    is_orphan: bool,
) -> DrawComponent:
    indices = np.asarray(list(component_indices), dtype=np.int64)
    component_points = np.asarray(points_abs[indices], dtype=np.float32)
    diffs = component_points[1:] - component_points[:-1]
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1))) if diffs.size else 0.0
    min_x = float(np.min(component_points[:, 0]))
    max_x = float(np.max(component_points[:, 0]))
    min_y = float(np.min(component_points[:, 1]))
    max_y = float(np.max(component_points[:, 1]))
    stroke_id = None
    if point_to_stroke is not None:
        for index in indices.tolist():
            if int(index) in point_to_stroke:
                stroke_id = int(point_to_stroke[int(index)])
                break
    return DrawComponent(
        indices=indices,
        points=component_points,
        start_index=int(indices[0]),
        end_index=int(indices[-1]),
        start_time=float(timestamps[int(indices[0])]) if timestamps.size else 0.0,
        end_time=float(timestamps[int(indices[-1])]) if timestamps.size else 0.0,
        stroke_id=stroke_id,
        min_x=min_x,
        max_x=max_x,
        min_y=min_y,
        max_y=max_y,
        width=max_x - min_x,
        height=max_y - min_y,
        center_x=float(np.mean(component_points[:, 0])),
        center_y=float(np.mean(component_points[:, 1])),
        path_length=path_length,
        disp_x=float(component_points[-1, 0] - component_points[0, 0]),
        disp_y=float(component_points[-1, 1] - component_points[0, 1]),
        is_orphan=is_orphan,
    )


def compute_i_or_j_pair_score(base: DrawComponent, mark: DrawComponent) -> Tuple[float, Dict[str, float]]:
    base_scale = max(base.height, base.path_length, 1.0)
    mark_scale = max(mark.height, mark.width, mark.path_length, 1.0)
    vertical_separation = base.center_y - mark.center_y
    x_distance = abs(mark.center_x - base.center_x)
    above = 1.0 if mark.center_y < base.center_y - 0.12 * max(base.height, 1.0) else 0.0
    x_aligned = 1.0 if x_distance <= max(base.width * 0.8, mark.width * 2.0, 8.0) else 0.0
    mark_small = 1.0 if mark_scale <= 0.75 * base_scale else 0.0
    base_tall = 1.0 if base.height >= max(mark.height * 1.5, 6.0) else 0.0
    mark_compact = 1.0 if max(mark.width, mark.height) <= max(base.width, base.height) * 0.7 + 4.0 else 0.0
    order = 1.0 if mark.start_index > base.end_index else 0.0
    score = (2.0 * above) + x_aligned + mark_small + base_tall + mark_compact + order
    details = {
        "above": above,
        "x_aligned": x_aligned,
        "mark_small": mark_small,
        "base_tall": base_tall,
        "mark_compact": mark_compact,
        "order": order,
        "vertical_separation": vertical_separation,
        "x_distance": x_distance,
    }
    return score, details


def compute_t_pair_score(base: DrawComponent, mark: DrawComponent) -> Tuple[float, Dict[str, float]]:
    height = max(base.height, 1.0)
    width = max(base.width, 1.0)
    mark_width = max(mark.width, 1.0)
    mark_height = max(mark.height, 1.0)
    base_vertical = 1.0 if base.height >= max(base.width * 1.4, 8.0) else 0.0
    mark_horizontal = 1.0 if mark.width >= max(mark.height * 1.2, 5.0) or abs(mark.disp_x) >= abs(mark.disp_y) * 1.2 else 0.0
    y_band = 1.0 if (base.min_y - 0.15 * height) <= mark.center_y <= (base.min_y + 0.8 * height) else 0.0
    x_overlap = 1.0 if (mark.max_x >= base.min_x - mark_width) and (mark.min_x <= base.max_x + mark_width) else 0.0
    mark_small = 1.0 if max(mark.path_length, mark_width, mark_height) <= max(base.path_length, height) * 0.9 + 6.0 else 0.0
    base_dominant = 1.0 if height >= max(mark_height * 1.3, mark_width * 0.8, 8.0) else 0.0
    order = 1.0 if mark.start_index > base.end_index else 0.0
    score = (2.0 * base_vertical) + (2.0 * mark_horizontal) + (2.0 * y_band) + x_overlap + mark_small + base_dominant + order
    details = {
        "base_vertical": base_vertical,
        "mark_horizontal": mark_horizontal,
        "y_band": y_band,
        "x_overlap": x_overlap,
        "mark_small": mark_small,
        "base_dominant": base_dominant,
        "order": order,
    }
    return score, details


def find_best_component_pair(target_char: str, components: Sequence[DrawComponent]) -> Tuple[Optional[DrawComponent], Optional[DrawComponent], Optional[dict]]:
    draw_components = [component for component in components if not component.is_orphan and component.indices.size >= 2]
    if len(draw_components) < 2:
        return None, None, None

    best: Optional[Tuple[float, DrawComponent, DrawComponent, Dict[str, float]]] = None
    score_fn = compute_i_or_j_pair_score if target_char in {"i", "j"} else compute_t_pair_score
    threshold = 5.0 if target_char in {"i", "j"} else 6.0
    margin = 0.75

    pair_scores: List[Tuple[float, DrawComponent, DrawComponent, Dict[str, float]]] = []
    for base in draw_components:
        for mark in draw_components:
            if base.start_index == mark.start_index and base.end_index == mark.end_index:
                continue
            if base.end_index >= mark.start_index:
                continue
            score, details = score_fn(base, mark)
            pair_scores.append((score, base, mark, details))

    if not pair_scores:
        return None, None, None

    pair_scores.sort(key=lambda item: item[0], reverse=True)
    best_score, best_base, best_mark, best_details = pair_scores[0]
    second_score = pair_scores[1][0] if len(pair_scores) > 1 else float("-inf")
    if best_score < threshold or (best_score - second_score) < margin:
        return None, None, {
            "reason": "low_confidence_pair",
            "best_score": float(best_score),
            "second_score": float(second_score) if math.isfinite(second_score) else None,
        }
    return best_base, best_mark, {
        "reason": "resolved",
        "score": float(best_score),
        "second_score": float(second_score) if math.isfinite(second_score) else None,
        "details": best_details,
    }


def categorize_delay(intervening_characters: int, stroke_index_gap: Optional[int]) -> str:
    if intervening_characters <= 0 and (stroke_index_gap is None or stroke_index_gap <= 1):
        return "immediate"
    if intervening_characters <= 2 and (stroke_index_gap is None or stroke_index_gap <= 4):
        return "nearby"
    return "delayed"


def bucket_counts(values: Iterable[Optional[float]], edges: Sequence[float], labels: Sequence[str]) -> Dict[str, int]:
    result = {label: 0 for label in labels}
    for value in values:
        if value is None:
            continue
        assigned = False
        for edge_index in range(len(edges) - 1):
            lower = edges[edge_index]
            upper = edges[edge_index + 1]
            is_last = edge_index == len(edges) - 2
            if lower <= float(value) <= upper if is_last else lower <= float(value) < upper:
                result[labels[edge_index]] += 1
                assigned = True
                break
        if not assigned and labels:
            result[labels[-1]] += 1
    return result


def safe_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def collect_word_indices(word_segments: Sequence[dict], word_index: int) -> np.ndarray:
    for word_segment in word_segments:
        if int(word_segment["word_index"]) == int(word_index):
            return flatten_ranges(word_segment.get("ranges") or [])
    return np.zeros((0,), dtype=np.int64)


def classify_occurrence(
    *,
    sample_meta: dict,
    points_abs: np.ndarray,
    timestamps: np.ndarray,
    events: np.ndarray,
    char_entry: dict,
    word_text_map: Dict[int, str],
    all_char_entries: Sequence[dict],
    point_to_stroke: Dict[int, int],
) -> dict:
    target_char = str(char_entry.get("char") or "").lower()
    char_indices = flatten_ranges(char_entry.get("ranges") or [])
    components = build_draw_components(
        char_indices,
        points_abs,
        timestamps,
        events,
        point_to_stroke=point_to_stroke,
    )
    word_index = int(char_entry["word_index"])
    word_indices = collect_word_indices(sample_meta["word_segments"], word_index)
    base_component, mark_component, resolution = find_best_component_pair(target_char, components)

    result = {
        "sample_id": str(sample_meta["sample_id"]),
        "split": str(sample_meta["split"]),
        "source": str(sample_meta["source"]),
        "writer_id": str(sample_meta["writer_id"]),
        "text": str(sample_meta["text"]),
        "word_index": word_index,
        "char_index": int(char_entry["char_index"]),
        "char": target_char,
        "word": word_text_map.get(word_index, ""),
        "eligible": True,
        "category": "ambiguous",
        "ambiguity_reason": None,
        "component_count": int(len(components)),
        "draw_component_count": int(sum(1 for component in components if not component.is_orphan and component.indices.size >= 2)),
        "component_spans": [
            {
                "start_index": int(component.start_index),
                "end_index": int(component.end_index),
                "width": float(component.width),
                "height": float(component.height),
                "path_length": float(component.path_length),
                "stroke_id": component.stroke_id,
                "is_orphan": bool(component.is_orphan),
            }
            for component in components
        ],
        "base_span": None,
        "mark_span": None,
        "point_index_gap": None,
        "stroke_index_gap": None,
        "timestamp_gap": None,
        "intervening_characters": None,
        "intervening_same_word_characters": None,
        "pair_resolution": resolution,
        "word_point_count": int(word_indices.size),
    }

    if base_component is None or mark_component is None:
        if resolution is None:
            if result["draw_component_count"] < 2:
                result["ambiguity_reason"] = "insufficient_draw_components"
            else:
                result["ambiguity_reason"] = "no_valid_base_mark_pair"
        else:
            result["ambiguity_reason"] = str(resolution.get("reason") or "unresolved")
        return result

    base_end = int(base_component.end_index)
    mark_start = int(mark_component.start_index)
    stroke_index_gap = None
    if base_component.stroke_id is not None and mark_component.stroke_id is not None:
        stroke_index_gap = int(mark_component.stroke_id - base_component.stroke_id)
    intervening_characters = 0
    intervening_same_word_characters = 0
    for other in all_char_entries:
        if int(other["word_index"]) == word_index and int(other["char_index"]) == int(char_entry["char_index"]):
            continue
        if int(other["start_index"]) > base_end and int(other["start_index"]) < mark_start:
            intervening_characters += 1
            if int(other["word_index"]) == word_index:
                intervening_same_word_characters += 1

    category = categorize_delay(intervening_characters, stroke_index_gap)
    result.update(
        {
            "category": category,
            "base_span": {
                "start_index": int(base_component.start_index),
                "end_index": int(base_component.end_index),
                "stroke_id": base_component.stroke_id,
                "width": float(base_component.width),
                "height": float(base_component.height),
                "path_length": float(base_component.path_length),
            },
            "mark_span": {
                "start_index": int(mark_component.start_index),
                "end_index": int(mark_component.end_index),
                "stroke_id": mark_component.stroke_id,
                "width": float(mark_component.width),
                "height": float(mark_component.height),
                "path_length": float(mark_component.path_length),
            },
            "point_index_gap": int(mark_start - base_end),
            "stroke_index_gap": stroke_index_gap,
            "timestamp_gap": float(timestamps[mark_start] - timestamps[base_end]) if timestamps.size else None,
            "intervening_characters": int(intervening_characters),
            "intervening_same_word_characters": int(intervening_same_word_characters),
            "ambiguity_reason": None,
        }
    )
    return result


def plot_word_time_segments(ax, points_abs: np.ndarray, events: np.ndarray, indices: np.ndarray) -> None:
    if indices.size < 2:
        return
    sorted_indices = np.asarray(sorted(set(int(value) for value in indices.tolist())), dtype=np.int64)
    draw_segments: List[Tuple[np.ndarray, float]] = []
    move_segments: List[np.ndarray] = []
    for prev, current in zip(sorted_indices[:-1], sorted_indices[1:]):
        if int(current) != int(prev) + 1:
            continue
        segment = points_abs[[int(prev), int(current)]]
        if int(events[int(prev)]) == 2:
            move_segments.append(segment)
            continue
        if int(events[int(current)]) in DRAW_EVENTS:
            draw_segments.append((segment, float(current)))
        else:
            move_segments.append(segment)

    for segment in move_segments:
        ax.plot(segment[:, 0], -segment[:, 1], color="0.85", linewidth=0.8, linestyle="--", alpha=0.9)

    if not draw_segments:
        return
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    cmap = plt.get_cmap("viridis")
    max_order = max(order for _, order in draw_segments)
    min_order = min(order for _, order in draw_segments)
    scale = max(max_order - min_order, 1.0)
    for segment, order in draw_segments:
        ax.plot(
            segment[:, 0],
            -segment[:, 1],
            color=cmap((order - min_order) / scale),
            linewidth=1.3,
            alpha=0.95,
        )


def plot_highlight_component(ax, points_abs: np.ndarray, events: np.ndarray, component: Optional[DrawComponent], color: str, label: str) -> None:
    if component is None or component.indices.size < 2:
        return
    for prev, current in zip(component.indices[:-1], component.indices[1:]):
        if int(current) != int(prev) + 1:
            continue
        if int(events[int(prev)]) == 2 or int(events[int(current)]) not in DRAW_EVENTS:
            continue
        segment = points_abs[[int(prev), int(current)]]
        ax.plot(segment[:, 0], -segment[:, 1], color=color, linewidth=2.6, alpha=0.95, label=label)
        label = "_nolegend_"


def save_occurrence_plot(
    *,
    output_path: Path,
    occurrence: dict,
    points_abs: np.ndarray,
    events: np.ndarray,
    word_indices: np.ndarray,
    base_component: Optional[DrawComponent],
    mark_component: Optional[DrawComponent],
) -> Optional[str]:
    plt = maybe_import_matplotlib()
    if plt is None:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    plot_word_time_segments(ax, points_abs, events, word_indices)
    plot_highlight_component(ax, points_abs, events, base_component, "tab:orange", "base")
    plot_highlight_component(ax, points_abs, events, mark_component, "tab:red", "mark")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    title = (
        f"{occurrence['char']} in \"{occurrence['word']}\" | {occurrence['category']} | "
        f"writer={occurrence['writer_id']} | split={occurrence['split']}"
    )
    subtitle = (
        f"gap_pts={occurrence['point_index_gap']} gap_strokes={occurrence['stroke_index_gap']} "
        f"gap_time={occurrence['timestamp_gap']} intervening_chars={occurrence['intervening_characters']}"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path.as_posix()


def save_category_grid(path: Path, entries: Sequence[dict]) -> Optional[str]:
    if not entries:
        return None
    plt = maybe_import_matplotlib()
    if plt is None:
        return None
    cols = 2
    rows = int(math.ceil(len(entries) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.4, rows * 3.2), squeeze=False)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for ax, entry in zip(axes.reshape(-1), entries):
        image = plt.imread(entry["plot_path"])
        ax.imshow(image)
        ax.set_title(
            f"{entry['char']} | {entry['category']} | {entry['word']}\n"
            f"gap_pts={entry['point_index_gap']} gap_chars={entry['intervening_characters']}",
            fontsize=8,
        )
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.as_posix()


def materialize_occurrence_plot(
    *,
    occurrence: dict,
    rows: Sequence[dict],
    arrays: Dict[str, np.ndarray],
    example_root: Path,
) -> Optional[str]:
    row_index = int(occurrence["row_index"])
    row = rows[row_index]
    points_abs = np.asarray(arrays["points_abs"][row_index], dtype=np.float32)
    timestamps = np.asarray(arrays["timestamps"][row_index], dtype=np.float64)
    events = np.asarray(arrays["events"][row_index], dtype=np.int64)
    point_to_stroke = build_global_stroke_ids(events)
    char_entry = None
    for entry in row["char_segments"]:
        if (
            int(entry["word_index"]) == int(occurrence["word_index"])
            and int(entry["char_index"]) == int(occurrence["char_index"])
            and str(entry.get("char") or "").lower() == str(occurrence["char"])
        ):
            char_entry = entry
            break
    if char_entry is None:
        return None

    char_indices = flatten_ranges(char_entry.get("ranges") or [])
    components = build_draw_components(
        char_indices,
        points_abs,
        timestamps,
        events,
        point_to_stroke=point_to_stroke,
    )
    base_component = None
    mark_component = None
    for component in components:
        base_span = occurrence.get("base_span")
        mark_span = occurrence.get("mark_span")
        if base_span and component.start_index == int(base_span["start_index"]) and component.end_index == int(base_span["end_index"]):
            base_component = component
        if mark_span and component.start_index == int(mark_span["start_index"]) and component.end_index == int(mark_span["end_index"]):
            mark_component = component

    file_stem = (
        f"{occurrence['char']}__{occurrence['category']}__{slugify_text(occurrence['word'])}"
        f"__{slugify_text(occurrence['sample_id'])}__w{occurrence['word_index']}_c{occurrence['char_index']}"
    )
    plot_path = example_root / occurrence["char"] / occurrence["category"] / f"{file_stem}.png"
    word_indices = collect_word_indices(row["word_segments"], int(occurrence["word_index"]))
    return save_occurrence_plot(
        output_path=plot_path,
        occurrence=occurrence,
        points_abs=points_abs,
        events=events,
        word_indices=word_indices,
        base_component=base_component,
        mark_component=mark_component,
    )


def summarize_character(entries: Sequence[dict]) -> dict:
    total = len(entries)
    resolved = [entry for entry in entries if entry["category"] != "ambiguous"]
    category_counts = Counter(entry["category"] for entry in entries)
    ambiguity_counts = Counter(entry["ambiguity_reason"] for entry in entries if entry["category"] == "ambiguous")
    point_gap_values = [float(entry["point_index_gap"]) for entry in resolved if entry["point_index_gap"] is not None]
    stroke_gap_values = [float(entry["stroke_index_gap"]) for entry in resolved if entry["stroke_index_gap"] is not None]
    timestamp_gap_values = [float(entry["timestamp_gap"]) for entry in resolved if entry["timestamp_gap"] is not None]
    intervening_char_values = [float(entry["intervening_characters"]) for entry in resolved if entry["intervening_characters"] is not None]
    return {
        "total_occurrences": int(total),
        "resolved_occurrences": int(len(resolved)),
        "resolved_fraction": float(len(resolved) / total) if total else None,
        "category_counts": {category: int(category_counts.get(category, 0)) for category in DELAY_CATEGORIES},
        "category_percentages": {
            category: float(category_counts.get(category, 0) / total) if total else 0.0
            for category in DELAY_CATEGORIES
        },
        "ambiguity_reason_counts": {str(key): int(value) for key, value in sorted(ambiguity_counts.items())},
        "point_gap_stats": safe_stats(point_gap_values),
        "stroke_gap_stats": safe_stats(stroke_gap_values),
        "timestamp_gap_stats": safe_stats(timestamp_gap_values),
        "intervening_character_stats": safe_stats(intervening_char_values),
        "point_gap_buckets": bucket_counts(
            point_gap_values,
            edges=[0, 11, 31, 101, float("inf")],
            labels=["0_10", "11_30", "31_100", "101_plus"],
        ),
        "intervening_character_buckets": bucket_counts(
            intervening_char_values,
            edges=[0, 1, 3, 6, float("inf")],
            labels=["0", "1_2", "3_5", "6_plus"],
        ),
    }


def choose_examples(entries: Sequence[dict], category: str, limit: int) -> List[dict]:
    subset = [entry for entry in entries if entry["category"] == category]
    if category == "immediate":
        subset.sort(key=lambda entry: (entry.get("intervening_characters", 9999), entry.get("point_index_gap", 9999)))
    elif category == "delayed":
        subset.sort(
            key=lambda entry: (
                -(entry.get("intervening_characters") or -1),
                -(entry.get("point_index_gap") or -1),
            )
        )
    elif category == "nearby":
        subset.sort(key=lambda entry: (entry.get("intervening_characters", 9999), entry.get("point_index_gap", 9999)))
    else:
        subset.sort(key=lambda entry: (entry.get("ambiguity_reason") or "", -(entry.get("component_count") or 0)))
    return subset[:limit]


def write_summary_md(path: Path, summary: dict) -> None:
    source_label = str(summary.get("source_label") or "Dataset")
    lines = [
        f"# {source_label} Dot/Cross Timing Audit",
        "",
        f"- Canonical samples scanned: `{summary['num_samples_scanned']}`",
        f"- {source_label} samples scanned: `{summary['num_source_samples']}`",
        f"- Char-valid {source_label} samples used: `{summary['num_char_valid_samples']}`",
        f"- Eligible occurrences audited: `{summary['eligible_occurrences']}`",
        "",
    ]
    overall = summary["overall"]
    lines.extend(
        [
            "## Overall",
            "",
            f"- Immediate: `{overall['category_counts']['immediate']}`",
            f"- Nearby: `{overall['category_counts']['nearby']}`",
            f"- Delayed: `{overall['category_counts']['delayed']}`",
            f"- Ambiguous: `{overall['category_counts']['ambiguous']}`",
            "",
        ]
    )
    for target_char in TARGET_CHARS:
        char_summary = summary["by_char"][target_char]
        lines.extend(
            [
                f"## `{target_char}`",
                "",
                f"- Total occurrences: `{char_summary['total_occurrences']}`",
                f"- Resolved occurrences: `{char_summary['resolved_occurrences']}`",
                f"- Immediate: `{char_summary['category_counts']['immediate']}` ({char_summary['category_percentages']['immediate']:.3f})",
                f"- Nearby: `{char_summary['category_counts']['nearby']}` ({char_summary['category_percentages']['nearby']:.3f})",
                f"- Delayed: `{char_summary['category_counts']['delayed']}` ({char_summary['category_percentages']['delayed']:.3f})",
                f"- Ambiguous: `{char_summary['category_counts']['ambiguous']}` ({char_summary['category_percentages']['ambiguous']:.3f})",
                f"- Median point gap: `{char_summary['point_gap_stats']['median']}`",
                f"- Median intervening chars: `{char_summary['intervening_character_stats']['median']}`",
                f"- Common ambiguity reasons: `{char_summary['ambiguity_reason_counts']}`",
                "",
            ]
        )
    interpretation = summary.get("interpretation") or {}
    if interpretation:
        lines.extend(
            [
                "## Interpretation",
                "",
                f"- Primary verdict: `{interpretation.get('verdict')}`",
                f"- Local-convention share among resolved occurrences: `{interpretation.get('resolved_local_fraction')}`",
                f"- Delayed share among resolved occurrences: `{interpretation.get('resolved_delayed_fraction')}`",
                f"- Reordering recommendation: `{interpretation.get('reordering_recommendation')}`",
                "",
            ]
        )
    path.write_text("\n".join(lines))


def build_interpretation(summary: dict) -> dict:
    source_label = str(summary.get("source_label") or "Dataset")
    overall = summary["overall"]
    resolved_total = overall["resolved_occurrences"]
    immediate = overall["category_counts"]["immediate"]
    delayed = overall["category_counts"]["delayed"]
    nearby = overall["category_counts"]["nearby"]
    local_fraction = float((immediate + nearby) / resolved_total) if resolved_total else None
    delayed_fraction = float(delayed / resolved_total) if resolved_total else None

    if resolved_total == 0:
        verdict = "insufficient_resolved_cases"
        recommendation = "no_claim"
    elif delayed_fraction is not None and delayed_fraction <= 0.10:
        verdict = f"{source_label} mostly follows the local dot/cross convention"
        recommendation = "likely unnecessary unless later modeling still struggles on dot/cross timing"
    elif delayed_fraction is not None and delayed_fraction <= 0.25:
        verdict = f"{source_label} often follows the local convention, but delayed marks are not negligible"
        recommendation = "worth considering only if dot/cross timing becomes a clear failure mode"
    else:
        verdict = "Delayed dot/cross marks are common enough to matter"
        recommendation = "dataset reordering is a serious candidate"

    return {
        "verdict": verdict,
        "resolved_local_fraction": local_fraction,
        "resolved_delayed_fraction": delayed_fraction,
        "reordering_recommendation": recommendation,
        "resolved_total": int(resolved_total),
        "immediate_plus_nearby": int(immediate + nearby),
        "delayed": int(delayed),
    }


def run_dot_cross_timing_audit(
    *,
    canonical_jsonl: Path,
    canonical_npz: Path,
    out_dir: Path,
    source_filter: str = "deepwriting",
    source_label: str = "DeepWriting",
    max_examples_per_category: int = 6,
    limit_samples: Optional[int] = None,
) -> dict:
    rows = [json.loads(line) for line in canonical_jsonl.open()]
    npz = np.load(canonical_npz, allow_pickle=True)
    for key in ("sample_id", "points_abs", "timestamps", "events"):
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

    out_dir.mkdir(parents=True, exist_ok=True)
    example_root = out_dir / "examples"
    grid_root = out_dir / "grids"
    occurrences_jsonl = out_dir / "occurrences.jsonl"

    sample_count = 0
    source_count = 0
    char_valid_count = 0
    all_results: List[dict] = []

    for row_index, row in enumerate(rows):
        sample_count += 1
        if limit_samples is not None and sample_count > limit_samples:
            break
        if str(row["source"]) != str(source_filter):
            continue
        source_count += 1
        if not bool(row.get("is_char_segmentation_valid")):
            continue
        char_valid_count += 1
        if char_valid_count % 250 == 0:
            print(f"[audit] processed {char_valid_count} {source_label} char-valid samples")

        if str(row["sample_id"]) != str(arrays["sample_id"][row_index]):
            raise ValueError(f"Sample mismatch at row {row_index}: {row['sample_id']} != {arrays['sample_id'][row_index]}")

        points_abs = np.asarray(arrays["points_abs"][row_index], dtype=np.float32)
        timestamps = np.asarray(arrays["timestamps"][row_index], dtype=np.float64)
        events = np.asarray(arrays["events"][row_index], dtype=np.int64)
        sample_meta = dict(row)
        word_text_map = compute_word_text_map(row["word_segments"])
        all_char_entries = build_char_entries(row["char_segments"])
        point_to_stroke = build_global_stroke_ids(events)

        for char_entry in row["char_segments"]:
            target_char = str(char_entry.get("char") or "").lower()
            if target_char not in TARGET_CHARS:
                continue
            occurrence = classify_occurrence(
                sample_meta=sample_meta,
                points_abs=points_abs,
                timestamps=timestamps,
                events=events,
                char_entry=char_entry,
                word_text_map=word_text_map,
                all_char_entries=all_char_entries,
                point_to_stroke=point_to_stroke,
            )
            occurrence["row_index"] = int(row_index)
            occurrence["plot_path"] = None
            all_results.append(occurrence)

    by_char: Dict[str, List[dict]] = {target_char: [] for target_char in TARGET_CHARS}
    for result in all_results:
        by_char[result["char"]].append(result)

    overall = summarize_character(all_results)
    char_summaries = {target_char: summarize_character(entries) for target_char, entries in by_char.items()}

    selected_examples = {}
    for target_char in TARGET_CHARS:
        selected_examples[target_char] = {}
        for category in DELAY_CATEGORIES:
            selected = choose_examples(by_char[target_char], category, max_examples_per_category)
            selected_examples[target_char][category] = selected
            for entry in selected:
                entry["plot_path"] = materialize_occurrence_plot(
                    occurrence=entry,
                    rows=rows,
                    arrays=arrays,
                    example_root=example_root,
                )
            if selected:
                grid_path = grid_root / f"{target_char}_{category}_grid.png"
                grid_file = save_category_grid(grid_path, selected)
                for entry in selected:
                    entry.setdefault("grid_paths", []).append(grid_file)

    occurrences_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with occurrences_jsonl.open("w") as handle:
        for occurrence in all_results:
            handle.write(json.dumps(occurrence) + "\n")

    summary = {
        "canonical_jsonl": canonical_jsonl.as_posix(),
        "canonical_npz": canonical_npz.as_posix(),
        "out_dir": out_dir.as_posix(),
        "source_filter": str(source_filter),
        "source_label": str(source_label),
        "num_samples_scanned": int(sample_count if limit_samples is None else min(sample_count, len(rows))),
        "num_source_samples": int(source_count),
        "num_char_valid_samples": int(char_valid_count),
        "eligible_occurrences": int(len(all_results)),
        "overall": overall,
        "by_char": char_summaries,
        "selected_examples": {
            target_char: {
                category: [
                    {
                        "sample_id": entry["sample_id"],
                        "word": entry["word"],
                        "plot_path": entry.get("plot_path"),
                        "point_index_gap": entry.get("point_index_gap"),
                        "intervening_characters": entry.get("intervening_characters"),
                        "category": entry["category"],
                    }
                    for entry in selected_examples[target_char][category]
                ]
                for category in DELAY_CATEGORIES
            }
            for target_char in TARGET_CHARS
        },
    }
    summary["interpretation"] = build_interpretation(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_summary_md(out_dir / "summary.md", summary)
    return summary


def build_parser(bundle_root: Path) -> argparse.ArgumentParser:
    project_root = bundle_root.parent
    default_canonical_dir = project_root / "Version1A" / "Data" / "processed" / "canonical_raw"
    parser = argparse.ArgumentParser(description="Audit i/j/t dot-cross timing using canonical segmentation metadata.")
    parser.add_argument(
        "--canonical-jsonl",
        type=Path,
        default=default_canonical_dir / "canonical_samples.jsonl",
        help="Canonical JSONL manifest with char segmentation metadata.",
    )
    parser.add_argument(
        "--canonical-npz",
        type=Path,
        default=default_canonical_dir / "canonical_samples.npz",
        help="Canonical NPZ with aligned points/timestamps/events arrays.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=bundle_root / "audits" / "dot_cross_timing",
        help="Output directory for audit summaries and example plots.",
    )
    parser.add_argument("--source-filter", type=str, default="deepwriting", help="Source name to audit from canonical rows.")
    parser.add_argument("--source-label", type=str, default="DeepWriting", help="Human-readable source label for reports.")
    parser.add_argument(
        "--max-examples-per-category",
        type=int,
        default=6,
        help="Maximum number of selected example plots per character/category grid.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional limit for debugging the audit on a smaller subset.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    bundle_root = Path(__file__).resolve().parents[1]
    parser = build_parser(bundle_root)
    args = parser.parse_args(argv)
    summary = run_dot_cross_timing_audit(
        canonical_jsonl=args.canonical_jsonl.expanduser().resolve(),
        canonical_npz=args.canonical_npz.expanduser().resolve(),
        out_dir=args.out_dir.expanduser().resolve(),
        source_filter=str(args.source_filter),
        source_label=str(args.source_label),
        max_examples_per_category=int(args.max_examples_per_category),
        limit_samples=args.limit_samples,
    )
    interpretation = summary.get("interpretation") or {}
    print(f"[audit] wrote {Path(summary['out_dir']).as_posix()}")
    print(f"[audit] verdict: {interpretation.get('verdict')}")
    print(
        f"[audit] resolved local fraction={interpretation.get('resolved_local_fraction')} "
        f"delayed fraction={interpretation.get('resolved_delayed_fraction')}"
    )
    return 0
