from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .data import ProcessedWordSplit
from .generation import offsets_to_absolute_points, plot_points
from .utils import load_json_dict, maybe_import_matplotlib


EPS = 1e-6

BASE_STYLE_FEATURE_KEYS: Tuple[str, ...] = (
    "offset_count",
    "draw_segment_count",
    "relocation_count",
    "draw_run_count",
    "draw_fraction",
    "pen_lift_ratio",
    "mean_draw_run_length",
    "mean_draw_length_norm",
    "median_draw_length_norm",
    "mean_relocation_length_norm",
    "relocation_to_draw_ratio",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "ink_density",
    "normalized_draw_path_length",
    "horizontal_motion_ratio",
    "vertical_motion_ratio",
    "path_efficiency",
    "baseline_slope",
    "slant_proxy",
    "mean_abs_turn",
    "sharp_turn_ratio",
)


@dataclass(frozen=True)
class StyleClusterMap:
    writer_to_cluster_id: Dict[str, int]
    cluster_ids: List[int]
    chosen_k: int
    train_writer_ids: List[str]
    feature_columns: List[str]
    split_stats: Dict[str, dict]
    cluster_stats: Dict[str, dict]
    fit_summary_path: Optional[str]

    @property
    def num_clusters(self) -> int:
        return len(self.cluster_ids)

    def encode(self, raw_writer_id: str) -> int:
        writer_id = str(raw_writer_id)
        if writer_id not in self.writer_to_cluster_id:
            raise ValueError(f"Writer {writer_id} missing from style cluster map")
        return int(self.writer_to_cluster_id[writer_id])


def load_style_cluster_map(path: Path) -> StyleClusterMap:
    payload = load_json_dict(path)
    writer_to_cluster_id = {
        str(key): int(value) for key, value in payload.get("writer_to_cluster_id", {}).items()
    }
    cluster_ids = [int(value) for value in payload.get("cluster_ids", [])]
    train_writer_ids = [str(value) for value in payload.get("train_writer_ids", [])]
    feature_columns = [str(value) for value in payload.get("feature_columns", [])]
    split_stats = {str(key): dict(value) for key, value in payload.get("split_stats", {}).items()}
    cluster_stats = {str(key): dict(value) for key, value in payload.get("cluster_stats", {}).items()}
    return StyleClusterMap(
        writer_to_cluster_id=writer_to_cluster_id,
        cluster_ids=cluster_ids,
        chosen_k=int(payload.get("chosen_k", len(cluster_ids))),
        train_writer_ids=train_writer_ids,
        feature_columns=feature_columns,
        split_stats=split_stats,
        cluster_stats=cluster_stats,
        fit_summary_path=payload.get("cluster_fit_summary_path"),
    )


def load_panel_clusters(path: Path, cluster_map: StyleClusterMap) -> List[dict]:
    payload = load_json_dict(path)
    clusters = payload.get("clusters", [])
    if not isinstance(clusters, list) or not clusters:
        raise ValueError(f"Expected a non-empty clusters list in {path}")
    valid_cluster_ids = set(cluster_map.cluster_ids)
    resolved: List[dict] = []
    for item in clusters:
        if not isinstance(item, dict):
            raise ValueError(f"Cluster panel entries must be objects in {path}")
        cluster_id = int(item.get("cluster_id"))
        if cluster_id not in valid_cluster_ids:
            raise ValueError(f"Panel cluster_id={cluster_id} is not present in {path}")
        label = str(item.get("label") or f"cluster_{cluster_id:02d}")
        resolved.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "train_writer_count": int(item.get("train_writer_count", 0)),
                "train_sample_count": int(item.get("train_sample_count", 0)),
                "representative_writer_ids": [str(value) for value in item.get("representative_writer_ids", [])],
            }
        )
    return resolved


def compute_sample_style_features(offsets: np.ndarray) -> Dict[str, float]:
    offsets = np.asarray(offsets, dtype=np.float32)
    if offsets.ndim != 2 or offsets.shape[0] == 0 or offsets.shape[1] < 3:
        return {key: 0.0 for key in BASE_STYLE_FEATURE_KEYS}

    draw_mask = offsets[:, 2] >= 0.5
    draw_offsets = offsets[draw_mask, :2]
    relocation_offsets = offsets[~draw_mask, :2]
    draw_lengths = np.linalg.norm(draw_offsets, axis=1) if draw_offsets.size else np.zeros((0,), dtype=np.float32)
    relocation_lengths = (
        np.linalg.norm(relocation_offsets, axis=1) if relocation_offsets.size else np.zeros((0,), dtype=np.float32)
    )

    points = offsets_to_absolute_points(offsets)
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    bbox_width = float(np.ptp(x_coords)) if x_coords.size else 0.0
    bbox_height = float(np.ptp(y_coords)) if y_coords.size else 0.0
    bbox_diag = float(math.sqrt(bbox_width * bbox_width + bbox_height * bbox_height) + EPS)
    bbox_area = float(max(bbox_width * bbox_height, EPS))

    draw_path_length = float(np.sum(draw_lengths)) if draw_lengths.size else 0.0
    relocation_path_length = float(np.sum(relocation_lengths)) if relocation_lengths.size else 0.0
    offset_count = int(offsets.shape[0])
    draw_segment_count = int(np.sum(draw_mask))
    relocation_count = int(offset_count - draw_segment_count)

    draw_run_lengths = _extract_draw_run_lengths(draw_mask)
    draw_run_count = len(draw_run_lengths)
    mean_draw_run_length = float(np.mean(draw_run_lengths)) if draw_run_lengths else 0.0
    draw_fraction = float(draw_segment_count / max(offset_count, 1))
    pen_lift_ratio = float(relocation_count / max(draw_segment_count, 1))

    draw_dx = np.abs(draw_offsets[:, 0]) if draw_offsets.size else np.zeros((0,), dtype=np.float32)
    draw_dy = np.abs(draw_offsets[:, 1]) if draw_offsets.size else np.zeros((0,), dtype=np.float32)
    horizontal_motion_ratio = float(np.sum(draw_dx) / max(draw_path_length, EPS))
    vertical_motion_ratio = float(np.sum(draw_dy) / max(draw_path_length, EPS))

    normalized_draw_path_length = float(draw_path_length / bbox_diag)
    mean_draw_length_norm = float(np.mean(draw_lengths) / bbox_diag) if draw_lengths.size else 0.0
    median_draw_length_norm = float(np.median(draw_lengths) / bbox_diag) if draw_lengths.size else 0.0
    mean_relocation_length_norm = (
        float(np.mean(relocation_lengths) / bbox_diag) if relocation_lengths.size else 0.0
    )
    relocation_to_draw_ratio = float(relocation_path_length / max(draw_path_length, EPS))
    aspect_ratio = float(bbox_width / max(bbox_height, EPS))
    ink_density = float(draw_path_length / bbox_area)
    path_efficiency = _compute_path_efficiency(points, draw_mask, draw_path_length)
    baseline_slope = _compute_baseline_slope(points, draw_mask)
    slant_proxy = _compute_slant_proxy(draw_offsets)
    mean_abs_turn, sharp_turn_ratio = _compute_turning_features(offsets)

    return {
        "offset_count": float(offset_count),
        "draw_segment_count": float(draw_segment_count),
        "relocation_count": float(relocation_count),
        "draw_run_count": float(draw_run_count),
        "draw_fraction": draw_fraction,
        "pen_lift_ratio": pen_lift_ratio,
        "mean_draw_run_length": mean_draw_run_length,
        "mean_draw_length_norm": mean_draw_length_norm,
        "median_draw_length_norm": median_draw_length_norm,
        "mean_relocation_length_norm": mean_relocation_length_norm,
        "relocation_to_draw_ratio": relocation_to_draw_ratio,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "aspect_ratio": aspect_ratio,
        "ink_density": ink_density,
        "normalized_draw_path_length": normalized_draw_path_length,
        "horizontal_motion_ratio": horizontal_motion_ratio,
        "vertical_motion_ratio": vertical_motion_ratio,
        "path_efficiency": path_efficiency,
        "baseline_slope": baseline_slope,
        "slant_proxy": slant_proxy,
        "mean_abs_turn": mean_abs_turn,
        "sharp_turn_ratio": sharp_turn_ratio,
    }


def build_style_cluster_artifacts(
    *,
    split_map: Mapping[str, ProcessedWordSplit],
    output_root: Path,
    k_values: Sequence[int],
    seed: int,
    representative_cluster_count: int = 3,
    examples_per_cluster: int = 9,
    pca_max_components: int = 8,
) -> dict:
    if "train" not in split_map:
        raise ValueError("Style clustering requires a train split")

    output_root.mkdir(parents=True, exist_ok=True)
    bundle_root = output_root.parent.parent if len(output_root.parents) >= 2 else output_root.parent

    def relpath(path: Path) -> str:
        try:
            return path.resolve().relative_to(bundle_root.resolve()).as_posix()
        except Exception:
            return path.as_posix()

    train_split = split_map["train"]
    all_writer_rows: List[dict] = []
    split_to_writer_rows: Dict[str, List[dict]] = {}

    for split_name, split in split_map.items():
        rows = build_writer_feature_rows(split, split_name=split_name)
        split_to_writer_rows[str(split_name)] = rows
        all_writer_rows.extend(rows)

    if not split_to_writer_rows["train"]:
        raise ValueError("No train-writer rows were available for style clustering")

    feature_columns = [key for key in split_to_writer_rows["train"][0].keys() if key.startswith("feat__")]
    train_matrix = np.asarray(
        [[float(row[column]) for column in feature_columns] for row in split_to_writer_rows["train"]],
        dtype=np.float64,
    )
    scaler = fit_robust_scaler(train_matrix)
    train_scaled = transform_with_scaler(train_matrix, scaler)
    pca_model = fit_pca(train_scaled, max_components=pca_max_components)
    train_projected = transform_with_pca(train_scaled, pca_model)

    candidate_results = []
    valid_k_values = sorted({int(value) for value in k_values if int(value) >= 2 and int(value) < train_projected.shape[0]})
    if not valid_k_values:
        raise ValueError("Need at least one valid k value for style clustering")
    for k in valid_k_values:
        fit = run_kmeans(train_projected, k=k, seed=seed, n_init=24, max_iter=200)
        silhouette = silhouette_score(train_projected, fit["labels"])
        candidate_results.append(
            {
                "k": int(k),
                "silhouette": float(silhouette),
                "inertia": float(fit["inertia"]),
                "labels": fit["labels"],
                "centroids": fit["centroids"],
            }
        )

    candidate_results.sort(key=lambda item: (-item["silhouette"], item["inertia"], item["k"]))
    chosen = candidate_results[0]
    chosen_k = int(chosen["k"])
    train_labels = np.asarray(chosen["labels"], dtype=np.int64)
    centroids = np.asarray(chosen["centroids"], dtype=np.float64)

    split_assignments: Dict[str, Dict[str, int]] = {}
    writer_feature_projection_rows: List[dict] = []
    for split_name, rows in split_to_writer_rows.items():
        matrix = np.asarray([[float(row[column]) for column in feature_columns] for row in rows], dtype=np.float64)
        scaled = transform_with_scaler(matrix, scaler)
        projected = transform_with_pca(scaled, pca_model)
        if split_name == "train":
            labels = train_labels
        else:
            labels = assign_to_nearest_centroids(projected, centroids)
        split_assignments[split_name] = {}
        for row, coords, label in zip(rows, projected, labels):
            writer_id = str(row["writer_id"])
            cluster_id = int(label)
            split_assignments[split_name][writer_id] = cluster_id
            row["assigned_cluster_id"] = cluster_id
            row["pca_x"] = float(coords[0]) if coords.size >= 1 else 0.0
            row["pca_y"] = float(coords[1]) if coords.size >= 2 else 0.0
            writer_feature_projection_rows.append(row)

    writer_to_cluster = {
        writer_id: cluster_id
        for split_name in split_assignments
        for writer_id, cluster_id in split_assignments[split_name].items()
    }

    split_stats, cluster_stats = summarize_cluster_assignments(split_map=split_map, split_assignments=split_assignments)
    train_cluster_sample_counts = split_stats["train"]["sample_count_by_cluster"]
    train_cluster_writer_counts = split_stats["train"]["writer_count_by_cluster"]
    top_writers_by_cluster = build_top_writers_by_cluster(split_to_writer_rows["train"], train_labels)

    writer_features_path = output_root / "writer_style_features.csv"
    writer_features_train_path = output_root / "writer_style_features_train.csv"
    write_csv(writer_features_path, writer_feature_projection_rows)
    write_csv(writer_features_train_path, split_to_writer_rows["train"])

    cluster_centroids_path = output_root / "cluster_centroids.json"
    cluster_centroids_payload = {
        "chosen_k": chosen_k,
        "pca_components": int(pca_model["components"].shape[0]),
        "centroids_pca_space": centroids.tolist(),
        "cluster_ids": list(range(chosen_k)),
        "cluster_sample_count_train": train_cluster_sample_counts,
        "cluster_writer_count_train": train_cluster_writer_counts,
    }
    cluster_centroids_path.write_text(json.dumps(cluster_centroids_payload, indent=2))

    writer_to_cluster_map_path = output_root / "writer_to_cluster_map.json"
    writer_to_cluster_payload = {
        "name": "paper_local_style_cluster_map",
        "feature_source": "filtered_processed_deepwriting_local_only_offsets",
        "seed": int(seed),
        "chosen_k": chosen_k,
        "cluster_ids": list(range(chosen_k)),
        "feature_columns": feature_columns,
        "train_writer_ids": [str(row["writer_id"]) for row in split_to_writer_rows["train"]],
        "writer_to_cluster_id": writer_to_cluster,
        "split_assignments": split_assignments,
        "split_stats": split_stats,
        "cluster_stats": cluster_stats,
        "scaler": {
            "center": scaler["center"].tolist(),
            "scale": scaler["scale"].tolist(),
            "feature_columns": feature_columns,
        },
        "pca": {
            "mean": pca_model["mean"].tolist(),
            "components": pca_model["components"].tolist(),
            "explained_variance_ratio": pca_model["explained_variance_ratio"].tolist(),
        },
        "assignment_method": "nearest_train_cluster_centroid_in_pca_space",
        "cluster_centroids_path": relpath(cluster_centroids_path),
        "cluster_fit_summary_path": relpath(output_root / "cluster_fit_summary.json"),
        "writer_style_features_path": relpath(writer_features_path),
    }
    writer_to_cluster_map_path.write_text(json.dumps(writer_to_cluster_payload, indent=2))

    representative_cluster_count = max(1, int(representative_cluster_count))
    representative_clusters = [
        {
            "cluster_id": int(cluster_id),
            "label": f"cluster_{int(cluster_id):02d}",
            "train_sample_count": int(train_cluster_sample_counts.get(str(cluster_id), 0)),
            "train_writer_count": int(train_cluster_writer_counts.get(str(cluster_id), 0)),
            "representative_writer_ids": top_writers_by_cluster.get(int(cluster_id), [])[:3],
        }
        for cluster_id, _ in sorted(
            [(int(key), int(value)) for key, value in train_cluster_sample_counts.items()],
            key=lambda item: (-item[1], item[0]),
        )[:representative_cluster_count]
    ]
    panel_clusters_path = output_root / "default_panel_clusters.json"
    panel_clusters_payload = {
        "name": "default_panel_clusters",
        "selection": "top_train_sample_count",
        "clusters": representative_clusters,
    }
    panel_clusters_path.write_text(json.dumps(panel_clusters_payload, indent=2))

    top_writers_rows = []
    for cluster_id, writer_ids in sorted(top_writers_by_cluster.items()):
        for rank, writer_id in enumerate(writer_ids, start=1):
            row = next(item for item in split_to_writer_rows["train"] if str(item["writer_id"]) == str(writer_id))
            top_writers_rows.append(
                {
                    "cluster_id": int(cluster_id),
                    "rank_within_cluster": int(rank),
                    "writer_id": str(writer_id),
                    "train_sample_count": int(row["sample_count"]),
                }
            )
    top_writers_path = output_root / "cluster_top_writers.csv"
    write_csv(top_writers_path, top_writers_rows)

    write_cluster_projection_plot(
        path=output_root / "cluster_pca_plot.png",
        rows=writer_feature_projection_rows,
        chosen_k=chosen_k,
    )
    write_cluster_example_grids(
        path_root=output_root / "cluster_examples",
        split=train_split,
        writer_to_cluster=split_assignments["train"],
        chosen_k=chosen_k,
        examples_per_cluster=examples_per_cluster,
    )

    cluster_fit_summary = {
        "seed": int(seed),
        "feature_source": "filtered_processed_deepwriting_local_only_offsets",
        "candidate_results": [
            {
                "k": int(item["k"]),
                "silhouette": float(item["silhouette"]),
                "inertia": float(item["inertia"]),
            }
            for item in sorted(candidate_results, key=lambda row: row["k"])
        ],
        "chosen_k": chosen_k,
        "selection_reason": "highest silhouette score on train-writer PCA embeddings",
        "feature_columns": feature_columns,
        "train_writer_count": int(len(split_to_writer_rows["train"])),
        "split_stats": split_stats,
        "cluster_stats": cluster_stats,
        "pca_components": int(pca_model["components"].shape[0]),
        "pca_explained_variance_ratio": pca_model["explained_variance_ratio"].tolist(),
        "writer_style_features_path": relpath(writer_features_path),
        "writer_style_features_train_path": relpath(writer_features_train_path),
        "writer_to_cluster_map_path": relpath(writer_to_cluster_map_path),
        "cluster_centroids_path": relpath(cluster_centroids_path),
        "default_panel_clusters_path": relpath(panel_clusters_path),
        "cluster_top_writers_path": relpath(top_writers_path),
    }
    (output_root / "cluster_fit_summary.json").write_text(json.dumps(cluster_fit_summary, indent=2))
    (output_root / "cluster_fit_summary.md").write_text(build_cluster_fit_summary_md(cluster_fit_summary))

    return cluster_fit_summary


def build_writer_feature_rows(split: ProcessedWordSplit, *, split_name: str) -> List[dict]:
    writer_to_features: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    writer_to_sample_rows: Dict[str, List[dict]] = defaultdict(list)

    for sample_id, writer_id, text, offsets in zip(split.sample_ids, split.writer_ids, split.texts, split.offsets):
        raw_writer_id = str(writer_id)
        features = compute_sample_style_features(np.asarray(offsets, dtype=np.float32))
        writer_to_features[raw_writer_id].append(features)
        writer_to_sample_rows[raw_writer_id].append(
            {
                "sample_id": str(sample_id),
                "text": str(text),
                "point_count": int(np.asarray(offsets).shape[0]),
            }
        )

    rows: List[dict] = []
    for writer_id in sorted(writer_to_features):
        feature_rows = writer_to_features[writer_id]
        row: Dict[str, object] = {
            "split": str(split_name),
            "writer_id": writer_id,
            "sample_count": int(len(feature_rows)),
        }
        for key in BASE_STYLE_FEATURE_KEYS:
            values = np.asarray([float(item[key]) for item in feature_rows], dtype=np.float64)
            row[f"feat__{key}__median"] = float(np.median(values))
            row[f"feat__{key}__iqr"] = float(np.percentile(values, 75) - np.percentile(values, 25))
        rows.append(row)
    return rows


def fit_robust_scaler(matrix: np.ndarray) -> dict:
    center = np.median(matrix, axis=0)
    q75 = np.percentile(matrix, 75, axis=0)
    q25 = np.percentile(matrix, 25, axis=0)
    scale = q75 - q25
    scale[np.abs(scale) < EPS] = 1.0
    return {"center": center.astype(np.float64), "scale": scale.astype(np.float64)}


def transform_with_scaler(matrix: np.ndarray, scaler: dict) -> np.ndarray:
    return (matrix - scaler["center"]) / scaler["scale"]


def fit_pca(matrix: np.ndarray, max_components: int) -> dict:
    max_components = max(1, min(int(max_components), matrix.shape[0], matrix.shape[1]))
    mean = np.mean(matrix, axis=0, keepdims=True)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:max_components]
    denom = max(matrix.shape[0] - 1, 1)
    explained_variance = (singular_values ** 2) / denom
    total_variance = float(np.sum(explained_variance)) if explained_variance.size else 0.0
    if total_variance <= 0.0:
        explained_variance_ratio = np.zeros((max_components,), dtype=np.float64)
    else:
        explained_variance_ratio = explained_variance[:max_components] / total_variance
    return {
        "mean": mean.astype(np.float64),
        "components": components.astype(np.float64),
        "explained_variance_ratio": explained_variance_ratio.astype(np.float64),
    }


def transform_with_pca(matrix: np.ndarray, pca_model: dict) -> np.ndarray:
    return (matrix - pca_model["mean"]) @ pca_model["components"].T


def run_kmeans(
    matrix: np.ndarray,
    *,
    k: int,
    seed: int,
    n_init: int,
    max_iter: int,
) -> dict:
    best_result = None
    for init_index in range(int(n_init)):
        rng = np.random.default_rng(int(seed) + init_index * 9_973 + int(k) * 101)
        centroids = initialize_kmeans_plus_plus(matrix, k=k, rng=rng)
        for _ in range(int(max_iter)):
            sq_dists = pairwise_sq_distances(matrix, centroids)
            labels = np.argmin(sq_dists, axis=1)
            new_centroids = centroids.copy()
            for cluster_id in range(k):
                mask = labels == cluster_id
                if np.any(mask):
                    new_centroids[cluster_id] = np.mean(matrix[mask], axis=0)
                else:
                    farthest_index = int(np.argmax(np.min(sq_dists, axis=1)))
                    new_centroids[cluster_id] = matrix[farthest_index]
            if np.allclose(new_centroids, centroids, atol=1e-6):
                centroids = new_centroids
                break
            centroids = new_centroids
        sq_dists = pairwise_sq_distances(matrix, centroids)
        labels = np.argmin(sq_dists, axis=1)
        inertia = float(np.sum(np.min(sq_dists, axis=1)))
        result = {"labels": labels.astype(np.int64), "centroids": centroids.astype(np.float64), "inertia": inertia}
        if best_result is None or inertia < best_result["inertia"]:
            best_result = result
    assert best_result is not None
    return best_result


def initialize_kmeans_plus_plus(matrix: np.ndarray, *, k: int, rng: np.random.Generator) -> np.ndarray:
    num_rows = matrix.shape[0]
    first_index = int(rng.integers(0, num_rows))
    centers = [matrix[first_index]]
    while len(centers) < k:
        current = np.asarray(centers, dtype=np.float64)
        sq_dists = pairwise_sq_distances(matrix, current)
        min_sq = np.min(sq_dists, axis=1)
        total = float(np.sum(min_sq))
        if total <= 0.0:
            candidate_index = int(rng.integers(0, num_rows))
        else:
            probs = min_sq / total
            candidate_index = int(rng.choice(num_rows, p=probs))
        centers.append(matrix[candidate_index])
    return np.asarray(centers, dtype=np.float64)


def pairwise_sq_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def assign_to_nearest_centroids(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    sq_dists = pairwise_sq_distances(matrix, centroids)
    return np.argmin(sq_dists, axis=1).astype(np.int64)


def silhouette_score(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique_labels = sorted(set(int(value) for value in labels.tolist()))
    if len(unique_labels) < 2:
        return -1.0
    dists = np.sqrt(pairwise_sq_distances(matrix, matrix))
    scores = []
    for index in range(matrix.shape[0]):
        cluster_id = int(labels[index])
        same_mask = labels == cluster_id
        same_indices = np.flatnonzero(same_mask)
        if same_indices.size <= 1:
            a_value = 0.0
        else:
            a_value = float(np.mean(dists[index, same_indices[same_indices != index]]))
        b_value = float("inf")
        for other_cluster in unique_labels:
            if other_cluster == cluster_id:
                continue
            other_indices = np.flatnonzero(labels == other_cluster)
            if other_indices.size == 0:
                continue
            b_value = min(b_value, float(np.mean(dists[index, other_indices])))
        denom = max(a_value, b_value, EPS)
        scores.append((b_value - a_value) / denom)
    return float(np.mean(scores))


def summarize_cluster_assignments(
    *,
    split_map: Mapping[str, ProcessedWordSplit],
    split_assignments: Mapping[str, Mapping[str, int]],
) -> Tuple[dict, dict]:
    split_stats: Dict[str, dict] = {}
    cluster_stats: Dict[str, dict] = {}
    all_cluster_ids = sorted(
        {int(cluster_id) for mapping in split_assignments.values() for cluster_id in mapping.values()}
    )
    for cluster_id in all_cluster_ids:
        cluster_stats[str(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "train_writer_count": 0,
            "train_sample_count": 0,
            "split_writer_count": {},
            "split_sample_count": {},
        }

    for split_name, split in split_map.items():
        writer_to_cluster = split_assignments[str(split_name)]
        writer_counter = Counter(str(writer_id) for writer_id in split.writer_ids)
        sample_count_by_cluster = Counter()
        writer_count_by_cluster = Counter()
        for writer_id, count in writer_counter.items():
            cluster_id = int(writer_to_cluster[writer_id])
            sample_count_by_cluster[str(cluster_id)] += int(count)
            writer_count_by_cluster[str(cluster_id)] += 1
        split_stats[str(split_name)] = {
            "sample_count": int(len(split.writer_ids)),
            "writer_count": int(len(writer_counter)),
            "sample_count_by_cluster": {str(key): int(value) for key, value in sorted(sample_count_by_cluster.items())},
            "writer_count_by_cluster": {str(key): int(value) for key, value in sorted(writer_count_by_cluster.items())},
        }
        for cluster_id in all_cluster_ids:
            cluster_key = str(cluster_id)
            cluster_stats[cluster_key]["split_writer_count"][str(split_name)] = int(writer_count_by_cluster.get(cluster_key, 0))
            cluster_stats[cluster_key]["split_sample_count"][str(split_name)] = int(sample_count_by_cluster.get(cluster_key, 0))
            if str(split_name) == "train":
                cluster_stats[cluster_key]["train_writer_count"] = int(writer_count_by_cluster.get(cluster_key, 0))
                cluster_stats[cluster_key]["train_sample_count"] = int(sample_count_by_cluster.get(cluster_key, 0))
    return split_stats, cluster_stats


def build_top_writers_by_cluster(rows: Sequence[dict], labels: np.ndarray) -> Dict[int, List[str]]:
    grouped: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for row, label in zip(rows, labels):
        grouped[int(label)].append((str(row["writer_id"]), int(row["sample_count"])))
    return {
        cluster_id: [writer_id for writer_id, _ in sorted(items, key=lambda item: (-item[1], item[0]))]
        for cluster_id, items in grouped.items()
    }


def write_cluster_projection_plot(*, path: Path, rows: Sequence[dict], chosen_k: int) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.cm.get_cmap("tab10", max(chosen_k, 3))
    for row in rows:
        cluster_id = int(row.get("assigned_cluster_id", 0))
        split_name = str(row.get("split", "train"))
        marker = "o" if split_name == "train" else ("s" if split_name == "val" else "^")
        ax.scatter(
            float(row.get("pca_x", 0.0)),
            float(row.get("pca_y", 0.0)),
            s=60,
            color=cmap(cluster_id % cmap.N),
            marker=marker,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.85,
        )
    ax.set_title("Writer Style Clusters (PCA projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_cluster_example_grids(
    *,
    path_root: Path,
    split: ProcessedWordSplit,
    writer_to_cluster: Mapping[str, int],
    chosen_k: int,
    examples_per_cluster: int,
) -> None:
    plt = maybe_import_matplotlib()
    if plt is None:
        return
    path_root.mkdir(parents=True, exist_ok=True)
    cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
    for index, writer_id in enumerate(split.writer_ids):
        cluster_to_indices[int(writer_to_cluster[str(writer_id)])].append(index)
    for cluster_id in range(chosen_k):
        indices = cluster_to_indices.get(cluster_id, [])[: max(1, int(examples_per_cluster))]
        if not indices:
            continue
        cols = 3
        rows = int(math.ceil(len(indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.6), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, index in zip(axes.ravel(), indices):
            offsets = np.asarray(split.offsets[index], dtype=np.float32)
            points = offsets_to_absolute_points(offsets)
            plot_points(ax, points, title="", color="tab:blue", alpha=0.95, linewidth=1.0)
            ax.set_title(f"{split.texts[index]}\nwriter {split.writer_ids[index]}", fontsize=8)
        fig.suptitle(f"Cluster {cluster_id:02d} real training examples", fontsize=12)
        fig.tight_layout()
        fig.savefig(path_root / f"cluster_{cluster_id:02d}_examples.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def build_cluster_fit_summary_md(summary: dict) -> str:
    lines = [
        "# Style Cluster Fit Summary",
        "",
        f"- Chosen k: `{summary['chosen_k']}`",
        f"- Selection reason: `{summary['selection_reason']}`",
        f"- Train writer count: `{summary['train_writer_count']}`",
        f"- PCA components: `{summary['pca_components']}`",
        "",
        "## Candidate k Results",
        "",
    ]
    for row in summary.get("candidate_results", []):
        lines.append(
            f"- k=`{row['k']}` silhouette=`{row['silhouette']:.4f}` inertia=`{row['inertia']:.4f}`"
        )
    lines.extend(["", "## Split Cluster Coverage", ""])
    for split_name, stats in summary.get("split_stats", {}).items():
        lines.append(
            f"- `{split_name}`: sample_count=`{stats['sample_count']}` writer_count=`{stats['writer_count']}`"
        )
        lines.append(
            f"  sample_count_by_cluster=`{stats['sample_count_by_cluster']}` "
            f"writer_count_by_cluster=`{stats['writer_count_by_cluster']}`"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- Feature table: `{summary['writer_style_features_path']}`",
            f"- Train feature table: `{summary['writer_style_features_train_path']}`",
            f"- Writer to cluster map: `{summary['writer_to_cluster_map_path']}`",
            f"- Cluster centroids: `{summary['cluster_centroids_path']}`",
            f"- Default panel clusters: `{summary['default_panel_clusters_path']}`",
            f"- Top writers by cluster: `{summary['cluster_top_writers_path']}`",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _extract_draw_run_lengths(draw_mask: np.ndarray) -> List[int]:
    lengths: List[int] = []
    current = 0
    for is_draw in draw_mask.tolist():
        if bool(is_draw):
            current += 1
        elif current > 0:
            lengths.append(current)
            current = 0
    if current > 0:
        lengths.append(current)
    return lengths


def _compute_path_efficiency(points: np.ndarray, draw_mask: np.ndarray, draw_path_length: float) -> float:
    if draw_path_length <= EPS or points.shape[0] <= 2:
        return 0.0
    draw_endpoints = points[1:][draw_mask]
    if draw_endpoints.shape[0] < 2:
        return 0.0
    start = draw_endpoints[0, :2]
    end = draw_endpoints[-1, :2]
    direct = float(np.linalg.norm(end - start))
    return float(direct / max(draw_path_length, EPS))


def _compute_baseline_slope(points: np.ndarray, draw_mask: np.ndarray) -> float:
    draw_points = points[1:][draw_mask]
    if draw_points.shape[0] < 3:
        return 0.0
    x_coords = draw_points[:, 0]
    y_coords = draw_points[:, 1]
    var_x = float(np.var(x_coords))
    if var_x <= EPS:
        return 0.0
    cov_xy = float(np.mean((x_coords - np.mean(x_coords)) * (y_coords - np.mean(y_coords))))
    return float(cov_xy / var_x)


def _compute_slant_proxy(draw_offsets: np.ndarray) -> float:
    if draw_offsets.size == 0:
        return 0.0
    dx = draw_offsets[:, 0]
    dy = draw_offsets[:, 1]
    mask = np.abs(dy) >= np.abs(dx)
    if not np.any(mask):
        mask = np.ones((draw_offsets.shape[0],), dtype=bool)
    selected_dx = dx[mask]
    selected_dy = dy[mask]
    return float(np.median(selected_dx / (np.abs(selected_dy) + 1e-3)))


def _compute_turning_features(offsets: np.ndarray) -> Tuple[float, float]:
    draw_mask = offsets[:, 2] >= 0.5
    abs_turns: List[float] = []
    sharp_turn_flags: List[float] = []
    current_run: List[np.ndarray] = []
    for row in offsets:
        if row[2] >= 0.5:
            current_run.append(row[:2])
        elif current_run:
            run_abs_turns, run_sharp_flags = _run_turn_features(np.asarray(current_run, dtype=np.float32))
            abs_turns.extend(run_abs_turns)
            sharp_turn_flags.extend(run_sharp_flags)
            current_run = []
    if current_run:
        run_abs_turns, run_sharp_flags = _run_turn_features(np.asarray(current_run, dtype=np.float32))
        abs_turns.extend(run_abs_turns)
        sharp_turn_flags.extend(run_sharp_flags)
    if not abs_turns:
        return 0.0, 0.0
    return float(np.mean(abs_turns)), float(np.mean(sharp_turn_flags))


def _run_turn_features(run_offsets: np.ndarray) -> Tuple[List[float], List[float]]:
    if run_offsets.shape[0] < 2:
        return [], []
    angles = np.arctan2(run_offsets[:, 1], run_offsets[:, 0])
    diffs = np.diff(angles)
    diffs = (diffs + np.pi) % (2 * np.pi) - np.pi
    abs_turns = np.abs(diffs)
    sharp = (abs_turns > (np.pi / 2)).astype(np.float32)
    return abs_turns.tolist(), sharp.tolist()
