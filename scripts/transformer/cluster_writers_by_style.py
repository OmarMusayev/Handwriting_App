#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from handwriting.data import load_processed_npz
from handwriting.style_clusters import build_style_cluster_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster filtered DeepWriting writers by aggregated style features.")
    parser.add_argument(
        "--data-root",
        type=str,
        default="Data/processed_deepwriting_local_only",
        help="Bundle-relative directory containing train/val/test NPZ files.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="configs/style_clusters",
        help="Bundle-relative output directory for style-cluster artifacts.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--k-min", type=int, default=4)
    parser.add_argument("--k-max", type=int, default=12)
    parser.add_argument("--panel-cluster-count", type=int, default=3)
    parser.add_argument("--examples-per-cluster", type=int, default=9)
    parser.add_argument("--pca-max-components", type=int, default=8)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    bundle_root = Path(__file__).resolve().parent
    data_root = (bundle_root / args.data_root).resolve()
    output_root = (bundle_root / args.output_root).resolve()

    split_map = {
        split_name: load_processed_npz(data_root / f"{split_name}.npz")
        for split_name in ["train", "val", "test"]
    }
    summary = build_style_cluster_artifacts(
        split_map=split_map,
        output_root=output_root,
        k_values=list(range(int(args.k_min), int(args.k_max) + 1)),
        seed=int(args.seed),
        representative_cluster_count=int(args.panel_cluster_count),
        examples_per_cluster=int(args.examples_per_cluster),
        pca_max_components=int(args.pca_max_components),
    )
    print(f"Saved style-cluster artifacts to {output_root}")
    print(f"Chosen k: {summary['chosen_k']}")
    print(f"Writer-to-cluster map: {summary['writer_to_cluster_map_path']}")
    print(f"Default panel clusters: {summary['default_panel_clusters_path']}")


if __name__ == "__main__":
    main()
