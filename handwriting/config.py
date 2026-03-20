from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

from .generation import DecodingMode
from .tokenizers import BUILTIN_TOKENIZER_SPECS
from .utils import load_json_dict, resolve_bundle_path


def load_word_panel(path: Path) -> List[str]:
    payload = load_json_dict(path) if path.suffix == ".json" else None
    if payload is None:
        raise ValueError(f"Unsupported word panel format: {path}")
    words = payload.get("words", [])
    if not isinstance(words, list) or not words:
        raise ValueError(f"Expected a non-empty word list in {path}")
    result = [str(word).strip() for word in words if str(word).strip()]
    if not result:
        raise ValueError(f"No valid evaluation words found in {path}")
    return result


def load_decoding_modes(path: Path) -> List[DecodingMode]:
    payload = load_json_dict(path) if path.suffix == ".json" else None
    if payload is None:
        raise ValueError(f"Unsupported decoding config format: {path}")
    modes = payload.get("modes", [])
    if not isinstance(modes, list) or not modes:
        raise ValueError(f"Expected a non-empty decoding mode list in {path}")

    result: List[DecodingMode] = []
    for item in modes:
        if not isinstance(item, dict):
            raise ValueError(f"Decoding mode entries must be objects in {path}")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError(f"Decoding mode entry is missing a name in {path}")
        result.append(
            DecodingMode(
                name=name,
                temperature=float(item.get("temperature", 1.0)),
                top_k=int(item.get("top_k", 0)),
                greedy=bool(item.get("greedy", False)),
            )
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Version1A text-to-trajectory model from processed word-level NPZ splits.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--train-npz", type=str, required=True)
    parser.add_argument("--val-npz", type=str, required=True)
    parser.add_argument("--test-npz", type=str, default=None)
    parser.add_argument(
        "--tokenizer-variant",
        type=str,
        default="finer_2token_128x64_split_radius_median_decode",
        choices=sorted(BUILTIN_TOKENIZER_SPECS),
    )
    parser.add_argument("--tokenizer-config", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    parser.add_argument("--sample-dir", type=str, default="samples/local")
    parser.add_argument("--experiment-name", type=str, default="deepwriting_only_experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--resume-reset-optimizer", action="store_true")
    parser.add_argument("--resume-reset-scheduler", action="store_true")
    parser.add_argument("--resume-reset-scaler", action="store_true")
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-stroke-tokens", type=int, default=1024)
    parser.add_argument("--max-points-for-bins", type=int, default=500_000)
    parser.add_argument("--writer-id-map", type=str, default="configs/writers/writer_id_map.json")
    parser.add_argument("--panel-writers-config", type=str, default="configs/writers/default_panel_writers.json")
    parser.add_argument("--use-writer-conditioning", action="store_true")
    parser.add_argument("--style-cluster-map", type=str, default="configs/style_clusters/writer_to_cluster_map.json")
    parser.add_argument("--panel-clusters-config", type=str, default="configs/style_clusters/default_panel_clusters.json")
    parser.add_argument("--use-style-cluster-conditioning", action="store_true")
    parser.add_argument(
        "--multi-gpu-mode",
        type=str,
        choices=["single", "dataparallel", "ddp"],
        default="single",
    )
    parser.add_argument("--writer-embed-dim", type=int, default=64)
    parser.add_argument("--style-cluster-embed-dim", type=int, default=48)
    parser.add_argument(
        "--writer-conditioning-mode",
        type=str,
        choices=["add_to_text", "add_to_stroke", "add_to_both"],
        default="add_to_both",
    )
    parser.add_argument(
        "--cluster-conditioning-mode",
        type=str,
        choices=["add_to_text", "add_to_stroke", "add_to_both"],
        default="add_to_both",
    )
    parser.add_argument(
        "--writer-unseen-policy",
        type=str,
        choices=["error", "map_to_unk"],
        default="map_to_unk",
    )
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--scheduler-type", type=str, choices=["cosine", "step"], default="cosine")
    parser.add_argument("--step-lr-every", type=int, default=0)
    parser.add_argument("--lr-decay", type=float, default=0.5)
    parser.add_argument("--train-downsample-keep-min", type=float, default=1.0)
    parser.add_argument("--train-downsample-keep-max", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-every-epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--panel-every-epochs", "--sample-every-epochs", dest="panel_every_epochs", type=int, default=5)
    parser.add_argument("--sample-max-gen-tokens", type=int, default=256)
    parser.add_argument("--eval-word-panel-config", type=str, default="configs/eval/default_word_panel.json")
    parser.add_argument("--decoding-modes-config", type=str, default="configs/eval/default_decoding_modes.json")
    parser.add_argument("--tokenizer-diagnostics-per-source-split", type=int, default=1)
    parser.add_argument("--skip-tokenizer-diagnostics", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    bundle_root = Path(__file__).resolve().parent.parent
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    bootstrap_args, _ = bootstrap.parse_known_args()

    config_path = resolve_bundle_path(bootstrap_args.config, bundle_root)
    config_defaults: Dict[str, Any] = {}
    if config_path is not None:
        config_defaults = load_json_dict(config_path)

    parser = build_arg_parser()
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    args.bundle_root = str(bundle_root)
    args.config = str(config_path) if config_path is not None else None
    if sys.version_info >= (3, 14) and args.num_workers > 0:
        print("Python 3.14+ detected; forcing --num-workers 0 to avoid DataLoader pickling issues.")
        args.num_workers = 0
    return args
