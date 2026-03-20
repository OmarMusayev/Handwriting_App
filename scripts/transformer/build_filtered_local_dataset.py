"""
Build the paper-leaning local-only DeepWriting training splits.

Filtering rule:
  - operational local => keepable
  - operational delayed => drop word
  - operational ambiguous => drop word

The filter operates at the processed word-sample level. A word is dropped if any audited
`i/j/t` occurrence inside that word is delayed or ambiguous under the operational audit.

Typical local run on the source workspace:
  python build_filtered_local_dataset.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np


def build_parser(bundle_root: Path) -> argparse.ArgumentParser:
    sibling_bundle = bundle_root.parent / "Version1A_DeepWritingOnly"
    parser = argparse.ArgumentParser(description="Filter DeepWriting word-level splits to safer local-only training cases.")
    parser.add_argument(
        "--source-train",
        type=Path,
        default=sibling_bundle / "Data" / "processed_deepwriting_only" / "train.npz",
    )
    parser.add_argument(
        "--source-val",
        type=Path,
        default=sibling_bundle / "Data" / "processed_deepwriting_only" / "val.npz",
    )
    parser.add_argument(
        "--source-test",
        type=Path,
        default=sibling_bundle / "Data" / "processed_deepwriting_only" / "test.npz",
    )
    parser.add_argument(
        "--operational-occurrences",
        type=Path,
        default=bundle_root / "Data" / "filter_inputs" / "occurrences_with_operational_label.jsonl",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=bundle_root / "Data" / "processed_deepwriting_local_only",
    )
    return parser


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
    for line in path.open():
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
    kept = 0
    dropped = 0

    for index in range(sample_ids.shape[0]):
        key = (str(canonical_ids[index]), int(word_indices[index]))
        status = status_map.get(key)
        keep, reason = decide_word_reason(status)
        keep_mask[index] = keep
        reason_counts[reason] += 1
        kept += int(keep)
        dropped += int(not keep)
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
    filtered_arrays = {}
    for key in npz.files:
        filtered_arrays[key] = npz[key][keep_mask]
    np.savez_compressed(out_path, **filtered_arrays)
    npz.close()

    return {
        "split": str(split_names[0]) if split_names.size else out_path.stem,
        "source_path": npz_path.as_posix(),
        "out_path": out_path.as_posix(),
        "original_count": int(sample_ids.shape[0]),
        "kept_count": int(kept),
        "dropped_count": int(dropped),
        "kept_fraction": float(kept / sample_ids.shape[0]) if sample_ids.size else None,
        "reason_counts": {str(key): int(value) for key, value in sorted(reason_counts.items())},
        "manifest_rows": manifest_rows,
    }


def write_manifest_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def write_summary_md(path: Path, summary: dict) -> None:
    lines = [
        "# Local-Only Filter Report",
        "",
        "Filtering rule:",
        "",
        "- drop any word with an operational `delayed` i/j/t occurrence",
        "- drop any word with an operational `ambiguous` i/j/t occurrence",
        "- keep words whose audited i/j/t occurrences are all operational `local`",
        "- keep words with no audited i/j/t occurrence",
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
    path.write_text("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    bundle_root = Path(__file__).resolve().parent
    parser = build_parser(bundle_root)
    args = parser.parse_args(argv)

    status_map = load_operational_status_map(args.operational_occurrences.expanduser().resolve())
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    split_reports = {}
    all_manifest_rows = []
    for split_name, source_path in (
        ("train", args.source_train),
        ("val", args.source_val),
        ("test", args.source_test),
    ):
        report = filter_split(
            source_path.expanduser().resolve(),
            status_map,
            out_dir / f"{split_name}.npz",
        )
        split_reports[split_name] = report
        all_manifest_rows.extend(report.pop("manifest_rows"))

    overall_reason_counts = Counter()
    original_count = kept_count = dropped_count = 0
    for report in split_reports.values():
        original_count += int(report["original_count"])
        kept_count += int(report["kept_count"])
        dropped_count += int(report["dropped_count"])
        overall_reason_counts.update(report["reason_counts"])

    summary = {
        "operational_occurrences_path": args.operational_occurrences.expanduser().resolve().as_posix(),
        "out_dir": out_dir.as_posix(),
        "overall": {
            "original_count": int(original_count),
            "kept_count": int(kept_count),
            "dropped_count": int(dropped_count),
            "kept_fraction": float(kept_count / original_count) if original_count else None,
            "reason_counts": {str(key): int(value) for key, value in sorted(overall_reason_counts.items())},
        },
        "by_split": split_reports,
    }

    write_manifest_jsonl(out_dir / "filter_manifest.jsonl", all_manifest_rows)
    (out_dir / "filtering_report.json").write_text(json.dumps(summary, indent=2))
    write_summary_md(out_dir / "filtering_report.md", summary)
    print(f"[filter] wrote {out_dir.as_posix()}")
    print(
        f"[filter] kept {summary['overall']['kept_count']}/{summary['overall']['original_count']} "
        f"({summary['overall']['kept_fraction']:.4f})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
