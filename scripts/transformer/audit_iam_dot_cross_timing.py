#!/usr/bin/env python3

"""Run the IAM-OnDB dot/cross timing audit on bundle-local canonical outputs."""

from pathlib import Path

from handwriting.dot_cross_audit import run_dot_cross_timing_audit


def main() -> int:
    bundle_root = Path(__file__).resolve().parent
    canonical_dir = bundle_root / "Data" / "canonical_iam_raw"
    out_dir = bundle_root / "audits" / "iam_dot_cross_timing"
    summary = run_dot_cross_timing_audit(
        canonical_jsonl=canonical_dir / "canonical_samples.jsonl",
        canonical_npz=canonical_dir / "canonical_samples.npz",
        out_dir=out_dir,
        source_filter="iamondb",
        source_label="IAM-OnDB",
        max_examples_per_category=0,
    )
    print(f"[audit] wrote {out_dir.as_posix()}")
    print(f"[audit] eligible_occurrences={summary['eligible_occurrences']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
