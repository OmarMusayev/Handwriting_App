from __future__ import annotations

"""Operational interpretation helpers for the DeepWriting dot/cross timing audit."""

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence


RAW_TO_OPERATIONAL_LABEL = {
    "immediate": "local",
    "nearby": "delayed",
    "delayed": "delayed",
    "ambiguous": "ambiguous",
}
OPERATIONAL_LABELS = ("local", "delayed", "ambiguous")
COMMITTED_OPERATIONAL_LABELS = ("local", "delayed")
OPERATIONAL_RULE_NAME = "immediate_local__nearby_delayed__ambiguous_manual_review"


def operational_label_from_audit_label(audit_label: str) -> str:
    label = RAW_TO_OPERATIONAL_LABEL.get(str(audit_label))
    if label is None:
        raise KeyError(f"Unknown audit label '{audit_label}'")
    return label


def is_committed_operational_label(label: str) -> bool:
    return str(label) in COMMITTED_OPERATIONAL_LABELS


def build_operational_row(row: Mapping[str, object]) -> dict:
    derived = dict(row)
    audit_label = str(derived.get("category") or "")
    operational_label = operational_label_from_audit_label(audit_label)
    derived["operational_label"] = operational_label
    derived["operational_status"] = "committed" if is_committed_operational_label(operational_label) else "unresolved"
    derived["is_committed"] = bool(is_committed_operational_label(operational_label))
    derived["operational_rule"] = OPERATIONAL_RULE_NAME
    return derived


def build_operational_rows(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    return [build_operational_row(row) for row in rows]


def load_annotation_rows(path: Optional[Path]) -> Dict[str, dict]:
    if path is None or not path.exists():
        return {}
    rows: Dict[str, dict] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            occurrence_id = str(row.get("occurrence_id") or "").strip()
            if not occurrence_id:
                continue
            rows[occurrence_id] = dict(row)
    return rows


def manual_operational_label(annotation: Optional[Mapping[str, object]]) -> str:
    if not annotation:
        return "unreviewed"
    likely = str(annotation.get("likely_true_delayed") or "").strip().lower()
    if likely == "yes":
        return "delayed"
    if likely == "no":
        return "local"
    human_label = str(annotation.get("human_label") or "").strip().lower()
    if human_label == "delayed":
        return "delayed"
    if human_label == "local":
        return "local"
    if human_label == "bad_body_mark_pair":
        return "bad_pair"
    if human_label == "unclear":
        return "unclear"
    return "unreviewed"


def summarize_operational_rows(rows: Sequence[Mapping[str, object]]) -> dict:
    total = len(rows)
    operational_counts = Counter(str(row["operational_label"]) for row in rows)
    raw_counts = Counter(str(row.get("category") or "") for row in rows)
    committed_total = int(operational_counts.get("local", 0) + operational_counts.get("delayed", 0))
    unresolved_total = int(operational_counts.get("ambiguous", 0))
    by_char: Dict[str, dict] = {}
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("char") or "")].append(row)
    for char, subset in sorted(grouped.items()):
        subset_counts = Counter(str(row["operational_label"]) for row in subset)
        subset_raw_counts = Counter(str(row.get("category") or "") for row in subset)
        subset_committed = int(subset_counts.get("local", 0) + subset_counts.get("delayed", 0))
        by_char[char] = {
            "total_occurrences": int(len(subset)),
            "operational_counts": {label: int(subset_counts.get(label, 0)) for label in OPERATIONAL_LABELS},
            "raw_audit_counts": {
                label: int(subset_raw_counts.get(label, 0))
                for label in ("immediate", "nearby", "delayed", "ambiguous")
            },
            "committed_occurrences": subset_committed,
            "unresolved_occurrences": int(subset_counts.get("ambiguous", 0)),
            "committed_fraction": float(subset_committed / len(subset)) if subset else None,
            "committed_local_fraction": (
                float(subset_counts.get("local", 0) / subset_committed) if subset_committed else None
            ),
            "committed_delayed_fraction": (
                float(subset_counts.get("delayed", 0) / subset_committed) if subset_committed else None
            ),
        }
    return {
        "rule": OPERATIONAL_RULE_NAME,
        "total_occurrences": int(total),
        "operational_counts": {label: int(operational_counts.get(label, 0)) for label in OPERATIONAL_LABELS},
        "raw_audit_counts": {
            label: int(raw_counts.get(label, 0))
            for label in ("immediate", "nearby", "delayed", "ambiguous")
        },
        "committed_occurrences": committed_total,
        "unresolved_occurrences": unresolved_total,
        "committed_fraction": float(committed_total / total) if total else None,
        "unresolved_fraction": float(unresolved_total / total) if total else None,
        "committed_local_fraction": (
            float(operational_counts.get("local", 0) / committed_total) if committed_total else None
        ),
        "committed_delayed_fraction": (
            float(operational_counts.get("delayed", 0) / committed_total) if committed_total else None
        ),
        "by_char": by_char,
    }


def compute_manual_agreement(
    operational_rows: Sequence[Mapping[str, object]],
    annotations_by_occurrence_id: Mapping[str, Mapping[str, object]],
) -> dict:
    reviewed_rows = [
        row for row in operational_rows if str(row.get("occurrence_id") or "") in annotations_by_occurrence_id
    ]
    confusion = Counter()
    audit_counts = Counter()
    human_counts = Counter()
    committed_total = 0
    committed_correct = 0
    by_char: Dict[str, Counter] = defaultdict(Counter)

    for row in reviewed_rows:
        occurrence_id = str(row["occurrence_id"])
        audit_label = str(row["operational_label"])
        human_label = manual_operational_label(annotations_by_occurrence_id.get(occurrence_id))
        confusion[(audit_label, human_label)] += 1
        audit_counts[audit_label] += 1
        human_counts[human_label] += 1
        by_char[str(row.get("char") or "")][(audit_label, human_label)] += 1
        if audit_label in COMMITTED_OPERATIONAL_LABELS and human_label in COMMITTED_OPERATIONAL_LABELS:
            committed_total += 1
            if audit_label == human_label:
                committed_correct += 1

    by_char_summary = {}
    for char, counter in sorted(by_char.items()):
        char_committed_total = 0
        char_committed_correct = 0
        for (audit_label, human_label), count in counter.items():
            if audit_label in COMMITTED_OPERATIONAL_LABELS and human_label in COMMITTED_OPERATIONAL_LABELS:
                char_committed_total += int(count)
                if audit_label == human_label:
                    char_committed_correct += int(count)
        by_char_summary[char] = {
            "reviewed_occurrences": int(sum(counter.values())),
            "confusion": {f"{audit} -> {human}": int(count) for (audit, human), count in sorted(counter.items())},
            "committed_accuracy": (
                float(char_committed_correct / char_committed_total) if char_committed_total else None
            ),
            "committed_total": int(char_committed_total),
            "committed_correct": int(char_committed_correct),
        }

    return {
        "reviewed_occurrences": int(len(reviewed_rows)),
        "audit_operational_counts": {label: int(audit_counts.get(label, 0)) for label in OPERATIONAL_LABELS},
        "human_operational_counts": {
            label: int(human_counts.get(label, 0))
            for label in ("local", "delayed", "bad_pair", "unclear", "unreviewed")
        },
        "confusion": {f"{audit} -> {human}": int(count) for (audit, human), count in sorted(confusion.items())},
        "committed_total": int(committed_total),
        "committed_correct": int(committed_correct),
        "committed_accuracy": float(committed_correct / committed_total) if committed_total else None,
        "by_char": by_char_summary,
    }


def render_operational_summary_markdown(summary: Mapping[str, object], manual_agreement: Optional[Mapping[str, object]]) -> str:
    lines = [
        "# Dot/Cross Operational Interpretation",
        "",
        "Operational mapping used:",
        "",
        "- `immediate => local`",
        "- `nearby => delayed`",
        "- `delayed => delayed`",
        "- `ambiguous => unresolved / manual review only`",
        "",
        "## Overall",
        "",
        f"- Total occurrences: `{summary['total_occurrences']}`",
        f"- Committed local: `{summary['operational_counts']['local']}`",
        f"- Committed delayed: `{summary['operational_counts']['delayed']}`",
        f"- Ambiguous / unresolved: `{summary['operational_counts']['ambiguous']}`",
        f"- Committed total: `{summary['committed_occurrences']}`",
        f"- Committed fraction: `{summary['committed_fraction']}`",
        f"- Committed local fraction: `{summary['committed_local_fraction']}`",
        f"- Committed delayed fraction: `{summary['committed_delayed_fraction']}`",
        "",
        "## Per Character",
        "",
    ]
    for char in ("i", "j", "t"):
        char_summary = summary["by_char"].get(char)
        if not char_summary:
            continue
        lines.extend(
            [
                f"### `{char}`",
                "",
                f"- Total occurrences: `{char_summary['total_occurrences']}`",
                f"- Committed local: `{char_summary['operational_counts']['local']}`",
                f"- Committed delayed: `{char_summary['operational_counts']['delayed']}`",
                f"- Ambiguous / unresolved: `{char_summary['operational_counts']['ambiguous']}`",
                f"- Committed fraction: `{char_summary['committed_fraction']}`",
                f"- Committed local fraction: `{char_summary['committed_local_fraction']}`",
                f"- Committed delayed fraction: `{char_summary['committed_delayed_fraction']}`",
                "",
            ]
        )
    if manual_agreement is not None:
        lines.extend(
            [
                "## Manual-Review Support",
                "",
                f"- Reviewed sample size: `{manual_agreement['reviewed_occurrences']}`",
                f"- Committed reviewed cases: `{manual_agreement['committed_total']}`",
                f"- Committed agreement under the operational mapping: `{manual_agreement['committed_accuracy']}`",
                "- Operational conclusion: `supported for workflow use on the current manual-review sample`",
                "- Caution: `this is operational validation, not a formal prevalence guarantee for every unresolved case`",
                "",
            ]
        )
    return "\n".join(lines)


def render_manual_agreement_markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Manual Review Agreement",
        "",
        f"- Reviewed occurrences: `{report['reviewed_occurrences']}`",
        f"- Committed reviewed cases: `{report['committed_total']}`",
        f"- Committed correct: `{report['committed_correct']}`",
        f"- Committed accuracy: `{report['committed_accuracy']}`",
        "",
        "## Confusion",
        "",
    ]
    for key, value in sorted(report["confusion"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## By Character", ""])
    for char, char_report in sorted(report["by_char"].items()):
        lines.extend(
            [
                f"### `{char}`",
                "",
                f"- Reviewed occurrences: `{char_report['reviewed_occurrences']}`",
                f"- Committed total: `{char_report['committed_total']}`",
                f"- Committed correct: `{char_report['committed_correct']}`",
                f"- Committed accuracy: `{char_report['committed_accuracy']}`",
                "",
            ]
        )
    return "\n".join(lines)
