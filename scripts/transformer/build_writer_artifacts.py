#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from handwriting.data import load_processed_npz
from handwriting.writers import build_writer_artifact_payloads


def main() -> None:
    bundle_root = Path(__file__).resolve().parent
    data_root = bundle_root / "Data" / "processed_deepwriting_local_only"
    output_root = bundle_root / "configs" / "writers"
    output_root.mkdir(parents=True, exist_ok=True)

    split_to_writer_ids = {}
    for split_name in ["train", "val", "test"]:
        split = load_processed_npz(data_root / f"{split_name}.npz")
        split_to_writer_ids[split_name] = split.writer_ids

    writer_map_payload, panel_payload = build_writer_artifact_payloads(
        split_to_writer_ids=split_to_writer_ids,
        representative_count=3,
    )

    writer_map_path = output_root / "writer_id_map.json"
    panel_writers_path = output_root / "default_panel_writers.json"
    writer_map_path.write_text(json.dumps(writer_map_payload, indent=2))
    panel_writers_path.write_text(json.dumps(panel_payload, indent=2))

    print(f"Saved writer map to {writer_map_path}")
    print(f"Saved default panel writers to {panel_writers_path}")


if __name__ == "__main__":
    main()
