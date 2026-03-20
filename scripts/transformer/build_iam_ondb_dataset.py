#!/usr/bin/env python3

"""Build the IAM-OnDB all-writers local-only dataset bundle.

This script inspects the root DATA XMLs, records why they are not sufficient as
online-trajectory training input on their own, then builds the actual canonical
and word-level dataset from the available local IAM trajectory JSON export,
runs the dot/cross audit, applies the operational local-only filter, and writes
all bundle-local outputs.

Typical local run:
  python build_iam_ondb_dataset.py
"""

from handwriting.iam_ondb_build import main


if __name__ == "__main__":
    raise SystemExit(main())
