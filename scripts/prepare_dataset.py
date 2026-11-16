#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unpack the gzipped training dataset.

The landmark dataset is stored compressed (keypoint.csv.gz, ~14 MB) so the
repository stays small; the training notebook and the logging mode in app.py both
expect the plain keypoint.csv (~57 MB). Run this once before training.

    python scripts/prepare_dataset.py
"""

import gzip
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "model" / "keypoint_classifier" / "keypoint.csv.gz"
TARGET = ROOT / "model" / "keypoint_classifier" / "keypoint.csv"


def main() -> int:
    if not ARCHIVE.exists():
        print(f"Archive not found: {ARCHIVE}")
        return 1

    if TARGET.exists():
        print(f"{TARGET.name} already exists ({TARGET.stat().st_size:,} bytes).")
        print("Delete it first if you want to restore the original dataset.")
        return 0

    print(f"Unpacking {ARCHIVE.name} -> {TARGET.name} ...")
    with gzip.open(ARCHIVE, "rb") as src, open(TARGET, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)

    rows = sum(1 for _ in open(TARGET, encoding="utf-8"))
    print(f"Done: {TARGET.stat().st_size:,} bytes, {rows:,} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
