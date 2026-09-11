#!/usr/bin/env python3
"""Verify local research-data files against the committed SHA-256 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPOSITORY_ROOT / "data" / "full"
MANIFEST = REPOSITORY_ROOT / "data" / "file_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    failures = 0

    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        path = root / row["filename"]
        if not path.exists():
            print(f"MISSING {row['filename']}")
            failures += 1
            continue
        try:
            records = len(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, TypeError) as exc:
            print(f"INVALID {row['filename']}: {exc}")
            failures += 1
            continue
        observed = (str(records), str(path.stat().st_size), sha256(path))
        expected = (row["records"], row["bytes"], row["sha256"])
        if observed != expected:
            print(f"MISMATCH {row['filename']}")
            failures += 1
        else:
            print(f"OK {row['filename']}")

    if failures:
        print(f"Data verification failed for {failures} file(s).")
        return 1
    print(f"Verified {len(rows)} files in {root}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
