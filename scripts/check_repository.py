#!/usr/bin/env python3
"""Run lightweight checks before publishing the repository."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "configs/main_experiments.json",
    "data/README.md",
    "data/schema.json",
    "data/example_records.json",
    "scripts/workflow_core.py",
    "scripts/common_held_out.py",
    "scripts/jaccard_comparison.py",
    "scripts/doi_grouped.py",
    "scripts/endpoint_transfer.py",
    "scripts/verify_data.py",
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".cff", ".example"}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(api[_-]?key)\s*=\s*['\"][^'\"\s]{12,}['\"]"),
)
PERSONAL_PATH = re.compile(r"/Users/[^/]+/")


def text_files() -> list[Path]:
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and "outputs" not in path.parts
        and not ("data" in path.parts and "full" in path.parts)
        and path.suffix.lower() in TEXT_SUFFIXES
    ]


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).exists():
            errors.append(f"Missing required file: {relative}")

    for path in text_files():
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(ROOT)
        if relative.as_posix() != "scripts/check_repository.py":
            for pattern in SECRET_PATTERNS:
                if pattern.search(content):
                    errors.append(f"Possible API secret in {relative}")
        if (
            PERSONAL_PATH.search(content)
            and relative.as_posix() != "scripts/check_repository.py"
        ):
            errors.append(f"Personal absolute path in {relative}")
        if "YOUR-USERNAME" in content and relative.as_posix() != "scripts/check_repository.py":
            warnings.append(f"Replace GitHub placeholder in {relative}")

    for relative in (
        "configs/main_experiments.json",
        "data/schema.json",
        "data/example_records.json",
    ):
        path = ROOT / relative
        if path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"Invalid JSON in {relative}: {exc}")

    if (ROOT / ".env").exists():
        warnings.append("Local .env exists; confirm it remains ignored by Git")
    if (ROOT / "data/full").exists():
        warnings.append("Local data/full exists; confirm it remains ignored by Git")

    for warning in sorted(set(warnings)):
        print(f"WARNING: {warning}")
    for error in sorted(set(errors)):
        print(f"ERROR: {error}")

    if errors:
        print(f"Repository check failed with {len(set(errors))} error(s).")
        return 1
    print("Repository check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
