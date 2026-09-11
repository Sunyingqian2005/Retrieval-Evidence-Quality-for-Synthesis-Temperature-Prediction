#!/usr/bin/env python3
"""
Task 2: DOI-grouped leakage-control experiment.

DOI groups, rather than individual records, are assigned to the test set.
Consequently, no DOI can occur in both the 1000-record held-out set and the
candidate retrieval pool.

Configurations
--------------
1. fixed_random       : fixed 40-shot examples from the DOI-blocked raw pool
2. composition_aware  : element-overlap retrieval from the same raw pool
3. type_audit         : element-overlap retrieval from the V4 Threshold + Type
                        Audit retained subset of the DOI-blocked pool

The DOI-grouped test set is created first. Threshold filtering and Type Audit
are then applied only to the remaining DOI-blocked candidate pool.

Examples
--------
python task2_doi_grouped_v4.py --data-only
python task2_doi_grouped_v4.py --prepare-only
python task2_doi_grouped_v4.py
"""

from __future__ import annotations

import asyncio
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from workflow_core import (
    METHOD_CONFIG,
    TASK_SPECS,
    TaskInput,
    build_parser,
    doi_overlap_summary,
    dump_json,
    finite_float,
    load_json,
    normalize_doi,
    record_key,
    run_experiment,
)


TASK2_METHODS = [
    "fixed_random",
    "composition_aware",
    "type_audit",
]

TASK2_COMPARISONS = [
    ("composition_aware", "fixed_random"),
    ("type_audit", "fixed_random"),
    ("type_audit", "composition_aware"),
]

DOI_PATTERN = re.compile(r"^10\.[0-9]{4,9}/\S+$")


def build_doi_grouped_inputs(
    root: Path,
    data_dir: Path,
    task_names: list[str],
    seed: int,
    test_size: int,
) -> list[TaskInput]:
    built: list[TaskInput] = []

    for task_name in task_names:
        spec = TASK_SPECS[task_name]
        source = load_json(root / spec["source"])

        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        invalid_indices: list[int] = []
        for index, record in enumerate(source):
            doi = normalize_doi(record.get("doi"))
            if not doi or not DOI_PATTERN.match(doi):
                invalid_indices.append(index)
            else:
                groups[doi].append(record)
        if invalid_indices:
            raise ValueError(
                f"{task_name}: {len(invalid_indices)} records have missing or "
                "malformed DOI values; DOI grouping cannot proceed safely."
            )

        shuffled_dois = sorted(groups)
        random.Random(seed).shuffle(shuffled_dois)

        test_dois: list[str] = []
        running_size = 0
        for doi in shuffled_dois:
            if running_size >= test_size:
                break
            test_dois.append(doi)
            running_size += len(groups[doi])

        if running_size != test_size:
            raise ValueError(
                f"{task_name}: whole-DOI grouping produced {running_size} test "
                f"records instead of exactly {test_size}. Do not truncate a DOI "
                "group; revise the group-selection rule explicitly."
            )

        test_doi_set = set(test_dois)
        test = [
            record
            for doi in test_dois
            for record in groups[doi]
        ]
        raw_pool = [
            record
            for record in source
            if normalize_doi(record.get("doi")) not in test_doi_set
        ]

        test_doi_values = {
            normalize_doi(record.get("doi")) for record in test
        }
        pool_doi_values = {
            normalize_doi(record.get("doi")) for record in raw_pool
        }
        overlap = test_doi_values.intersection(pool_doi_values)
        if overlap:
            raise AssertionError(
                f"{task_name}: DOI leakage detected after grouped split."
            )

        threshold_pool = [
            record
            for record in raw_pool
            if (
                (temperature := finite_float(record.get("target_temperature")))
                is not None
                and 300.0 <= temperature <= 2200.0
            )
        ]

        pools = {
            "raw": raw_pool,
            "threshold": threshold_pool,
        }
        overlap_summary = doi_overlap_summary(test, raw_pool)
        metadata = {
            "split": "DOI-grouped split",
            "seed": seed,
            "source_file": spec["source"],
            "test_size": len(test),
            "source_size": len(source),
            "test_doi_groups": len(test_doi_set),
            "pool_doi_groups": len(pool_doi_values),
            "doi_overlap_count": len(overlap),
            "pool_sizes": {name: len(pool) for name, pool in pools.items()},
            **overlap_summary,
        }

        task_dir = data_dir / task_name
        dump_json(task_dir / "test.json", test)
        for pool_name, pool in pools.items():
            dump_json(task_dir / f"pool_{pool_name}.json", pool)
        dump_json(task_dir / "test_dois.json", sorted(test_doi_set))
        dump_json(task_dir / "manifest.json", metadata)

        built.append(
            TaskInput(
                name=task_name,
                temp_key=spec["temp_key"],
                test=test,
                pools=pools,
                metadata=metadata,
            )
        )

    return built


def main() -> None:
    # `threshold` is not run as a separate Task-2 condition, but retaining this
    # mapping makes the shared runner's configuration explicit and auditable.
    if METHOD_CONFIG["type_audit"] != ("type_audit", "composition"):
        raise RuntimeError("Unexpected shared method configuration.")

    parser = build_parser(
        "Task 2: DOI-grouped leakage-control comparison",
        "task2_doi_grouped_strict_v4_results",
    )
    args = parser.parse_args()
    asyncio.run(
        run_experiment(
            args,
            build_doi_grouped_inputs,
            TASK2_METHODS,
            TASK2_COMPARISONS,
            ["type_audit"],
            "task2_doi_grouped_strict_v4_results",
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; completed predictions remain checkpointed.")
        sys.exit(130)
