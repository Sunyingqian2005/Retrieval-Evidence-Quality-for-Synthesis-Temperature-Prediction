#!/usr/bin/env python3
"""
Controlled diagnostic: original V3 audited pools on the fixed seed-42 test set.

This experiment keeps the exact same unaudited 1000-record held-out set used by
``workflow_core.py``. The original V3 audit-retained records are rebuilt by
joining each old 1000-record split with its corresponding validation-rest split.
The fixed held-out records are then removed occurrence-by-occurrence.  Only the
prediction stage is run with the current DeepSeek endpoint.

Configurations
--------------
1. fixed_random       : reused from the completed fixed-test V4 experiment
2. composition_aware  : reused from the completed fixed-test V4 experiment
3. threshold          : reused from the completed fixed-test V4 experiment
4. physical_audit     : original V3 Physical Audit retained pool
5. type_audit         : original V3 Type Audit retained pool
6. autonomous_audit   : original V3 Autonomous Audit retained pool

The first three predictions are copied only after their metadata and query
records have been checked.  Therefore a normal run makes new API calls only for
the three original-V3 audited pools (3 methods x 2 tasks x 1000 queries).

Examples
--------
python scripts/common_held_out.py --data-only
python scripts/common_held_out.py --prepare-only
python scripts/common_held_out.py --concurrency 30

The script uses the OpenAI-compatible Python client to call DeepSeek.  Set:
    export DEEPSEEK_API_KEY="..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from workflow_core import (
    DEFAULT_OUTPUT_ROOT,
    TASK1_COMPARISONS,
    TASK1_METHODS,
    TASK_SPECS,
    TaskInput,
    build_fixed_inputs,
    build_parser,
    dump_json,
    load_json,
    record_key,
    run_experiment,
)


EXPERIMENT_NAME = "task1_fixed_test_original_v3_pools_v4_results"
BASELINE_EXPERIMENT_NAME = "task1_fixed_test_full_4_2_v4_results"
REUSED_BASELINE_METHODS = [
    "fixed_random",
    "composition_aware",
    "threshold",
]


# Each original script audited the complete dataset, shuffled the retained set
# with seed 42, and then wrote the first 1000 records and the validation rest to
# separate files. Concatenating these two files reconstructs the complete V3
# audit-retained set in its original retrieval order.
OLD_V3_POOL_PARTS: dict[str, dict[str, tuple[str, str]]] = {
    "sintering": {
        "physical_audit": (
            "2019_sintering_operations_filtered_val_2_"
            "train_1000_llm_audited.json",
            "2019_sintering_operations_filtered_val_2_"
            "validation_rest_llm_audited.json",
        ),
        "type_audit": (
            "2019_sintering_operations_filtered_val_2_"
            "train_1000_llm_audited_bond.json",
            "2019_sintering_operations_filtered_val_2_"
            "validation_rest_llm_audited_bond.json",
        ),
        "autonomous_audit": (
            "2019_sintering_operations_filtered_val_2_"
            "train_1000_llm_audited_auto.json",
            "2019_sintering_operations_filtered_val_2_"
            "validation_rest_llm_audited_auto.json",
        ),
    },
    "calcination": {
        "physical_audit": (
            "2019_calcination_operations_filtered_val_"
            "train_1000_llm_audited.json",
            "2019_calcination_operations_filtered_val_"
            "validation_rest_llm_audited.json",
        ),
        "type_audit": (
            "2019_calcination_operations_filtered_val_2_"
            "train_1000_llm_audited_bond.json",
            "2019_calcination_operations_filtered_val_2_"
            "validation_rest_llm_audited_bond.json",
        ),
        "autonomous_audit": (
            "2019_calcination_operations_filtered_val_2_"
            "train_1000_llm_audited_auto.json",
            "2019_calcination_operations_filtered_val_2_"
            "validation_rest_llm_audited_auto.json",
        ),
    },
}


# These checks protect against accidentally selecting one of the many similarly
# named historical files in the working directory.
EXPECTED_OLD_VALIDATION_SIZES: dict[str, dict[str, int]] = {
    "sintering": {
        "physical_audit": 6495,
        "type_audit": 4002,
        "autonomous_audit": 5334,
    },
    "calcination": {
        "physical_audit": 6645,
        "type_audit": 5402,
        "autonomous_audit": 5935,
    },
}


def multiset(records: list[dict[str, Any]]) -> Counter[str]:
    return Counter(record_key(record) for record in records)


def subtract_test_occurrences(
    retained_records: list[dict[str, Any]],
    fixed_test: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """
    Remove exactly one retained occurrence for each matching held-out record.

    The historical JSON files contain no stable record ID.  A multiset
    subtraction is therefore the faithful reconstruction of a record-level
    split: duplicate source records remain candidates only when the source
    contains additional occurrences beyond those assigned to the fixed test.
    """
    removals_needed = multiset(fixed_test)
    remaining: list[dict[str, Any]] = []
    removed = 0

    for record in retained_records:
        signature = record_key(record)
        if removals_needed[signature] > 0:
            removals_needed[signature] -= 1
            removed += 1
        else:
            remaining.append(record)

    return remaining, removed


def count_test_signature_collisions(
    test: list[dict[str, Any]],
    pool: list[dict[str, Any]],
) -> int:
    """
    Count test rows whose content signature still occurs in the candidate pool.

    This can be non-zero when the raw source itself contains duplicate records.
    It is reported rather than silently deleting all duplicates, because Task 1
    uses a record-level split. DOI-level leakage is handled by Task 2.
    """
    pool_signatures = set(multiset(pool))
    return sum(record_key(record) in pool_signatures for record in test)


def build_original_v3_pool_inputs(
    root: Path,
    data_dir: Path,
    task_names: list[str],
    seed: int,
    test_size: int,
) -> list[TaskInput]:
    # This verifies that test/raw/threshold are the exact fixed seed-42 files
    # used in the completed controlled experiment.
    task_inputs = build_fixed_inputs(
        root,
        data_dir,
        task_names,
        seed,
        test_size,
    )

    for task in task_inputs:
        source = load_json(root / TASK_SPECS[task.name]["source"])
        source_counts = multiset(source)
        raw_pool_counts = multiset(task.pools["raw"])
        audit_metadata: dict[str, dict[str, Any]] = {}

        for method, part_names in OLD_V3_POOL_PARTS[task.name].items():
            old_test_part = load_json(root / part_names[0])
            old_validation_part = load_json(root / part_names[1])
            expected_validation_size = EXPECTED_OLD_VALIDATION_SIZES[
                task.name
            ][method]

            if len(old_test_part) != 1000:
                raise ValueError(
                    f"{task.name}/{method}: expected 1000 records in "
                    f"{part_names[0]}, found {len(old_test_part)}."
                )
            if len(old_validation_part) != expected_validation_size:
                raise ValueError(
                    f"{task.name}/{method}: expected "
                    f"{expected_validation_size} records in {part_names[1]}, "
                    f"found {len(old_validation_part)}."
                )

            complete_old_retained = old_test_part + old_validation_part
            complete_counts = multiset(complete_old_retained)
            outside_source = complete_counts - source_counts
            if outside_source:
                raise ValueError(
                    f"{task.name}/{method}: historical pool contains "
                    f"{sum(outside_source.values())} records not present in "
                    f"{TASK_SPECS[task.name]['source']}."
                )

            candidate_pool, removed = subtract_test_occurrences(
                complete_old_retained,
                task.test,
            )
            candidate_counts = multiset(candidate_pool)
            outside_fixed_raw_pool = candidate_counts - raw_pool_counts
            if outside_fixed_raw_pool:
                raise AssertionError(
                    f"{task.name}/{method}: "
                    f"{sum(outside_fixed_raw_pool.values())} retained "
                    "occurrences are outside the fixed candidate universe."
                )

            task.pools[method] = candidate_pool
            collisions = count_test_signature_collisions(
                task.test,
                candidate_pool,
            )
            audit_metadata[method] = {
                "audit_model": "DeepSeek-V3 (historical retained records)",
                "old_test_part": part_names[0],
                "old_validation_part": part_names[1],
                "old_test_part_size": len(old_test_part),
                "old_validation_part_size": len(old_validation_part),
                "complete_old_retained_size": len(complete_old_retained),
                "fixed_test_occurrences_removed": removed,
                "candidate_pool_size": len(candidate_pool),
                "remaining_exact_signature_collisions": collisions,
                "exclusion_rule": "multiset occurrence subtraction",
            }
            task.metadata["pool_sizes"][method] = len(candidate_pool)

            task_dir = data_dir / task.name
            dump_json(task_dir / f"pool_{method}.json", candidate_pool)

        task.metadata["historical_v3_pools"] = audit_metadata
        task.metadata["prediction_model_note"] = (
            "Only downstream prediction uses the current DeepSeek endpoint; "
            "semantic audit decisions are the retained records produced by V3."
        )
        dump_json(data_dir / task.name / "manifest.json", task.metadata)

    return task_inputs


def validate_reusable_prediction_rows(
    rows: list[dict[str, Any]],
    task_name: str,
    method: str,
    test_size: int,
    requested_model: str,
) -> None:
    if len(rows) != test_size:
        raise ValueError(
            f"Reusable {task_name}/{method} predictions contain {len(rows)} "
            f"rows; expected {test_size}."
        )

    expected_ids = {
        f"{task_name}-{index:04d}" for index in range(test_size)
    }
    observed_ids = {str(row.get("query_id")) for row in rows}
    if observed_ids != expected_ids:
        raise ValueError(
            f"Reusable {task_name}/{method} predictions do not contain the "
            "exact fixed-test query IDs."
        )

    if any(row.get("method") != method for row in rows):
        raise ValueError(
            f"Reusable {task_name}/{method} file contains a different method."
        )
    incompatible_models = {
        str(row.get("requested_model"))
        for row in rows
        if row.get("requested_model") not in {None, requested_model}
    }
    if incompatible_models:
        raise ValueError(
            f"Reusable {task_name}/{method} predictions requested "
            f"{sorted(incompatible_models)}, not {requested_model}."
        )


def seed_completed_baseline_predictions(
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    baseline_dir = (
        args.baseline_results_dir.resolve()
        if args.baseline_results_dir
        else (DEFAULT_OUTPUT_ROOT / BASELINE_EXPERIMENT_NAME)
    )
    baseline_manifest_path = baseline_dir / "run_manifest.json"
    if not baseline_manifest_path.exists():
        raise FileNotFoundError(
            "Cannot reuse the completed baseline because this manifest is "
            f"missing: {baseline_manifest_path}"
        )

    with baseline_manifest_path.open("r", encoding="utf-8") as handle:
        baseline_manifest = json.load(handle)
    checks = {
        "seed": args.seed,
        "test_size": args.test_size,
        "k": args.k,
        "requested_model": args.model,
        "temperature": args.temperature,
    }
    for key, expected in checks.items():
        observed = baseline_manifest.get(key)
        if observed != expected:
            raise ValueError(
                f"Baseline manifest mismatch for {key}: expected "
                f"{expected!r}, found {observed!r}."
            )

    destination_dir = output_dir / "predictions"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for task_name in args.tasks:
        for method in REUSED_BASELINE_METHODS:
            source = baseline_dir / "predictions" / f"{task_name}_{method}.json"
            destination = destination_dir / source.name
            rows = load_json(source)
            validate_reusable_prediction_rows(
                rows,
                task_name,
                method,
                args.test_size,
                args.model,
            )
            if destination.exists():
                existing = load_json(destination)
                validate_reusable_prediction_rows(
                    existing,
                    task_name,
                    method,
                    args.test_size,
                    args.model,
                )
                print(f"Baseline checkpoint already present: {destination}")
                continue
            shutil.copy2(source, destination)
            print(f"Reused completed baseline: {source.name}")


def main() -> None:
    parser = build_parser(
        (
            "Original V3 audited candidate pools evaluated on the fixed "
            "1000-record held-out set with current DeepSeek prediction"
        ),
        EXPERIMENT_NAME,
    )
    parser.add_argument(
        "--baseline-results-dir",
        type=Path,
        default=None,
        help=(
            "Completed fixed-test V4 result directory used to reuse "
            "fixed-random, composition-aware, and threshold predictions. "
            f"Default: <repository>/outputs/{BASELINE_EXPERIMENT_NAME}"
        ),
    )
    parser.add_argument(
        "--no-reuse-baselines",
        action="store_true",
        help=(
            "Do not copy the three completed baseline prediction files; "
            "re-run all six configurations instead."
        ),
    )
    args = parser.parse_args()

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / EXPERIMENT_NAME)
    )
    if (
        not args.no_reuse_baselines
        and not args.data_only
        and not args.prepare_only
    ):
        seed_completed_baseline_predictions(args, output_dir)

    asyncio.run(
        run_experiment(
            args,
            build_original_v3_pool_inputs,
            TASK1_METHODS,
            TASK1_COMPARISONS,
            [],
            EXPERIMENT_NAME,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; completed predictions remain checkpointed.")
        sys.exit(130)
