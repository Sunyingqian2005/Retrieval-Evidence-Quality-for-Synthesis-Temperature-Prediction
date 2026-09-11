#!/usr/bin/env python3
"""
Re-predict the original V3 audit-conditioned evaluation sets with V4-Flash.

Purpose
-------
This experiment separates predictor-endpoint effects from evaluation-population
effects without repeating semantic auditing:

1. Reuse the original DeepSeek-V3 Type/Autonomous Audit retained test sets.
2. Reuse their original V3 retained candidate retrieval pools.
3. Regenerate the same 40-shot composition-aware prompts.
4. Call the current DeepSeek endpoint only for downstream prediction.

The experiment therefore contains no audit API calls. Existing prediction files
are checkpointed every 20 completed requests and are resumed automatically.

Examples
--------
Validate source files only:
    python scripts/endpoint_transfer.py --data-only

Generate prompts with the pymatgen-compatible environment:
    python scripts/endpoint_transfer.py --prepare-only

Run/resume prediction:
    python scripts/endpoint_transfer.py --concurrency 30

Retry only failed predictions:
    python scripts/endpoint_transfer.py \
      --concurrency 30 --retry-failures
"""

from __future__ import annotations

import asyncio
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import workflow_core as common


EXPERIMENT_NAME = "v4_on_original_v3_audit_conditioned_results"
STRATEGIES = ("type_audit", "autonomous_audit")

FILES: dict[str, dict[str, Any]] = {
    "sintering": {
        "temp_key": "sintering_temp",
        "type_audit": {
            "test": (
                "2019_sintering_operations_filtered_val_2_"
                "train_1000_llm_audited_bond.json"
            ),
            "pool": (
                "2019_sintering_operations_filtered_val_2_"
                "validation_rest_llm_audited_bond.json"
            ),
            "expected_pool_size": 4002,
        },
        "autonomous_audit": {
            "test": (
                "2019_sintering_operations_filtered_val_2_"
                "train_1000_llm_audited_auto.json"
            ),
            "pool": (
                "2019_sintering_operations_filtered_val_2_"
                "validation_rest_llm_audited_auto.json"
            ),
            "expected_pool_size": 5334,
        },
    },
    "calcination": {
        "temp_key": "calcination_temp",
        "type_audit": {
            "test": (
                "2019_calcination_operations_filtered_val_2_"
                "train_1000_llm_audited_bond.json"
            ),
            "pool": (
                "2019_calcination_operations_filtered_val_2_"
                "validation_rest_llm_audited_bond.json"
            ),
            "expected_pool_size": 5402,
        },
        "autonomous_audit": {
            "test": (
                "2019_calcination_operations_filtered_val_2_"
                "train_1000_llm_audited_auto.json"
            ),
            "pool": (
                "2019_calcination_operations_filtered_val_2_"
                "validation_rest_llm_audited_auto.json"
            ),
            "expected_pool_size": 5935,
        },
    },
}


def prepare_conditioned_prompts(
    operation: str,
    strategy: str,
    temp_key: str,
    test: list[dict[str, Any]],
    pool: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    valid_pool = common.valid_example_pool(pool)
    if len(valid_pool) < k:
        raise ValueError(
            f"{operation}/{strategy}: only {len(valid_pool)} valid examples "
            f"for k={k}."
        )

    prompts: list[dict[str, Any]] = []
    for index, record in enumerate(test):
        reaction = str(record.get("reaction_string") or "").strip()
        temperature = common.finite_float(record.get("target_temperature"))
        if not reaction or temperature is None:
            raise ValueError(
                f"{operation}/{strategy}: invalid query at index {index}."
            )

        examples = common.select_composition_examples(
            record.get("target_material"),
            valid_pool,
            k,
        )
        if len(examples) != k:
            raise ValueError(
                f"{operation}/{strategy}: retrieved {len(examples)} examples "
                f"for query {index}; expected {k}."
            )
        examples_block = "\n\n".join(
            common.format_example(example, operation) for example in examples
        )
        prompts.append(
            {
                "query_id": f"{operation}-{strategy}-{index:04d}",
                "method": f"{strategy}_conditioned",
                "source_doi": record.get("doi"),
                "target": record.get("target_material"),
                "reaction_string": reaction,
                "ground_truth": {temp_key: temperature},
                "prompt": common.build_prompt(
                    operation,
                    temp_key,
                    reaction,
                    examples_block,
                ),
            }
        )
    return prompts


def bootstrap_mae_ci(
    rows: list[dict[str, Any]],
    temp_key: str,
    iterations: int,
    seed: int,
) -> dict[str, float | int]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Bootstrap calculation requires numpy. Install it with "
            "`python -m pip install numpy`."
        ) from exc

    errors = [
        value[2]
        for value in common.result_errors(rows, temp_key).values()
    ]
    if not errors:
        return {
            "n": 0,
            "mae": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }

    values = np.asarray(errors, dtype=float)
    rng = np.random.default_rng(seed)
    sampled_means: list[Any] = []
    remaining = iterations
    while remaining > 0:
        batch_size = min(500, remaining)
        indices = rng.integers(
            0,
            values.size,
            size=(batch_size, values.size),
        )
        sampled_means.append(values[indices].mean(axis=1))
        remaining -= batch_size
    sampled = np.concatenate(sampled_means)
    ci_low, ci_high = np.percentile(sampled, [2.5, 97.5])
    return {
        "n": int(values.size),
        "mae": float(values.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def audit_split_diagnostics(
    test: list[dict[str, Any]],
    pool: list[dict[str, Any]],
) -> dict[str, int]:
    test_dois = {
        common.normalize_doi(record.get("doi"))
        for record in test
        if common.normalize_doi(record.get("doi"))
    }
    pool_dois = {
        common.normalize_doi(record.get("doi"))
        for record in pool
        if common.normalize_doi(record.get("doi"))
    }
    pool_keys = {common.record_key(record) for record in pool}
    return {
        "test_unique_doi": len(test_dois),
        "pool_unique_doi": len(pool_dois),
        "shared_unique_doi": len(test_dois.intersection(pool_dois)),
        "test_records_with_pool_doi": sum(
            common.normalize_doi(record.get("doi")) in pool_dois
            for record in test
            if common.normalize_doi(record.get("doi"))
        ),
        "test_records_exactly_duplicated_in_pool": sum(
            common.record_key(record) in pool_keys for record in test
        ),
    }


def load_common_fixed_metrics(root: Path) -> dict[tuple[str, str], float]:
    path = (
        common.DEFAULT_OUTPUT_ROOT
        / "task1_fixed_test_original_v3_pools_v4_results"
        / "metrics.csv"
    )
    if not path.exists():
        return {}
    values: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                values[(row["task"], row["method"])] = float(row["mae_c"])
            except (KeyError, TypeError, ValueError):
                continue
    return values


async def run(args: Any) -> None:
    common._allow_regex_fallback = args.allow_regex_fallback

    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (common.DEFAULT_OUTPUT_ROOT / EXPERIMENT_NAME).resolve()
    )
    prompt_dir = output_dir / "prompts"
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded: dict[
        tuple[str, str],
        tuple[str, list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    task_manifest: dict[str, dict[str, Any]] = {}

    for operation in args.tasks:
        spec = FILES[operation]
        operation_manifest: dict[str, Any] = {}
        for strategy in STRATEGIES:
            source = spec[strategy]
            test_path = root / source["test"]
            pool_path = root / source["pool"]
            test = common.load_json(test_path)
            pool = common.load_json(pool_path)

            if len(test) != args.test_size:
                raise ValueError(
                    f"{operation}/{strategy}: expected {args.test_size} test "
                    f"records, found {len(test)} in {test_path.name}."
                )
            if len(pool) != source["expected_pool_size"]:
                raise ValueError(
                    f"{operation}/{strategy}: expected "
                    f"{source['expected_pool_size']} pool records, found "
                    f"{len(pool)} in {pool_path.name}."
                )

            temp_key = str(spec["temp_key"])
            loaded[(operation, strategy)] = (temp_key, test, pool)
            diagnostics = audit_split_diagnostics(test, pool)
            operation_manifest[strategy] = {
                "audit_model": "DeepSeek-V3 (historical retained records)",
                "prediction_model_requested": args.model,
                "test_file": test_path.name,
                "candidate_pool_file": pool_path.name,
                "test_size": len(test),
                "candidate_pool_size": len(pool),
                "retrieval": "raw element-overlap, stable pool-order tie-break",
                "k": args.k,
                **diagnostics,
            }
            print(
                f"{operation}/{strategy}: test={len(test)}, pool={len(pool)}"
            )
        task_manifest[operation] = operation_manifest

    manifest = {
        "experiment": EXPERIMENT_NAME,
        "purpose": (
            "Re-predict original V3 audit-conditioned test sets using the "
            "current DeepSeek endpoint; no audit calls are made."
        ),
        "audit_decisions": "historical DeepSeek-V3 retained datasets",
        "seed": args.seed,
        "test_size": args.test_size,
        "k": args.k,
        "requested_prediction_model": args.model,
        "api_base_url": args.api_base_url,
        "temperature": args.temperature,
        "json_mode": not args.disable_json_mode,
        "strategies": list(STRATEGIES),
        "tasks": task_manifest,
    }
    common.dump_json(output_dir / "run_manifest.json", manifest)
    if args.data_only:
        print(f"Data validation completed: {output_dir}")
        return

    all_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for operation in args.tasks:
        for strategy in STRATEGIES:
            temp_key, test, pool = loaded[(operation, strategy)]
            prompt_path = prompt_dir / f"{operation}_{strategy}.json"
            if prompt_path.exists() and not args.overwrite_prompts:
                prompts = common.load_json(prompt_path)
            else:
                print(f"Preparing prompts: {operation}/{strategy}")
                prompts = prepare_conditioned_prompts(
                    operation,
                    strategy,
                    temp_key,
                    test,
                    pool,
                    args.k,
                )
                common.dump_json(prompt_path, prompts)

            if len(prompts) != len(test):
                raise ValueError(
                    f"{operation}/{strategy}: {len(prompts)} prompts for "
                    f"{len(test)} test records."
                )
            if args.prepare_only:
                continue

            result_path = (
                prediction_dir / f"{operation}_{strategy}.json"
            )
            all_results[(operation, strategy)] = (
                await common.run_prediction_file(
                    prompts,
                    result_path,
                    temp_key,
                    args,
                )
            )

    manifest["retrieval_element_parser"] = common._composition_backend
    common.dump_json(output_dir / "run_manifest.json", manifest)
    if args.prepare_only:
        print(f"Prompt preparation completed: {prompt_dir}")
        return

    fixed_metrics = load_common_fixed_metrics(root)
    metric_rows: list[dict[str, Any]] = []
    ci_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for operation in args.tasks:
        for strategy in STRATEGIES:
            temp_key, test, pool = loaded[(operation, strategy)]
            rows = all_results[(operation, strategy)]
            metrics = common.calculate_metrics(rows, temp_key)
            interval = bootstrap_mae_ci(
                rows,
                temp_key,
                args.bootstrap,
                args.seed,
            )
            response_models = sorted(
                {
                    str(row.get("response_model"))
                    for row in rows
                    if row.get("response_model")
                }
            )
            metric_rows.append(
                {
                    "task": operation,
                    "strategy": strategy,
                    "evaluation_population": "original_v3_audit_conditioned",
                    "test_size": len(test),
                    "valid_predictions": metrics["n"],
                    "candidate_pool_size": len(pool),
                    "mae_c": metrics["mae"],
                    "rmse_c": metrics["rmse"],
                    "r2": metrics["r2"],
                    "requested_model": args.model,
                    "response_models": "|".join(response_models),
                }
            )
            ci_rows.append(
                {
                    "task": operation,
                    "strategy": strategy,
                    "n": interval["n"],
                    "mae_c": interval["mae"],
                    "ci_95_low_c": interval["ci_low"],
                    "ci_95_high_c": interval["ci_high"],
                    "iterations": args.bootstrap,
                }
            )

            fixed_mae = fixed_metrics.get((operation, strategy))
            if fixed_mae is not None:
                comparison_rows.append(
                    {
                        "task": operation,
                        "strategy": strategy,
                        "audit_conditioned_mae_c": metrics["mae"],
                        "common_fixed_test_mae_c": fixed_mae,
                        "conditioned_minus_common_mae_c": (
                            float(metrics["mae"]) - fixed_mae
                        ),
                        "comparison_type": (
                            "descriptive_unpaired_different_test_populations"
                        ),
                    }
                )

    common.write_csv(
        output_dir / "metrics.csv",
        metric_rows,
        [
            "task",
            "strategy",
            "evaluation_population",
            "test_size",
            "valid_predictions",
            "candidate_pool_size",
            "mae_c",
            "rmse_c",
            "r2",
            "requested_model",
            "response_models",
        ],
    )
    common.write_csv(
        output_dir / "bootstrap_mae_ci.csv",
        ci_rows,
        [
            "task",
            "strategy",
            "n",
            "mae_c",
            "ci_95_low_c",
            "ci_95_high_c",
            "iterations",
        ],
    )
    if comparison_rows:
        common.write_csv(
            output_dir / "protocol_comparison.csv",
            comparison_rows,
            [
                "task",
                "strategy",
                "audit_conditioned_mae_c",
                "common_fixed_test_mae_c",
                "conditioned_minus_common_mae_c",
                "comparison_type",
            ],
        )

    print(f"Experiment completed: {output_dir}")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Bootstrap intervals: {output_dir / 'bootstrap_mae_ci.csv'}")
    if comparison_rows:
        print(
            "Descriptive protocol comparison: "
            f"{output_dir / 'protocol_comparison.csv'}"
        )


def main() -> None:
    parser = common.build_parser(
        (
            "Re-predict original V3 Type/Autonomous audit-conditioned "
            "evaluation sets using the current DeepSeek endpoint."
        ),
        EXPERIMENT_NAME,
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; completed predictions remain checkpointed.")
        sys.exit(130)
