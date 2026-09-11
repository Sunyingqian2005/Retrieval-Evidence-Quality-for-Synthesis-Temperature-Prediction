#!/usr/bin/env python3
"""
Normalized element-set Jaccard retrieval on the fixed common-held-out sets.

This experiment reuses the exact seed-42, 1000-record test sets and unaudited
candidate pools used by ``common_held_out.py``. It changes
only the example-selection rule:

    J(A, B) = |A intersect B| / |A union B|

where A and B are the sets of constituent elements in the query and candidate
target compositions.  Candidates are ranked by descending Jaccard similarity;
ties are resolved by their existing candidate-pool order.  The top 40 examples
are inserted into the same prediction prompt used by the common-held-out
experiment.

No audit calls are made.  The script:

1. validates and materializes the exact fixed test/pool files;
2. generates Jaccard-retrieval prompts with retrieved indices and scores;
3. calls the DeepSeek OpenAI-compatible endpoint with checkpoint/resume;
4. recalculates fixed-random, element-overlap, and Jaccard metrics;
5. performs 5000 paired bootstrap comparisons.

The ``openai`` package is only the API client.  Requests are sent to the
DeepSeek base URL and use ``DEEPSEEK_API_KEY``.

Examples
--------
Generate and inspect prompts without API calls:

    python scripts/jaccard_comparison.py \
        --prepare-only

Run predictions, resuming automatically if interrupted:

    python scripts/jaccard_comparison.py --concurrency 30

Retry only previously failed prediction records:

    python scripts/jaccard_comparison.py --concurrency 30 --retry-failures
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

import workflow_core as common


DEFAULT_ROOT = common.DEFAULT_DATA_ROOT
DEFAULT_BASELINE_EXPERIMENT = (
    "task1_fixed_test_original_v3_pools_v4_results"
)
DEFAULT_OUTPUT_NAME = "jaccard_fixed_test_v4_results"
METHOD = "jaccard"
BASELINE_METHODS = ("fixed_random", "composition_aware")
COMPARISONS = (
    ("composition_aware", "fixed_random"),
    ("jaccard", "fixed_random"),
    ("jaccard", "composition_aware"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalized element-set Jaccard retrieval on the fixed 1000-record "
            "common-held-out sets."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help=(
            "Completed fixed-test baseline experiment. Default: "
            "<repository>/outputs/task1_fixed_test_original_v3_pools_v4_results"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <repository>/outputs/jaccard_fixed_test_v4_results",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(common.TASK_SPECS),
        default=sorted(common.TASK_SPECS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Validate and copy the exact fixed test/pool files, then stop.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate Jaccard prompts without making API calls.",
    )
    parser.add_argument(
        "--overwrite-prompts",
        action="store_true",
        help="Regenerate prompt files even when they already exist.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry prediction rows previously saved with status=Failure.",
    )
    parser.add_argument(
        "--disable-json-mode",
        action="store_true",
        help="Do not send response_format=json_object.",
    )
    parser.add_argument(
        "--allow-regex-fallback",
        action="store_true",
        help=(
            "Use the strict periodic-table regex parser if pymatgen is "
            "unavailable. Leave disabled for consistency with the original "
            "composition retriever."
        ),
    )
    parser.add_argument(
        "--allow-model-mismatch",
        action="store_true",
        help=(
            "Write metrics even if the API reports a response model different "
            "from the completed fixed-test baseline. Without this flag, a "
            "model mismatch stops final comparison."
        ),
    )
    return parser


def select_jaccard_examples(
    target_formula: Any,
    pool: list[dict[str, Any]],
    k: int,
) -> list[tuple[int, float, dict[str, Any]]]:
    """Return top-k candidates by normalized element-set Jaccard similarity."""
    target_elements = common.formula_elements(target_formula)
    if not target_elements:
        return [
            (index, 0.0, record)
            for index, record in enumerate(pool[: min(k, len(pool))])
        ]

    ranked: list[tuple[float, int, dict[str, Any]]] = []
    for index, record in enumerate(pool):
        candidate_elements = common.formula_elements(
            record.get("target_material")
        )
        if not candidate_elements:
            continue
        union = target_elements.union(candidate_elements)
        score = (
            len(target_elements.intersection(candidate_elements)) / len(union)
            if union
            else 0.0
        )
        ranked.append((-score, index, record))

    ranked.sort(key=lambda row: (row[0], row[1]))
    selected = ranked[: min(k, len(ranked))]
    return [(index, -negative_score, record) for negative_score, index, record in selected]


def prepare_jaccard_prompts(
    task: common.TaskInput,
    k: int,
) -> list[dict[str, Any]]:
    pool = common.valid_example_pool(task.pools["raw"])
    if len(pool) < k:
        raise ValueError(
            f"{task.name}/jaccard: only {len(pool)} valid examples for k={k}."
        )

    prompts: list[dict[str, Any]] = []
    for index, record in enumerate(task.test):
        reaction = str(record.get("reaction_string") or "").strip()
        temperature = common.finite_float(record.get("target_temperature"))
        if not reaction or temperature is None:
            raise ValueError(
                f"{task.name}: invalid query record at test index {index}."
            )

        selected = select_jaccard_examples(
            record.get("target_material"),
            pool,
            k,
        )
        if len(selected) != k:
            raise ValueError(
                f"{task.name}/jaccard: retrieved {len(selected)} examples for "
                f"test index {index}; expected {k}."
            )

        examples_block = "\n\n".join(
            common.format_example(example, task.name)
            for _, _, example in selected
        )
        prompts.append(
            {
                "query_id": f"{task.name}-{index:04d}",
                "method": METHOD,
                "source_doi": record.get("doi"),
                "target": record.get("target_material"),
                "reaction_string": reaction,
                "ground_truth": {task.temp_key: temperature},
                "retrieval_metric": "normalized_element_set_jaccard",
                "retrieved_candidate_indices": [
                    candidate_index for candidate_index, _, _ in selected
                ],
                "retrieved_jaccard_scores": [
                    round(score, 12) for _, score, _ in selected
                ],
                "prompt": common.build_prompt(
                    task.name,
                    task.temp_key,
                    reaction,
                    examples_block,
                ),
            }
        )

    return prompts


def validate_prediction_queries(
    rows: list[dict[str, Any]],
    task: common.TaskInput,
    label: str,
) -> None:
    """Verify that a prediction file uses the exact fixed held-out queries."""
    if len(rows) != len(task.test):
        raise ValueError(
            f"{task.name}/{label}: expected {len(task.test)} prediction rows, "
            f"found {len(rows)}."
        )
    by_id = {str(row.get("query_id")): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"{task.name}/{label}: duplicate or missing query IDs.")

    for index, record in enumerate(task.test):
        query_id = f"{task.name}-{index:04d}"
        row = by_id.get(query_id)
        if row is None:
            raise ValueError(f"{task.name}/{label}: missing {query_id}.")
        expected_temperature = common.finite_float(
            record.get("target_temperature")
        )
        observed_temperature = common.finite_float(
            (row.get("ground_truth") or {}).get(task.temp_key)
        )
        if observed_temperature != expected_temperature:
            raise ValueError(
                f"{task.name}/{label}/{query_id}: ground truth mismatch."
            )
        if str(row.get("reaction_string") or "").strip() != str(
            record.get("reaction_string") or ""
        ).strip():
            raise ValueError(
                f"{task.name}/{label}/{query_id}: reaction mismatch."
            )


def response_models(rows: list[dict[str, Any]]) -> set[str]:
    return {
        str(row.get("response_model"))
        for row in rows
        if row.get("status") == "Success" and row.get("response_model")
    }


def metric_row(
    task: common.TaskInput,
    method: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = common.calculate_metrics(rows, task.temp_key)
    return {
        "task": task.name,
        "method": method,
        "test_size": len(task.test),
        "valid_predictions": metrics["n"],
        "candidate_pool_size": len(task.pools["raw"]),
        "retention_rate": 1.0,
        "mae_c": metrics["mae"],
        "rmse_c": metrics["rmse"],
        "r2": metrics["r2"],
    }


async def run(args: argparse.Namespace) -> None:
    if args.k != 40:
        raise ValueError(
            "This controlled baseline script requires --k 40 because the "
            "reused fixed-random and composition-aware predictions are 40-shot."
        )
    common._allow_regex_fallback = args.allow_regex_fallback

    root = args.root.resolve()
    baseline_dir = (
        args.baseline_dir.resolve()
        if args.baseline_dir
        else (common.DEFAULT_OUTPUT_ROOT / DEFAULT_BASELINE_EXPERIMENT).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (common.DEFAULT_OUTPUT_ROOT / DEFAULT_OUTPUT_NAME).resolve()
    )
    data_dir = output_dir / "data"
    prompt_dir = output_dir / "prompts"
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_inputs = common.build_fixed_inputs(
        root,
        data_dir,
        args.tasks,
        args.seed,
        args.test_size,
    )
    print(f"Fixed-test validation completed: {output_dir}")
    for task in task_inputs:
        print(
            f"  {task.name}: test={len(task.test)}, "
            f"raw_pool={len(task.pools['raw'])}"
        )

    manifest_path = output_dir / "run_manifest.json"
    previous_parser: str | None = None
    if manifest_path.exists():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous_parser = previous.get("retrieval_element_parser")
        except (OSError, json.JSONDecodeError, AttributeError):
            previous_parser = None

    manifest: dict[str, Any] = {
        "experiment": DEFAULT_OUTPUT_NAME,
        "purpose": (
            "Normalized element-set Jaccard retrieval baseline on the exact "
            "fixed common-held-out evaluation records."
        ),
        "seed": args.seed,
        "test_size": args.test_size,
        "k": args.k,
        "retrieval_metric": "|A intersection B| / |A union B|",
        "tie_break": "stable candidate-pool order",
        "candidate_pool": "unaudited fixed raw pool",
        "baseline_experiment": str(baseline_dir),
        "requested_model": args.model,
        "api_base_url": args.api_base_url,
        "temperature": args.temperature,
        "json_mode": not args.disable_json_mode,
        "retrieval_element_parser": previous_parser,
        "tasks": {task.name: task.metadata for task in task_inputs},
    }
    common.dump_json(manifest_path, manifest)
    if args.data_only:
        return

    prompts_by_task: dict[str, list[dict[str, Any]]] = {}
    for task in task_inputs:
        prompt_path = prompt_dir / f"{task.name}_{METHOD}.json"
        if prompt_path.exists() and not args.overwrite_prompts:
            prompts = common.load_json(prompt_path)
        else:
            print(f"Preparing Jaccard prompts: {task.name}")
            prompts = prepare_jaccard_prompts(task, args.k)
            common.dump_json(prompt_path, prompts)
        if len(prompts) != len(task.test):
            raise ValueError(
                f"{task.name}/jaccard: prompt count {len(prompts)} does not "
                f"equal test size {len(task.test)}."
            )
        validate_prediction_queries(prompts, task, "jaccard_prompts")
        prompts_by_task[task.name] = prompts

    manifest["retrieval_element_parser"] = (
        common._composition_backend or previous_parser
    )
    common.dump_json(manifest_path, manifest)
    if args.prepare_only:
        print(f"Prompt preparation completed: {prompt_dir}")
        return

    all_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    baseline_models: set[str] = set()
    for task in task_inputs:
        for method in BASELINE_METHODS:
            baseline_path = (
                baseline_dir
                / "predictions"
                / f"{task.name}_{method}.json"
            )
            rows = common.load_json(baseline_path)
            validate_prediction_queries(rows, task, method)
            all_results[(task.name, method)] = rows
            baseline_models.update(response_models(rows))

        result_path = prediction_dir / f"{task.name}_{METHOD}.json"
        results = await common.run_prediction_file(
            prompts_by_task[task.name],
            result_path,
            task.temp_key,
            args,
        )
        validate_prediction_queries(results, task, METHOD)
        all_results[(task.name, METHOD)] = results

    jaccard_models: set[str] = set()
    for task in task_inputs:
        jaccard_models.update(
            response_models(all_results[(task.name, METHOD)])
        )
    manifest["baseline_response_models"] = sorted(baseline_models)
    manifest["jaccard_response_models"] = sorted(jaccard_models)
    common.dump_json(manifest_path, manifest)

    if (
        baseline_models
        and jaccard_models
        and baseline_models != jaccard_models
        and not args.allow_model_mismatch
    ):
        raise RuntimeError(
            "Response-model mismatch: completed baselines used "
            f"{sorted(baseline_models)}, while Jaccard used "
            f"{sorted(jaccard_models)}. Re-run with a matching endpoint or use "
            "--allow-model-mismatch to write descriptive results explicitly."
        )

    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for task in task_inputs:
        for method in (*BASELINE_METHODS, METHOD):
            metric_rows.append(
                metric_row(
                    task,
                    method,
                    all_results[(task.name, method)],
                )
            )

        for method, reference in COMPARISONS:
            statistics = common.paired_bootstrap(
                all_results[(task.name, method)],
                all_results[(task.name, reference)],
                task.temp_key,
                args.bootstrap,
                args.seed,
            )
            bootstrap_rows.append(
                {
                    "task": task.name,
                    "method": method,
                    "reference": reference,
                    "n_common": statistics["n_common"],
                    "delta_mae_c": statistics["delta_mae"],
                    "ci_95_low_c": statistics["ci_low"],
                    "ci_95_high_c": statistics["ci_high"],
                    "iterations": args.bootstrap,
                    "significant_better": (
                        statistics["ci_high"] < 0
                        if math.isfinite(float(statistics["ci_high"]))
                        else False
                    ),
                }
            )

    common.write_csv(
        output_dir / "metrics.csv",
        metric_rows,
        [
            "task",
            "method",
            "test_size",
            "valid_predictions",
            "candidate_pool_size",
            "retention_rate",
            "mae_c",
            "rmse_c",
            "r2",
        ],
    )
    common.write_csv(
        output_dir / "paired_bootstrap.csv",
        bootstrap_rows,
        [
            "task",
            "method",
            "reference",
            "n_common",
            "delta_mae_c",
            "ci_95_low_c",
            "ci_95_high_c",
            "iterations",
            "significant_better",
        ],
    )
    print(f"Experiment completed: {output_dir}")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Paired bootstrap: {output_dir / 'paired_bootstrap.csv'}")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; completed predictions remain checkpointed.")
        sys.exit(130)
