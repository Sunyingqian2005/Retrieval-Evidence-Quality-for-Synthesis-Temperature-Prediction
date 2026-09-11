#!/usr/bin/env python3
"""
Task 1: fixed held-out-set retrieval experiment.

The same unaudited seed-42 test set is used for every configuration. Only the
candidate retrieval pool and example-selection method are allowed to change.

Configurations
--------------
1. fixed_random       : fixed 40-shot examples from the unaudited pool
2. composition_aware  : element-overlap retrieval from the unaudited pool
3. threshold          : element-overlap retrieval after 300--2200 °C filtering
4. physical_audit     : Threshold + Physical Audit
5. type_audit         : Threshold + Type Audit
6. autonomous_audit   : Threshold + Autonomous Audit

The 1000-record query set is fixed before any reliability control. Threshold
filtering and all semantic audits are applied only to the remaining candidate
pool.

The script can:
* verify and materialize the controlled datasets;
* generate prompts;
* call the DeepSeek endpoint with checkpoint/resume support;
* calculate MAE, RMSE, R², and paired bootstrap MAE differences.

Examples
--------
python scripts/workflow_core.py --data-only
python scripts/workflow_core.py --prepare-only
python scripts/workflow_core.py
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import heapq
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data" / "full"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "outputs"

TASK_SPECS: dict[str, dict[str, str]] = {
    "sintering": {
        "source": "2019_sintering_operations_filtered_val_2.json",
        "fixed_test": (
            "2019_sintering_operations_filtered_val_2_"
            "train_1000_filtered_val_nofilter.json"
        ),
        "fixed_pool": (
            "2019_sintering_operations_filtered_val_2_"
            "validation_rest_filtered_val_nofilter.json"
        ),
        "temp_key": "sintering_temp",
    },
    "calcination": {
        "source": "2019_calcination_operations_filtered_val_2.json",
        "fixed_test": (
            "2019_calcination_operations_filtered_val_2_"
            "train_1000_filtered_val_nofilter.json"
        ),
        "fixed_pool": (
            "2019_calcination_operations_filtered_val_2_"
            "validation_rest_filtered_val_nofilter.json"
        ),
        "temp_key": "calcination_temp",
    },
}

METHOD_CONFIG: dict[str, tuple[str, str]] = {
    "fixed_random": ("raw", "fixed"),
    "composition_aware": ("raw", "composition"),
    "threshold": ("threshold", "composition"),
    "physical_audit": ("physical_audit", "composition"),
    "type_audit": ("type_audit", "composition"),
    "autonomous_audit": ("autonomous_audit", "composition"),
}

TASK1_METHODS = [
    "fixed_random",
    "composition_aware",
    "threshold",
    "physical_audit",
    "type_audit",
    "autonomous_audit",
]

TASK1_COMPARISONS = [
    ("composition_aware", "fixed_random"),
    ("threshold", "composition_aware"),
    ("physical_audit", "composition_aware"),
    ("type_audit", "composition_aware"),
    ("autonomous_audit", "composition_aware"),
    ("physical_audit", "threshold"),
    ("type_audit", "threshold"),
    ("autonomous_audit", "threshold"),
]

TASK1_AUDITS = [
    "physical_audit",
    "type_audit",
    "autonomous_audit",
]


@dataclass
class TaskInput:
    name: str
    temp_key: str
    test: list[dict[str, Any]]
    pools: dict[str, list[dict[str, Any]]]
    metadata: dict[str, Any]


def build_parser(description: str, default_output_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: <repository>/outputs/"
            f"{default_output_name}"
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASK_SPECS),
        default=sorted(TASK_SPECS),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--api-base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Only construct and validate test/pool files.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Construct fixed data and run/reuse audits, then stop.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Construct datasets and prompts, but do not call the API.",
    )
    parser.add_argument(
        "--overwrite-prompts",
        action="store_true",
        help="Regenerate prompt files even if they already exist.",
    )
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry records previously saved with status=Failure.",
    )
    parser.add_argument(
        "--retry-audit-failures",
        action="store_true",
        help="Retry audit records previously retained because all calls failed.",
    )
    parser.add_argument(
        "--overwrite-audits",
        action="store_true",
        help="Discard existing audit checkpoints and audit every candidate again.",
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
            "Allow a strict periodic-table regex parser when pymatgen is "
            "unavailable. Leave disabled for exact consistency with the "
            "original retrieval implementation."
        ),
    )
    return parser


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def record_key(record: dict[str, Any]) -> str:
    return json.dumps(
        record,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def normalize_doi(value: Any) -> str:
    doi = str(value or "").strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    return doi.strip()


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def valid_example_pool(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("reaction_string")
        and finite_float(record.get("target_temperature")) is not None
    ]


def doi_overlap_summary(
    test: list[dict[str, Any]],
    pool: list[dict[str, Any]],
) -> dict[str, int]:
    pool_dois = {normalize_doi(record.get("doi")) for record in pool}
    pool_keys = {record_key(record) for record in pool}
    return {
        "test_unique_doi": len(
            {normalize_doi(record.get("doi")) for record in test}
        ),
        "pool_unique_doi": len(pool_dois),
        "test_records_with_pool_doi": sum(
            normalize_doi(record.get("doi")) in pool_dois for record in test
        ),
        "test_records_exactly_duplicated_in_pool": sum(
            record_key(record) in pool_keys for record in test
        ),
    }


def build_fixed_inputs(
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
        test = load_json(root / spec["fixed_test"])
        raw_pool = load_json(root / spec["fixed_pool"])

        expected = list(source)
        random.Random(seed).shuffle(expected)
        if test != expected[:test_size] or raw_pool != expected[test_size:]:
            raise ValueError(
                f"{task_name}: existing fixed files are not the exact seed-{seed} "
                f"split of {spec['source']}."
            )
        if len(test) != test_size:
            raise ValueError(
                f"{task_name}: expected {test_size} test records, found {len(test)}."
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
        overlap = doi_overlap_summary(test, raw_pool)
        metadata = {
            "split": "record-level random split",
            "seed": seed,
            "source_file": spec["source"],
            "test_source_file": spec["fixed_test"],
            "raw_pool_source_file": spec["fixed_pool"],
            "test_size": len(test),
            "source_size": len(source),
            "pool_sizes": {name: len(pool) for name, pool in pools.items()},
            **overlap,
        }

        task_dir = data_dir / task_name
        dump_json(task_dir / "test.json", test)
        for pool_name, pool in pools.items():
            dump_json(task_dir / f"pool_{pool_name}.json", pool)
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


def audit_prompt(
    strategy: str,
    operation: str,
    record: dict[str, Any],
) -> str:
    target = record.get("target_material", "Unknown")
    reaction = record.get("reaction_string", "N/A")
    temperature = record.get("target_temperature")
    category = record.get("operation_category", operation.capitalize())

    if strategy == "physical_audit":
        return f"""You are a materials science expert. Judge the physical plausibility of this synthesis record:
- Material: {target}
- Reaction: {reaction}
- Operation: {category}
- Temperature: {temperature} °C

Is this temperature scientifically plausible for this operation and material system in a solid-state synthesis context?
Respond ONLY in JSON format: {{"plausible": true/false, "reason": "brief explanation"}}"""

    if strategy == "type_audit" and operation == "sintering":
        return f"""You are a materials science expert specializing in solid-state synthesis. 
Judge the physical plausibility of this final sintering record:

- Material System: {target}
- Reaction: {reaction}
- Intended Operation: {category}
- Reported Temperature: {temperature} °C

Evaluation Guidelines:
1. THERMAL ENERGY: Does the temperature provide sufficient activation energy for the intended phase formation or densification? (Consider typical ranges: Sintering usually requires significant atomic diffusion).
2. MATERIAL LIMITS: Is the temperature significantly above the melting point or below the minimum reactive threshold for this material system?
3. ANOMALY DETECTION: Does the temperature (e.g., extremely low values < 300°C) statistically align better with secondary steps like drying or solvent removal rather than the main '{category}' stage?

Respond ONLY in JSON format:
{{
    "plausible": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "A brief scientific justification based on thermodynamics or kinetics, considering if specialized techniques might justify this value."
}}"""

    if strategy == "type_audit" and operation == "calcination":
        return f"""You are a materials science expert specializing in solid-state synthesis. 
Judge the physical plausibility of this final calcination record:

- Material System: {target}
- Reaction: {reaction}
- Intended Operation: {category}
- Reported Temperature: {temperature} °C

Evaluation Guidelines:
1. THERMAL ENERGY: Does the temperature provide sufficient energy for precursor decomposition (e.g., removal of CO2 from carbonates, H2O from hydroxides) or initial phase nucleation? (Note: Calcination is typically a pre-reaction step to transform precursors into oxides).
2. REACTION LIMITS: Is the temperature high enough to trigger the desired solid-state reaction but low enough to avoid premature sintering or excessive grain growth that could hinder subsequent processing?
3. ANOMALY DETECTION: Does the temperature (e.g., values < 300°C) statistically align better with drying, solvent removal, or degasification rather than the '{category}' stage, which usually requires breaking chemical bonds in precursors?

Respond ONLY in JSON format:
{{
    "plausible": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "A brief scientific justification based on thermodynamics or kinetics, considering if specialized techniques might justify this value."
}}"""

    if strategy == "autonomous_audit":
        return f"""You are a materials science expert. Your task is to evaluate the physical plausibility of a synthesis step.

[Context]
Material: {target}
Reaction: {reaction}
Operation: {category}
Temperature: {temperature} °C

[Task]
1. First, briefly define the key physical/chemical criteria that should be used to judge this specific material system and operation.
2. Then, evaluate if the reported temperature of {temperature} °C is scientifically plausible based on those criteria.
3. Consider if specialized methods might justify an "atypical" temperature.

Respond ONLY in JSON format:
{{
    "self_generated_guidelines": ["list of criteria you defined"],
    "plausible": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "Scientific justification based on your defined criteria."
}}"""

    raise ValueError(f"Unsupported audit strategy: {strategy}/{operation}")


def record_digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(record_key(record).encode("utf-8")).hexdigest()


def interpret_audit_decision(
    strategy: str,
    parsed: dict[str, Any] | None,
) -> tuple[bool, str, bool | None, float | None]:
    if not parsed or not isinstance(parsed.get("plausible"), bool):
        return True, "RetainedOnMalformedDecision", None, None

    plausible = parsed["plausible"]
    if strategy == "physical_audit":
        return plausible, "Success", plausible, None

    confidence = finite_float(parsed.get("confidence_score"))
    if confidence is None or not 0.0 <= confidence <= 1.0:
        return True, "RetainedOnMalformedDecision", plausible, None
    retained = plausible or confidence < 0.7
    return retained, "Success", plausible, confidence


async def audit_one(
    candidate_id: str,
    record: dict[str, Any],
    strategy: str,
    operation: str,
    client: Any,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = audit_prompt(strategy, operation, record)
    last_error: str | None = None
    last_response: str | None = None

    async with semaphore:
        for attempt in range(1, args.max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": (
                        0.0 if strategy == "physical_audit" else 0.1
                    ),
                    "timeout": 30.0,
                }
                if not args.disable_json_mode:
                    request["response_format"] = {"type": "json_object"}
                response = await client.chat.completions.create(**request)
                last_response = response.choices[0].message.content
                parsed = extract_json(last_response)
                retained, status, plausible, confidence = (
                    interpret_audit_decision(strategy, parsed)
                )
                return {
                    "candidate_id": candidate_id,
                    "record_sha256": record_digest(record),
                    "strategy": strategy,
                    "retained": retained,
                    "plausible": plausible,
                    "confidence_score": confidence,
                    "status": status,
                    "parsed_response": parsed,
                    "raw_response": last_response,
                    "requested_model": args.model,
                    "response_model": getattr(response, "model", None),
                    "attempts": attempt,
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                await asyncio.sleep(1)

    # Conservative failure handling: an API/format failure never removes data.
    return {
        "candidate_id": candidate_id,
        "record_sha256": record_digest(record),
        "strategy": strategy,
        "retained": True,
        "plausible": None,
        "confidence_score": None,
        "status": "RetainedOnFailure",
        "parsed_response": None,
        "raw_response": last_response,
        "requested_model": args.model,
        "response_model": None,
        "attempts": args.max_retries,
        "error": last_error,
    }


async def run_audit_strategy(
    task: TaskInput,
    strategy: str,
    audit_dir: Path,
    data_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates = task.pools["threshold"]
    audit_path = audit_dir / f"{task.name}_{strategy}.json"

    existing: list[dict[str, Any]] = []
    if audit_path.exists() and not args.overwrite_audits:
        existing = load_json(audit_path)
    retained_existing = [
        row
        for row in existing
        if not (
            args.retry_audit_failures
            and row.get("status") in {
                "RetainedOnFailure",
                "RetainedOnMalformedDecision",
            }
        )
    ]
    existing_by_id = {
        str(row.get("candidate_id")): row
        for row in retained_existing
        if row.get("candidate_id")
    }

    pending: list[tuple[str, dict[str, Any]]] = []
    for index, record in enumerate(candidates):
        candidate_id = f"{task.name}-{index:05d}"
        previous = existing_by_id.get(candidate_id)
        if previous and previous.get("record_sha256") == record_digest(record):
            continue
        pending.append((candidate_id, record))

    print(
        f"Audit {task.name}/{strategy}: "
        f"{len(existing_by_id)} restored, {len(pending)} pending"
    )

    results = list(existing_by_id.values())
    if pending:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {args.api_key_env} is not set."
            )
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Audit calls require openai. Install it with "
                "`python -m pip install openai`."
            ) from exc

        client = AsyncOpenAI(api_key=api_key, base_url=args.api_base_url)
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            audit_one(
                candidate_id,
                record,
                strategy,
                task.name,
                client,
                semaphore,
                args,
            )
            for candidate_id, record in pending
        ]
        completed = 0
        for future in asyncio.as_completed(tasks):
            results.append(await future)
            completed += 1
            if completed % 20 == 0:
                results.sort(key=lambda row: str(row.get("candidate_id")))
                dump_json(audit_path, results)
                print(f"  completed {completed}/{len(tasks)}")

    results.sort(key=lambda row: str(row.get("candidate_id")))
    dump_json(audit_path, results)
    result_by_id = {
        str(row["candidate_id"]): row
        for row in results
        if row.get("candidate_id")
    }

    retained_pool: list[dict[str, Any]] = []
    for index, record in enumerate(candidates):
        candidate_id = f"{task.name}-{index:05d}"
        decision = result_by_id.get(candidate_id)
        if decision is None:
            raise RuntimeError(
                f"Missing audit decision for {task.name}/{strategy}/"
                f"{candidate_id}."
            )
        if decision.get("record_sha256") != record_digest(record):
            raise RuntimeError(
                f"Audit checkpoint no longer matches {candidate_id}."
            )
        if decision.get("retained", True):
            retained_pool.append(record)

    task.pools[strategy] = retained_pool
    task.metadata.setdefault("audit_results", {})[strategy] = {
        "audited_records": len(candidates),
        "retained_records": len(retained_pool),
        "rejected_records": len(candidates) - len(retained_pool),
        "retained_on_failure": sum(
            row.get("status") == "RetainedOnFailure" for row in results
        ),
        "retained_on_malformed_decision": sum(
            row.get("status") == "RetainedOnMalformedDecision"
            for row in results
        ),
        "requested_model": args.model,
        "response_models": sorted(
            {
                str(row.get("response_model"))
                for row in results
                if row.get("response_model")
            }
        ),
    }
    task.metadata["pool_sizes"][strategy] = len(retained_pool)
    task_dir = data_dir / task.name
    dump_json(task_dir / f"pool_{strategy}.json", retained_pool)
    dump_json(task_dir / "manifest.json", task.metadata)
    return retained_pool


_composition_class: Any = None
_composition_backend: str | None = None
_allow_regex_fallback = False
_element_cache: dict[str, frozenset[str] | None] = {}

_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


def regex_formula_elements(text: str) -> frozenset[str] | None:
    symbols = {
        symbol
        for symbol in re.findall(r"[A-Z][a-z]?", text)
        if symbol in _ELEMENT_SYMBOLS
    }
    return frozenset(symbols) if symbols else None


def formula_elements(formula: Any) -> frozenset[str] | None:
    global _composition_class, _composition_backend
    text = str(formula or "").strip()
    if not text:
        return None
    if text in _element_cache:
        return _element_cache[text]
    if _composition_backend is None:
        try:
            from pymatgen.core import Composition
            # Force one parse here because binary incompatibilities in scipy or
            # numpy can surface during import of pymatgen.core.
            Composition("LiFePO4")
            _composition_class = Composition
            _composition_backend = "pymatgen"
        except Exception as exc:
            if not _allow_regex_fallback:
                raise RuntimeError(
                    "pymatgen is unavailable or binary-incompatible. For exact "
                    "consistency with the original retrieval method, generate "
                    "prompts with the compatible environment:\n"
                    "  python <script> --prepare-only\n"
                    "Then run the same script with the API-enabled environment; "
                    "existing prompts will be reused. Use "
                    "--allow-regex-fallback only if this methodological change "
                    "is intentional."
                ) from exc
            _composition_class = False
            _composition_backend = "periodic-table regex fallback"
            print(
                "Warning: pymatgen is unavailable or incompatible "
                f"({type(exc).__name__}: {exc}). Using the built-in strict "
                "element-symbol parser."
            )

    if _composition_class:
        try:
            elements = frozenset(
                element.symbol for element in _composition_class(text).elements
            )
        except Exception:
            elements = regex_formula_elements(text)
    else:
        elements = regex_formula_elements(text)
    _element_cache[text] = elements
    return elements


def select_composition_examples(
    target_formula: Any,
    pool: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    target_elements = formula_elements(target_formula)
    if not target_elements:
        return pool[: min(k, len(pool))]

    ranked: list[tuple[int, int, dict[str, Any]]] = []
    for index, example in enumerate(pool):
        example_elements = formula_elements(example.get("target_material"))
        if not example_elements:
            continue
        overlap = len(target_elements.intersection(example_elements))
        ranked.append((-overlap, index, example))

    selected = heapq.nsmallest(
        min(k, len(ranked)),
        ranked,
        key=lambda row: (row[0], row[1]),
    )
    return [row[2] for row in selected]


def format_example(record: dict[str, Any], operation: str) -> str:
    temperature = finite_float(record.get("target_temperature"))
    if temperature is None:
        raise ValueError("Example has no valid target_temperature.")
    return (
        f"Reaction: {record['reaction_string']}\n"
        f"{operation.capitalize()} Temperature (°C): {temperature:.1f}"
    )


def build_prompt(
    operation: str,
    temp_key: str,
    reaction: str,
    examples_block: str,
) -> str:
    return f"""You are a computational chemistry expert specializing in solid-state synthesis.
Assume there is only one {operation} step involved.

Your task is to predict the optimal {operation} temperature for the following chemical reaction:
{reaction}

**Required Condition to Predict**:
1. {operation.capitalize()} Temperature (in °C)

**Guidelines for Prediction**:
- Base your predictions on established solid-state chemistry principles.
- Provide a single, scientifically plausible temperature value.
- Assume there is only one {operation} step involved.

**Examples of Synthesis Conditions:**
{examples_block}

**Output Format:**
Respond ONLY with a JSON object. Do not include any explanation.
Example:
{{
  "{temp_key}": 1100.0
}}"""


def prepare_prompts(
    task: TaskInput,
    method: str,
    k: int,
    seed: int,
) -> list[dict[str, Any]]:
    pool_key, selection = METHOD_CONFIG[method]
    pool = valid_example_pool(task.pools[pool_key])
    if len(pool) < k:
        raise ValueError(
            f"{task.name}/{method}: only {len(pool)} valid examples for k={k}."
        )

    fixed_block: str | None = None
    if selection == "fixed":
        fixed_examples = random.Random(seed).sample(pool, k)
        fixed_block = "\n\n".join(
            format_example(example, task.name) for example in fixed_examples
        )

    prompts: list[dict[str, Any]] = []
    for index, record in enumerate(task.test):
        reaction = str(record.get("reaction_string") or "").strip()
        temperature = finite_float(record.get("target_temperature"))
        if not reaction or temperature is None:
            raise ValueError(
                f"{task.name}: invalid query record at test index {index}."
            )

        if selection == "fixed":
            examples_block = fixed_block or ""
        else:
            examples = select_composition_examples(
                record.get("target_material"),
                pool,
                k,
            )
            if len(examples) != k:
                raise ValueError(
                    f"{task.name}/{method}: retrieved {len(examples)} examples "
                    f"for test index {index}; expected {k}."
                )
            examples_block = "\n\n".join(
                format_example(example, task.name) for example in examples
            )

        prompts.append(
            {
                "query_id": f"{task.name}-{index:04d}",
                "method": method,
                "source_doi": record.get("doi"),
                "target": record.get("target_material"),
                "reaction_string": reaction,
                "ground_truth": {task.temp_key: temperature},
                "prompt": build_prompt(
                    task.name,
                    task.temp_key,
                    reaction,
                    examples_block,
                ),
            }
        )

    return prompts


def extract_json(text: Any) -> dict[str, Any] | None:
    if not text:
        return None
    value = str(text).strip()
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", value)
    if match:
        candidate = match.group(1)
    else:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end < start:
            return None
        candidate = value[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


async def predict_one(
    prompt_record: dict[str, Any],
    temp_key: str,
    client: Any,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> dict[str, Any]:
    last_error: str | None = None
    last_response: str | None = None

    async with semaphore:
        for attempt in range(1, args.max_retries + 1):
            try:
                request: dict[str, Any] = {
                    "model": args.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant designed to output "
                                "strict JSON."
                            ),
                        },
                        {"role": "user", "content": prompt_record["prompt"]},
                    ],
                    "temperature": args.temperature,
                }
                if not args.disable_json_mode:
                    request["response_format"] = {"type": "json_object"}

                response = await client.chat.completions.create(**request)
                last_response = response.choices[0].message.content
                prediction = extract_json(last_response)
                predicted_value = (
                    finite_float(prediction.get(temp_key))
                    if prediction
                    else None
                )
                if prediction is not None and predicted_value is not None:
                    return {
                        "query_id": prompt_record["query_id"],
                        "method": prompt_record["method"],
                        "source_doi": prompt_record.get("source_doi"),
                        "target": prompt_record.get("target"),
                        "reaction_string": prompt_record.get("reaction_string"),
                        "ground_truth": prompt_record["ground_truth"],
                        "prediction": prediction,
                        "status": "Success",
                        "requested_model": args.model,
                        "response_model": getattr(response, "model", None),
                        "attempts": attempt,
                        "raw_response": last_response,
                    }
                last_error = f"Valid JSON did not contain numeric `{temp_key}`."
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                await asyncio.sleep(1)

    return {
        "query_id": prompt_record["query_id"],
        "method": prompt_record["method"],
        "source_doi": prompt_record.get("source_doi"),
        "target": prompt_record.get("target"),
        "reaction_string": prompt_record.get("reaction_string"),
        "ground_truth": prompt_record["ground_truth"],
        "prediction": None,
        "status": "Failure",
        "requested_model": args.model,
        "response_model": None,
        "attempts": args.max_retries,
        "raw_response": last_response,
        "error": last_error,
    }


async def run_prediction_file(
    prompts: list[dict[str, Any]],
    result_path: Path,
    temp_key: str,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Environment variable {args.api_key_env} is not set."
        )
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "API prediction requires openai. Install it with "
            "`python -m pip install openai`."
        ) from exc

    existing: list[dict[str, Any]] = []
    if result_path.exists():
        existing = load_json(result_path)

    retained_existing = [
        row
        for row in existing
        if not (args.retry_failures and row.get("status") != "Success")
    ]
    processed = {row.get("query_id") for row in retained_existing}
    pending = [row for row in prompts if row["query_id"] not in processed]

    print(
        f"  {result_path.stem}: {len(retained_existing)} restored, "
        f"{len(pending)} pending"
    )
    if not pending:
        return retained_existing

    client = AsyncOpenAI(api_key=api_key, base_url=args.api_base_url)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        predict_one(row, temp_key, client, semaphore, args)
        for row in pending
    ]
    results = list(retained_existing)

    completed = 0
    for future in asyncio.as_completed(tasks):
        results.append(await future)
        completed += 1
        if completed % 20 == 0:
            results.sort(key=lambda row: str(row.get("query_id")))
            dump_json(result_path, results)
            print(f"    completed {completed}/{len(tasks)}")

    results.sort(key=lambda row: str(row.get("query_id")))
    dump_json(result_path, results)
    return results


def result_errors(
    rows: list[dict[str, Any]],
    temp_key: str,
) -> dict[str, tuple[float, float, float]]:
    values: dict[str, tuple[float, float, float]] = {}
    for row in rows:
        if row.get("status") != "Success":
            continue
        true_value = finite_float(
            (row.get("ground_truth") or {}).get(temp_key)
        )
        predicted_value = finite_float(
            (row.get("prediction") or {}).get(temp_key)
        )
        query_id = row.get("query_id")
        if true_value is None or predicted_value is None or not query_id:
            continue
        values[str(query_id)] = (
            true_value,
            predicted_value,
            abs(predicted_value - true_value),
        )
    return values


def calculate_metrics(
    rows: list[dict[str, Any]],
    temp_key: str,
) -> dict[str, float | int]:
    values = list(result_errors(rows, temp_key).values())
    if not values:
        return {"n": 0, "mae": math.nan, "rmse": math.nan, "r2": math.nan}
    true_values = [row[0] for row in values]
    predicted_values = [row[1] for row in values]
    residuals = [
        predicted - true
        for true, predicted in zip(true_values, predicted_values)
    ]
    mae = sum(abs(value) for value in residuals) / len(residuals)
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    mean_true = sum(true_values) / len(true_values)
    denominator = sum((value - mean_true) ** 2 for value in true_values)
    r2 = (
        1.0 - sum(value * value for value in residuals) / denominator
        if denominator > 0
        else math.nan
    )
    return {"n": len(values), "mae": mae, "rmse": rmse, "r2": r2}


def paired_bootstrap(
    method_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
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

    method = result_errors(method_rows, temp_key)
    reference = result_errors(reference_rows, temp_key)
    common_ids = sorted(set(method).intersection(reference))
    if not common_ids:
        return {
            "n_common": 0,
            "delta_mae": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
        }

    differences = np.asarray(
        [method[query_id][2] - reference[query_id][2] for query_id in common_ids],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    bootstrap_means: list[Any] = []
    remaining = iterations
    while remaining > 0:
        batch_size = min(500, remaining)
        indices = rng.integers(
            0,
            differences.size,
            size=(batch_size, differences.size),
        )
        bootstrap_means.append(differences[indices].mean(axis=1))
        remaining -= batch_size
    sampled = np.concatenate(bootstrap_means)
    ci_low, ci_high = np.percentile(sampled, [2.5, 97.5])
    return {
        "n_common": len(common_ids),
        "delta_mae": float(differences.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


async def run_experiment(
    args: argparse.Namespace,
    builder: Callable[
        [Path, Path, list[str], int, int],
        list[TaskInput],
    ],
    methods: list[str],
    comparisons: list[tuple[str, str]],
    audit_strategies: list[str],
    experiment_name: str,
) -> None:
    global _allow_regex_fallback
    _allow_regex_fallback = args.allow_regex_fallback

    root = args.root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (DEFAULT_OUTPUT_ROOT / experiment_name).resolve()
    )
    data_dir = output_dir / "data"
    audit_dir = output_dir / "audits"
    prompt_dir = output_dir / "prompts"
    prediction_dir = output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    task_inputs = builder(
        root,
        data_dir,
        args.tasks,
        args.seed,
        args.test_size,
    )
    print(f"Data validation completed: {output_dir}")
    for task in task_inputs:
        print(
            f"  {task.name}: test={len(task.test)}, "
            + ", ".join(
                f"{name}={len(pool)}" for name, pool in task.pools.items()
            )
        )

    manifest_path = output_dir / "run_manifest.json"
    previous_element_parser: str | None = None
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                previous_manifest = json.load(handle)
            if isinstance(previous_manifest, dict):
                previous_element_parser = previous_manifest.get(
                    "retrieval_element_parser"
                )
        except (OSError, json.JSONDecodeError):
            previous_element_parser = None

    run_manifest = {
        "experiment": experiment_name,
        "seed": args.seed,
        "test_size": args.test_size,
        "k": args.k,
        "requested_model": args.model,
        "api_base_url": args.api_base_url,
        "temperature": args.temperature,
        "json_mode": not args.disable_json_mode,
        "retrieval_element_parser": previous_element_parser,
        "methods": methods,
        "audit_strategies": audit_strategies,
        "tasks": {task.name: task.metadata for task in task_inputs},
    }
    dump_json(manifest_path, run_manifest)
    if args.data_only:
        return

    for task in task_inputs:
        for strategy in audit_strategies:
            await run_audit_strategy(
                task,
                strategy,
                audit_dir,
                data_dir,
                args,
            )
        print(
            f"  {task.name} audited pools: "
            + ", ".join(
                f"{strategy}={len(task.pools[strategy])}"
                for strategy in audit_strategies
            )
        )

    run_manifest["tasks"] = {
        task.name: task.metadata for task in task_inputs
    }
    dump_json(manifest_path, run_manifest)
    if args.audit_only:
        print(f"Audit stage completed: {audit_dir}")
        return

    all_results: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for task in task_inputs:
        for method in methods:
            prompt_path = prompt_dir / f"{task.name}_{method}.json"
            if prompt_path.exists() and not args.overwrite_prompts:
                prompts = load_json(prompt_path)
            else:
                print(f"Preparing prompts: {task.name}/{method}")
                prompts = prepare_prompts(task, method, args.k, args.seed)
                dump_json(prompt_path, prompts)
            if len(prompts) != len(task.test):
                raise ValueError(
                    f"{task.name}/{method}: prompt count {len(prompts)} does "
                    f"not equal test size {len(task.test)}."
                )

            if args.prepare_only:
                continue

            result_path = prediction_dir / f"{task.name}_{method}.json"
            results = await run_prediction_file(
                prompts,
                result_path,
                task.temp_key,
                args,
            )
            all_results[(task.name, method)] = results

    if args.prepare_only:
        run_manifest["retrieval_element_parser"] = (
            _composition_backend or previous_element_parser
        )
        dump_json(manifest_path, run_manifest)
        print(f"Prompt preparation completed: {prompt_dir}")
        return

    metric_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for task in task_inputs:
        raw_size = len(task.pools["raw"])
        for method in methods:
            pool_key, _ = METHOD_CONFIG[method]
            metrics = calculate_metrics(
                all_results[(task.name, method)],
                task.temp_key,
            )
            metric_rows.append(
                {
                    "task": task.name,
                    "method": method,
                    "test_size": len(task.test),
                    "valid_predictions": metrics["n"],
                    "candidate_pool_size": len(task.pools[pool_key]),
                    "retention_rate": len(task.pools[pool_key]) / raw_size,
                    "mae_c": metrics["mae"],
                    "rmse_c": metrics["rmse"],
                    "r2": metrics["r2"],
                }
            )

        for method, reference in comparisons:
            statistics = paired_bootstrap(
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

    write_csv(
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
    write_csv(
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
    run_manifest["retrieval_element_parser"] = (
        _composition_backend or previous_element_parser
    )
    dump_json(manifest_path, run_manifest)
    print(f"Experiment completed: {output_dir}")
    print(f"Metrics: {output_dir / 'metrics.csv'}")
    print(f"Paired bootstrap: {output_dir / 'paired_bootstrap.csv'}")


def main() -> None:
    parser = build_parser(
        "Task 1: fixed 1000-record held-out-set full Section 4.2 comparison",
        "task1_fixed_test_full_4_2_v4_results",
    )
    args = parser.parse_args()
    asyncio.run(
        run_experiment(
            args,
            build_fixed_inputs,
            TASK1_METHODS,
            TASK1_COMPARISONS,
            TASK1_AUDITS,
            "task1_fixed_test_full_4_2_v4_results",
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted; completed predictions remain checkpointed.")
        sys.exit(130)
