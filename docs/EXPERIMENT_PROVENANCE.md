# Experiment provenance

This document maps the cleaned repository to the scripts and artifacts in the
historical research archive. It distinguishes preserved historical outputs
from code reconstructed after the original experiments.

| Manuscript analysis | Clean script | Historical source | Archived compact result |
|---|---|---|---|
| Common-held-out comparison | `scripts/common_held_out.py` | `task_1_v3_audited_pools_fixed_test_v4.py` plus `task_1_complement.py` | `results/common_held_out/` |
| Jaccard retrieval control | `scripts/jaccard_comparison.py` | `jaccard_fixed_test_v4.py` | `results/jaccard/` |
| DOI-grouped leakage control | `scripts/doi_grouped.py` | `task_2_complement.py` plus `task_1_complement.py` | `results/doi_grouped/` |
| Predictor-endpoint transfer | `scripts/endpoint_transfer.py` | `v4_on_original_v3_audit_cond.py` plus `task_1_complement.py` | `results/endpoint_transfer/` |
| Shared retrieval, prompts, prediction, metrics, and bootstrap | `scripts/workflow_core.py` | `task_1_complement.py` | Used by the analyses above |

## Original audit-conditioned workflow

The historical research archive retains the DeepSeek-V3 audited partitions used in
the paper. The original source-script mapping is:

- Physical Audit: `training_set_preparation_llm.py` and the corresponding
  calcination variant/output set;
- Type Audit: `training_set_preparation_llm_bond_sinter.py` and
  `training_set_preparation_llm_bond_calc.py`;
- Threshold-Free Autonomous sensitivity: `training_set_preparation_llm_auto.py`;
- prediction prompt construction: `prompt_generation_dy_sinter_val.py` and
  `prompt_generation_dy_calc_val.py`;
- prediction: `prediction_deepseek_acc_val.py` and
  `prediction_deepseek_acc_val_edited.py`;
- evaluation: `evaluation_results_val_sinter.py` and
  `evaluation_results_val_calc.py`.

The retained filenames ending in `_llm_audited_auto.json` establish the
historical Threshold + Autonomous partitions, but the research archive does not
contain one unambiguous surviving source script that both enables the threshold
and writes those exact filenames. The cleaned shared implementation preserves
the manuscript-level method and prompt, but must not be described as the exact
historical execution file.

## Input scope

All three semantic audits use structured fields only:

- target material;
- reaction string;
- intended operation;
- extracted temperature.

No source-text window is passed to the auditor. Consequently, the method audits
structured-field plausibility and consistency; it does not verify extraction
correctness against the original article text.

## LLM reproducibility

LLM services are versioned external dependencies. Even when the requested model
name, prompt, temperature, and seed are unchanged, a provider-side endpoint
update may change outputs. The archived retained partitions, run manifests, and
result CSVs are therefore the authoritative artifacts for the reported numbers.
