# Reproduction guide

## 1. Environment

Create a clean virtual environment and install `requirements.txt`. Export
`DEEPSEEK_API_KEY` only for stages that call the API.

## 2. Data

Copy the files listed in `data/README.md` into `data/full/`. Do not rename them.
The scripts validate record counts, split order, and—in the common-held-out
analysis—occurrence-level removal of the fixed evaluation records.

## 3. Deterministic validation before API use

```bash
python scripts/common_held_out.py --data-only --no-reuse-baselines
python scripts/doi_grouped.py --data-only
python scripts/endpoint_transfer.py --data-only
```

If a validation fails, do not run the API stage. Resolve the indicated filename,
split, count, or ordering mismatch first.

## 4. Prompt preparation

```bash
python scripts/common_held_out.py --prepare-only --no-reuse-baselines
python scripts/doi_grouped.py --prepare-only
python scripts/endpoint_transfer.py --prepare-only
```

Prompt generation requires `pymatgen` for exact composition parsing. The
optional regex fallback is exposed only for diagnostic use and changes the
retrieval implementation for formulas that `pymatgen` and the regex parse
differently.

## 5. Prediction and checkpointing

Run the same commands without `--prepare-only`. Partial predictions are written
to `outputs/` and reused when the command is restarted. Use
`--retry-failures` to retry previously failed prediction calls.

The main current scripts default to `deepseek-chat`. To reproduce an archived
historical model, the corresponding endpoint must still be available and the
exact provider/model identifier must be supplied with `--model` and
`--api-base-url`.

## 6. Statistics

The shared workflow calculates MAE, RMSE, R², and 5,000-iteration paired
bootstrap intervals for common-query comparisons. The endpoint-transfer script
also writes percentile bootstrap confidence intervals for each configuration.

## 7. Result audit trail

Every completed run writes a `run_manifest.json` containing the requested model,
endpoint, seed, evaluation size, retrieval context length, parser, methods, and
task-specific pool metadata. Preserve the manifest next to its CSV outputs.
