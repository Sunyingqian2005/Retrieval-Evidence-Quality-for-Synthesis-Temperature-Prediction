# Retrieval Evidence Quality for Solid-State Synthesis Temperature Prediction

Code and compact result artifacts accompanying the manuscript **“Retrieval
Evidence Quality for Solid-State Synthesis Temperature Prediction:
Composition-Aware Retrieval and LLM Semantic Auditing.”**

The repository separates two functions that are easy to conflate:

1. **relevance**, handled by composition-aware retrieval; and
2. **reliability**, handled by threshold filtering and LLM semantic auditing.

Sintering and calcination are treated as separate prediction tasks. The
semantic auditor receives only the structured material, reaction, intended
operation, and extracted temperature fields. It does **not** receive the source
article paragraph or a source-text window.

## Repository contents

```text
configs/                 Main experimental settings
data/                    Schema, examples, and instructions for local data
docs/                    Reproduction and experiment-provenance notes
prompts/                 Human-readable audit and prediction prompt templates
results/                 Compact CSV results and run manifests used in the paper
scripts/                 Executable experiment code
tests/                   Offline unit tests for core deterministic behavior
```

The original 3.2 GB working directory is intentionally not mirrored here.
Intermediate prompts, API responses, checkpoints, figures, and duplicate
historical scripts are excluded. Full literature-mined records are also not
committed automatically because their redistribution terms must be checked
against the source dataset. See `data/README.md`.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Sunyingqian2005/Retrieval-Evidence-Quality-for-Synthesis-Temperature-Prediction.git
cd Retrieval-Evidence-Quality-for-Synthesis-Temperature-Prediction
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For API-backed experiments, copy the environment template and add the key only
to your local shell or local `.env` file. Never commit an API key.

```bash
cp .env.example .env
export DEEPSEEK_API_KEY=<your-key>
```

The scripts read environment variables directly; they do not automatically
load `.env`.

## Data preparation

Place the required JSON files in `data/full/`, or pass another directory with
`--root /absolute/path/to/data`. The exact historical filenames and their roles
are listed in `data/README.md`.

Verify the local study files against the committed hashes:

```bash
python scripts/verify_data.py \
  --root "/path/to/data-directory"
```

The `train_1000` filenames are historical. In this work those 1,000 records are
evaluation queries; no supervised model training is performed.

## Reproduce the principal analyses

First validate local files without making API calls:

```bash
python scripts/common_held_out.py --data-only --no-reuse-baselines
python scripts/doi_grouped.py --data-only
python scripts/endpoint_transfer.py --data-only
```

Prepare prompts locally:

```bash
python scripts/common_held_out.py --prepare-only --no-reuse-baselines
python scripts/jaccard_comparison.py --prepare-only
python scripts/doi_grouped.py --prepare-only
python scripts/endpoint_transfer.py --prepare-only
```

Run an API-backed experiment, for example:

```bash
python scripts/common_held_out.py \
  --no-reuse-baselines \
  --model deepseek-chat \
  --concurrency 10
```

Outputs are checkpointed under `outputs/`, which is ignored by Git. API calls
may incur charges. Run `--help` on any script to inspect all options.

## Historical-model reproducibility boundary

The main audit decisions were generated with the DeepSeek-V3 endpoint available
at the time of the original study. The retained V3 partitions are the archival
objects used by `common_held_out.py` and `endpoint_transfer.py`. The surviving
historical research archive does not contain a complete, unambiguous
threshold-enabled source script for the original Autonomous Audit run.
`workflow_core.py` preserves the
reported prompt, confidence rule, threshold, fail-open handling, split seed,
retrieval rule, and prediction procedure, but a present-day API rerun cannot be
expected to reproduce historical LLM decisions byte-for-byte.

The committed CSV files report the archived manuscript results. They are not
silently regenerated or overwritten by the example commands.

For a function-by-function mapping from the original `training`, `prompt`,
`prediction`, and `evaluation` scripts to the consolidated implementation, see
`docs/ORIGINAL_CODE_MAPPING.md`.

## Verification

Offline checks do not require the dataset or an API key:

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
```

## Citation

Citation metadata is provided in `CITATION.cff`. Add the article DOI after
publication.

## License

The code is released under the MIT License. Dataset records and third-party
model outputs remain subject to their respective source terms.
