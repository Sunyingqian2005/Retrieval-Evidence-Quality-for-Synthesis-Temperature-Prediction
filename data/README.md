# Data preparation

`data/full/` is ignored by Git. Put local copies of the following files there,
or pass their parent directory to each script with `--root`.

## Base operation records

- `2019_sintering_operations_filtered_val_2.json`
- `2019_calcination_operations_filtered_val_2.json`

## Fixed seed-42 common-held-out split

- `2019_sintering_operations_filtered_val_2_train_1000_filtered_val_nofilter.json`
- `2019_sintering_operations_filtered_val_2_validation_rest_filtered_val_nofilter.json`
- `2019_calcination_operations_filtered_val_2_train_1000_filtered_val_nofilter.json`
- `2019_calcination_operations_filtered_val_2_validation_rest_filtered_val_nofilter.json`

Despite their historical names, each `train_1000` file is the 1,000-record
evaluation set and each `validation_rest` file is the candidate retrieval pool.

## Archived DeepSeek-V3 audited partitions

Physical Audit:

- `2019_sintering_operations_filtered_val_2_train_1000_llm_audited.json`
- `2019_sintering_operations_filtered_val_2_validation_rest_llm_audited.json`
- `2019_calcination_operations_filtered_val_train_1000_llm_audited.json`
- `2019_calcination_operations_filtered_val_validation_rest_llm_audited.json`

Type Audit:

- `2019_sintering_operations_filtered_val_2_train_1000_llm_audited_bond.json`
- `2019_sintering_operations_filtered_val_2_validation_rest_llm_audited_bond.json`
- `2019_calcination_operations_filtered_val_2_train_1000_llm_audited_bond.json`
- `2019_calcination_operations_filtered_val_2_validation_rest_llm_audited_bond.json`

Autonomous Audit:

- `2019_sintering_operations_filtered_val_2_train_1000_llm_audited_auto.json`
- `2019_sintering_operations_filtered_val_2_validation_rest_llm_audited_auto.json`
- `2019_calcination_operations_filtered_val_2_train_1000_llm_audited_auto.json`
- `2019_calcination_operations_filtered_val_2_validation_rest_llm_audited_auto.json`

## Record schema

The executable code expects these fields:

- `doi`
- `target_material`
- `reaction_string`
- `operation_category`
- `operation_string_raw` (optional for modeling)
- `target_temperature`
- `units`

See `schema.json` and `example_records.json` for a machine-readable schema and
two synthetic-format examples. The examples illustrate structure only and are
not a substitute for the study dataset.

`file_manifest.csv` records the exact byte size, JSON record count, and SHA-256
digest of each local file used by the cleaned reproduction scripts.

## Redistribution note

The operation records were derived from the literature-mined solid-state
synthesis dataset reported by Kononova et al. Before placing complete records
in a public repository, confirm that the upstream license permits redistribution
of the derived records. If not, publish the preprocessing instructions, hashes,
and compact result tables here, and deposit redistributable research artifacts
in an appropriate archival repository.
