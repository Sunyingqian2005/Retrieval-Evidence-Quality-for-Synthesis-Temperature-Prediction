# Mapping from the original research scripts

The cleaned repository consolidates repeated logic instead of uploading every
historical copy. The original stages remain represented as follows.

| Original research code | Clean implementation | Role |
|---|---|---|
| `training_set_preparation_llm.py` | `workflow_core.audit_prompt()` and `run_audit_strategy()` | Physical Audit |
| `training_set_preparation_llm_bond_sinter.py` | `workflow_core.audit_prompt()` | Type Audit for sintering |
| `training_set_preparation_llm_bond_calc.py` | `workflow_core.audit_prompt()` | Type Audit for calcination |
| `training_set_preparation_llm_auto.py` | `workflow_core.audit_prompt()` and the `autonomous_audit` strategy | Autonomous prompt and confidence rule; the surviving original script is the threshold-free sensitivity version |
| `prompt_generation_dy_sinter_val.py` | `workflow_core.prepare_prompts()` and `build_prompt()` | composition-aware 40-shot sintering prompts |
| `prompt_generation_dy_calc_val.py` | `workflow_core.prepare_prompts()` and `build_prompt()` | composition-aware 40-shot calcination prompts |
| `prediction_deepseek_acc_val.py` and `prediction_deepseek_acc_val_edited.py` | `workflow_core.run_prediction_file()` | API prediction, JSON parsing, retry, and checkpointing |
| `evaluation_results_val_sinter.py` and `evaluation_results_val_calc.py` | `workflow_core.calculate_metrics()` and `paired_bootstrap()` | MAE, RMSE, R², and confidence intervals |

The historical files are not copied verbatim because they contain duplicated
logic, hard-coded working-directory assumptions, inconsistent output names, and
obsolete exploratory branches. The clean functions preserve the method-level
behavior used in the manuscript while the archived result CSVs and DeepSeek-V3
retained partitions preserve the numerical audit trail.

The endpoint-sensitivity scripts under the original
`multi-llm_llm_auto_pipeline/` directory are not the source of the main
DeepSeek-V3 Autonomous result and are therefore not included as principal
reproduction scripts.
