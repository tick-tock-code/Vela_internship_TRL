# Step 3 PLS Reasoning

Step 3 is the frozen blockwise-PLS reference workspace.

The key distinction from the old model pipeline is that `PLS` is fit on the reasoning block only, inside each outer fold, before the latent components are concatenated with raw `HQ`.

Baseline rule:

- Every method stage must include one frozen `HQ` benchmark on the exact outer CV used by that stage.
- For the current Step 3 `3 x 16` run, that benchmark is:
  - `LR`: `0.207 +/- 0.046`
  - `XGB1`: `0.204 +/- 0.033`

Read order:

- [summary.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_3_PLS_reasoning/summary.md)
- [10_reasoning_block_pls_protocol.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_3_PLS_reasoning/10_reasoning_block_pls_protocol.md)
- [11_reasoning_block_pls_summary.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_3_PLS_reasoning/11_reasoning_block_pls_summary.md)
- [12_reasoning_block_pls_family_details.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_3_PLS_reasoning/12_reasoning_block_pls_family_details.md)

Supporting artifacts:

- [reasoning_block_pls_summary.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_3_PLS_reasoning/reasoning_block_pls_summary.csv)
- [reasoning_block_pls_fold_metrics.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_3_PLS_reasoning/reasoning_block_pls_fold_metrics.csv)
- [reasoning_block_pls_component_summary.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_3_PLS_reasoning/reasoning_block_pls_component_summary.csv)
