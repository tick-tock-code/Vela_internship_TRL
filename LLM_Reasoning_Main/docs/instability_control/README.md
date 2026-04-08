# Instability Control

This folder is the active workspace for the evidence-first rewrite of the reasoning-stability study.

## Stage Order

1. Anchor continuity:
   keep `HQ_anchor_xgb1_unpruned` frozen.
2. Step 1 evidence map:
   restate the raw-route failures and transformed `PLS` wins cleanly.
3. Step 2 stability analysis:
   preserve the row-subsampled stability-selection reference on `HQ + A-F`.
4. Step 3 PLS reasoning:
   fit `PLS` on the reasoning block only, then concatenate those latent components with raw `HQ`.
5. Step 4 supervised grouping:
   learn train-only grouped routes for augmentation and competition views of `HQ + reasoning`.
6. Final synthesis:
   maintain one concise current-state summary of what is raw-failing, transform-sensitive, or not reproducible.

## Required Reading

- [step1/02_protocol.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step1/02_protocol.md)
- [step1/05_hq_anchor_and_route_protocol.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step1/05_hq_anchor_and_route_protocol.md)
- [step1/06_step1_family_diagnostics_and_admission.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step1/06_step1_family_diagnostics_and_admission.md)

## Folder Map

- [00_scope.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/00_scope.md)
- [01_family_map.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/01_family_map.md)
- [04_readings_and_recommended_order.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/04_readings_and_recommended_order.md)
- [step1/README.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step1/README.md)
- [step_2_stability_analysis/README.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_2_stability_analysis/README.md)
- [step_2_stability_analysis/09_step2_stability_selection_method_note.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_2_stability_analysis/09_step2_stability_selection_method_note.md)
- [step_3_PLS_reasoning/README.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_3_PLS_reasoning/README.md)
- [step_4_supervised_grouping/README.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/README.md)
- [current_status.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/current_status.md)
- [figures/README.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/figures/README.md)
- [09_final_report.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/09_final_report.md)
