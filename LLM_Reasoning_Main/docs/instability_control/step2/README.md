# Step 2

Step 2 is the active method-testing area. The current method is row-subsampled stability selection on `HQ + family`.

Current read order:

- [09_step2_stability_selection_method_note.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step2/09_step2_stability_selection_method_note.md)
- [10_stability_selection_protocol.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step2/10_stability_selection_protocol.md)
- [11_stability_selection_summary.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step2/11_stability_selection_summary.md)
- [12_stability_selection_family_details.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step2/12_stability_selection_family_details.md)
- [13_stability_selection_final_audit.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step2/13_stability_selection_final_audit.md)

Supporting files:

- [stability_selection_summary.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step2/stability_selection_summary.csv)
- [stability_selection_feature_frequencies.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step2/stability_selection_feature_frequencies.csv)
- [08_method_benchmark.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step2/08_method_benchmark.md)

Minimal document roles:

- `09`: what the experiment is actually doing
- `10`: exact protocol and fixed settings
- `11`: high-level results table
- `12`: per-family selector behavior
- `13`: external audit placeholder

Method order:

1. `stability_selection` on `HQ + A-F`
2. `sparse_group_lasso`
3. `sPLS`
4. supervised regrouping
5. LLM-guided grouping or penalty design only after the statistical routes are established
