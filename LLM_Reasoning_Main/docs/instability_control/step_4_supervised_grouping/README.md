# Step 4 Supervised Grouping

Step 4 is the active grouped-routing workspace.

This stage clusters features inside each outer training fold, then collapses each cluster to one latent feature using `1`-component `PLS`.

Baseline rule:

- Every method stage must include one frozen `HQ` benchmark on the exact outer CV used by that stage.
- Step 4 uses the same `3 x 16` outer CV as Step 3.

Read order:

- [summary.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/summary.md)
- [10_supervised_grouping_protocol.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/10_supervised_grouping_protocol.md)
- [11_supervised_grouping_summary.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/11_supervised_grouping_summary.md)
- [12_supervised_grouping_details.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/12_supervised_grouping_details.md)

Supporting artifacts:

- [supervised_grouping_summary.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/supervised_grouping_summary.csv)
- [supervised_grouping_fold_metrics.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/supervised_grouping_fold_metrics.csv)
- [supervised_grouping_cluster_assignments.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/supervised_grouping_cluster_assignments.csv)
- [supervised_grouping_component_summary.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/supervised_grouping_component_summary.csv)
- [supervised_grouping_cocluster_summary.csv](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step_4_supervised_grouping/supervised_grouping_cocluster_summary.csv)
