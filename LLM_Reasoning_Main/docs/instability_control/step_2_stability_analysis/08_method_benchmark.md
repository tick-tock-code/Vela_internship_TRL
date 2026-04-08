# Method Benchmark Scaffold

This file records the active instability-control method order and the current priority inputs.

- Active methods configured: 3

## Active Methods

- `stability_selection`: Completed Step 2 method. Use row-subsampled stability selection on HQ + A-F as the frozen stability-analysis reference.
- `reasoning_block_pls`: Completed Step 3 method. Fit leakage-safe PLS on the reasoning block only, then concatenate the latent components with raw HQ.
- `supervised_grouping`: Active Step 4 method. Learn train-only supervised clusters, collapse each cluster to a 1-component PLS latent, then compare grouped augmentation and grouped competition routes.

## Planned Order

- `stability_selection`: Frozen Step 2 reference. Use row subsampling on HQ + A-F to measure stable reasoning-feature membership.
- `reasoning_block_pls`: Completed Step 3 route. Compress the reasoning block inside each outer fold before concatenating it with HQ.
- `supervised_grouping`: Active Step 4 route. Use supervised train-only grouping to collapse correlated HQ + reasoning features into compact grouped routes.
- `sparse_group_lasso`: Apply after supervised grouping if grouped penalties are still needed to control redundancy within HQ + reasoning routes.
- `spls`: Revisit after supervised grouping is understood. Treat as a later transform benchmark, not the active next step.
- `llm_guided_selection`: Keep as a late extension after the statistical stability routes are established.

## Step 1 Priority Inputs

- Compression priority: (none)
- Stability-selection priority: (none)
- Grouped-penalty priority: (none)
