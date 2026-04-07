# Method Benchmark Scaffold

This file records the active Step 2 method order and the current priority inputs.

- Active methods configured: 1

## Active Methods

- `stability_selection`: Active Step 2 method. Run row-subsampled stability selection on HQ + A-F before any new PLS work.

## Planned Order

- `stability_selection`: Active next method. Use row subsampling on HQ + A-F to measure stable reasoning-feature membership before any new transforms.
- `sparse_group_lasso`: Apply after stability selection if grouped penalties are needed to control redundancy within HQ + reasoning routes.
- `spls`: Revisit after the subsampling route is understood. Treat as a later transform benchmark, not the next implementation step.
- `supervised_grouping`: Only introduce if grouped penalties require data-driven regrouping inside unstable families.
- `llm_guided_selection`: Keep as a late extension after the statistical stability routes are established.

## Step 1 Priority Inputs

- Compression priority: (none)
- Stability-selection priority: (none)
- Grouped-penalty priority: (none)
