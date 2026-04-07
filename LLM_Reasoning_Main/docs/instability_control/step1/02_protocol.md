# Protocol

Default workflow for the instability-control path:

1. Run anchor continuity checks.
   Keep `HQ_anchor_xgb1_unpruned` fixed and use unpruned mirror-style `XGB1` only as the benchmark continuity route.
2. Run `evidence_map.py`.
   This is Step 1. It evaluates atomic families and fixed legacy combos across the canonical raw and transformed routes.
3. Review the Step 1 evidence artifacts.
   Use `step1_route_metrics`, `step1_overlap_metrics`, the synthesis note, and the legacy-alignment note to determine where the signal appears raw, where it only appears after transformation, and where it is not reproducible.
4. Run `method_benchmark.py`.
   This pass is scaffold-only for now, but it locks the order of the next mathematical methods.
5. Run `final_report.py`.
   This writes the high-level synthesis for the current instability-control state.

## Canonical Step 1 Routes

- `anchor_xgb1_unpruned`
- `raw_lr_base`
- `raw_xgb1`
- `pls_lr_n6`
- `pls_mlp4_n6`

The `pls_mlp4_n6` route is audit-only and does not drive the main classification logic.

## Required Step 1 Outputs

- `step1_route_metrics.csv/json`
- `step1_overlap_metrics.csv/json`
- `step1_synthesis.md`
- `step1_legacy_alignment.md`

## Working Rules

- `HQ_anchor_xgb1_unpruned` is the only canonical benchmark baseline.
- Pruned or compressed HQ variants are route variants, not baselines.
- Step 1 is evidence mapping, not sequential family admission.
- New mathematical methods should not be added until the Step 1 evidence map has been refreshed and reviewed.
