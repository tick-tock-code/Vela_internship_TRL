# Step 1 Legacy Alignment

This note compares the new evidence-map rerun against the curated legacy evidence surfaces.

## Model-Testing Alignment

- Parsed legacy summary rows: 9 from `docs/model_testing/summary_final_table.md`.

- [PASS] `LR_BASE_HQ` remains ahead of raw `A` on legacy test.
- [PASS] `LR_PLS_F` remains ahead of `LR_BASE_HQ` on legacy test.
- [PASS] `LR_PLS_DEF` remains ahead of `LR_BASE_HQ` on legacy test.
- [PASS] `LR_PLS_A` does not overtake `LR_BASE_HQ` by default.

## Paper-Pipeline Alignment

- Parsed paper top-pick rows: 7 from `docs/paper_stats/experiment_top_picks.md`.
- The paper-stage raw CV tables still show large apparent gains for richer reasoning combinations.
- That earlier raw-CV optimism is consistent with the current Step 1 framing: raw reasoning signal exists, but does not hold up cleanly without stronger extraction or control methods.
