# Family Map

Current Step 1 evaluation units:

## Atomic Families

- `HQ_anchor_xgb1_unpruned`: canonical frozen mirror benchmark anchor
- `engineered_set_05`
- `reasoning_A`
- `reasoning_B`
- `reasoning_C`
- `reasoning_D`
- `reasoning_E`
- `reasoning_F`

## Fixed Legacy Combos

- `HQ`
- `A+C`
- `D+E+F`
- `C+D+E+F`
- `A+B+C+D+E+F`

## Route Controls

Step 1 uses route controls rather than route baselines with admission logic:

- `anchor_xgb1_unpruned`
- `raw_lr_base`
- `raw_xgb1`
- `pls_lr_n6`
- `pls_mlp4_n6`

Pruned or compressed HQ variants are route variants, not new baselines.
