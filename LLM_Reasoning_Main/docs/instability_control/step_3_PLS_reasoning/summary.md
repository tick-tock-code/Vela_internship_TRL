# Step 3 Summary

Step 3 is mainly a representation-screening result, not yet a generalisability result.

The current `3 x 16` CV outputs show whether blockwise reasoning-only `PLS` changes internal behaviour when the compressed reasoning block is added back to raw `HQ`. They do not yet show whether validation-to-test collapse is reduced.

## Main Read

- `A` is effectively a negative result. Blockwise `PLS` does not rescue `HQ + A`.
  - Raw `LR`: `0.302`
  - Best blockwise `LR`: `0.299` at `k = 3`
  - Raw `XGB1`: `0.298`
  - Best blockwise `XGB1`: `0.297` at `k = 3`
- `F` is the clearest positive Step 3 signal.
  - Raw `LR`: `0.227`
  - Best blockwise `LR`: `0.227` at `k = 2`
  - Raw `XGB1`: `0.206`
  - Best blockwise `XGB1`: `0.241` at `k = 3`
- `D` shows a mild positive signal under `XGB1`.
  - Raw `XGB1`: `0.216`
  - Best blockwise `XGB1`: `0.226` at `k = 2`
- `C` is weak or ambiguous.
  - `LR` is flat
  - `XGB1` improves slightly: `0.208 -> 0.215`
- `B` and `E` show no useful evidence.

## Interpretation

- Blockwise reasoning-only `PLS` is not a broad fix.
- For `A`, the same tension remains after compression: the family still contains mixed directions, and compression does not turn that into a cleaner CV result.
- For `F`, and to a lesser extent `D`, the `XGB1` improvement suggests that compressing the reasoning block may be making the signal more consistent for some internal metrics.
- That is still only an internal-CV result. It is best treated as a hint that these families may benefit from structured compression, not as evidence that generalisability has improved.

## Practical Outcome

- Carry forward:
  - `HQ + F` blockwise `PLS`, especially `XGB1`, `k = 3`
  - `HQ + D` blockwise `PLS`, especially `XGB1`, `k = 2`
- Keep as an explicit negative/control:
  - `HQ + A` blockwise `PLS`
- Do not claim from Step 3 alone that stability or out-of-sample robustness has improved.

The current value of Step 3 is that it narrows the next candidates. It does not settle the generalisation question.
