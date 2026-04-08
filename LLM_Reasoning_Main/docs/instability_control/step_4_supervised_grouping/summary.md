# Step 4 Summary

Step 4 tests supervised grouping as a train-only compression route. The method is most useful in the augmentation track, where grouped reasoning features are appended to raw `HQ`. It is not useful as a full competition-track replacement for `HQ + reasoning`.

Frozen `HQ` on the Step 4 `3 x 16` outer CV was:

- `LR`: `0.207 +/- 0.046`
- `XGB1`: `0.204 +/- 0.033`

Main positive results:

- `HQ + A`, augmentation, `3` groups per family:
  `XGB1` improved from `0.298` to `0.308` (`+0.009`), while `LR` was effectively flat at `0.302 -> 0.301`.
- `HQ + A+B+C+D+E`, augmentation, `2` groups per family:
  `LR` improved from `0.304` to `0.318` (`+0.014`), which was the strongest gain in Step 4.
- `HQ + B+C+D+E+F`, augmentation, `2` groups per family:
  `XGB1` improved from `0.215` to `0.221` (`+0.006`).

Mostly neutral results:

- `HQ + F`, augmentation:
  the grouped routes were close to raw, with best-case `XGB1` moving from `0.206` to `0.209` and `LR` staying around `0.227`.
- `HQ + A+B+C+D+E`, augmentation, `3` groups per family:
  both models were near-flat, `LR 0.304 -> 0.306`, `XGB1 0.302 -> 0.304`.

Negative results:

- competition-track grouping was usually worse than the raw comparison route.
- `HQ + A+B+C+D+E`, competition, `6` groups:
  `LR` fell from `0.304` to `0.267`, `XGB1` from `0.302` to `0.282`.
- `HQ + B+C+D+E+F`, competition, `6` groups:
  `LR` fell from `0.252` to `0.214`.
- even grouped `HQ` alone was slightly weaker than frozen `HQ`.

Structural interpretation:

- `A` grouped very cleanly. The stable `2`-group layout was:
  `{A_career_coherence, A_evidence_support_rating, A_trajectory_strength}`
  and `{A_ownership_signal, A_scrappiness}`.
- `F` also grouped cleanly, but the grouped route did not materially improve `LR`, and only slightly helped `XGB1`.
- in the competition track, the model often formed one very large mixed cluster containing most reasoning features, which is consistent with the weaker results there.

Current takeaway:

- supervised grouping looks promising as an augmentation route for selected reasoning sets;
- it does not look promising as a full competition-track replacement for `HQ + reasoning`;
- the most interesting Step 4 candidates are:
  - `HQ + A`, augmentation, `3` groups, `XGB1`
  - `HQ + A+B+C+D+E`, augmentation, `2` groups, `LR`

This is still a CV-only result. It suggests cleaner internal behaviour, but it does not yet show improved out-of-sample generalisability.
