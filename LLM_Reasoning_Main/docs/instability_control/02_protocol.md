# Protocol

Default workflow for the instability-control path:

1. Run `family_diagnostics.py` to estimate redundancy, cross-baseline overlap, and stable incremental lift.
2. Run `family_admission.py` to admit families sequentially under explicit stability thresholds.
3. Run `route_comparison.py` on the admitted family stack.
4. Run `final_report.py` to regenerate the curated summary in this folder.

Raw outputs should land in `.tmp/runs/instability_control/`.
