# Instability-Control Scope

This workspace is the active path for the new project direction.

The unit of analysis is the feature family or fixed legacy combo, not the one-off prompt run or concatenated feature dump.

The operating questions are:

1. Which reasoning units fail in raw form relative to the frozen unpruned HQ `XGB1` anchor?
2. Which units only become useful after structured extraction such as `PLS`?
3. Which mathematical methods should be tried next to improve stability without redefining the baseline?

Baseline rule:

- the canonical benchmark anchor is the unpruned mirror HQ `XGB1` model
- route-level pruning or transformation must not redefine that anchor

The active code lives under `python/lib/stability/` and `python/pipelines/instability_control/`.
