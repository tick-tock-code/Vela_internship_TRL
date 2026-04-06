# Instability-Control Scope

This workspace is the active path for the new project direction.

The unit of analysis is the feature family, not the one-off prompt run or concatenated feature dump.

The operating questions are:

1. Which LLM-derived families add stable incremental signal over the HQ human baseline?
2. Which families collapse out-of-sample because they are redundant, unstable, or poorly calibrated?
3. Which model routes preserve gains without validation-to-holdout collapse?

The active code lives under `python/lib/stability/` and `python/pipelines/instability_control/`.
