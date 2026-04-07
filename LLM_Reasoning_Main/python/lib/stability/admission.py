from __future__ import annotations


def run_sequential_admission(*args, **kwargs):
    raise RuntimeError(
        "run_sequential_admission is deprecated. Step 1 now uses evidence mapping rather than admission. "
        "Use python/pipelines/instability_control/evidence_map.py instead."
    )
