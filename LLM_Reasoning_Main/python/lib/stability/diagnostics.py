from __future__ import annotations


def run_family_diagnostics(*args, **kwargs):
    raise RuntimeError(
        "run_family_diagnostics is deprecated. Use python/pipelines/instability_control/evidence_map.py "
        "or lib.stability.evidence.run_step1_evidence_map instead."
    )
