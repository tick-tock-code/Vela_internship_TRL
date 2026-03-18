"""Tests for prompt evolution guards (no API calls)."""

import numpy as np
import pandas as pd

from prompt_evolution import _build_critic_payload, _validate_mutation


def test_validate_mutation_rejects_schema_markers() -> None:
    ok, _ = _validate_mutation("Output keys: rubric_score, rubric_score_justification")
    assert not ok


def test_validate_mutation_accepts_plain_rubric() -> None:
    ok, _ = _validate_mutation(
        "Score founders based on evidence of ownership and progression. "
        "Be consistent and justify scores in one sentence."
    )
    assert ok


def test_build_critic_payload_strips_labels() -> None:
    df = pd.DataFrame(
        {
            "success": [1, 0],
            "rubric_score": [3, 4],
            "rubric_score_justification": ["a", "b"],
            "evidence_support_rating": [2, 3],
        }
    )
    outputs, summary = _build_critic_payload(
        df=df,
        numeric_keys=["evidence_support_rating", "rubric_score"],
        sample_size=2,
        rng=np.random.RandomState(0),
    )
    for item in outputs:
        assert "success" not in item
    assert "numeric" in summary
