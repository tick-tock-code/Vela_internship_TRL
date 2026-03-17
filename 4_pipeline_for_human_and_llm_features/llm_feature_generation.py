"""LLM feature generation for VCBench (mirrors example script behavior)."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.datasets import VCBENCH_HELPERS, VCBENCH_SCHEMA
from think_reason_learn.features import FeatureEvaluator, FeatureGenerator


async def generate_llm_features(
    train_recs: list[dict[str, Any]],
    y_train: Iterable[int],
    test_recs: list[dict[str, Any]],
    model: str,
    n_features: int,
    all_recs: list[dict[str, Any]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Generate LLM features exactly like the example script.

    Returns (l_all, l_train, l_test, llm_names).
    """
    generator = FeatureGenerator(
        schema=VCBENCH_SCHEMA,
        helpers=VCBENCH_HELPERS,
        llm_priority=[OpenAIChoice(model=model)],
        temperature=0.7,
    )
    rules = await generator.generate(
        samples=train_recs,
        labels=list(y_train),
        n_rules=n_features,
        n_samples=min(60, len(train_recs)),
    )
    if not rules:
        raise RuntimeError("No LLM rules generated. Check API key.")

    evaluator = FeatureEvaluator(rules=rules, helpers=VCBENCH_HELPERS)
    errors = evaluator.compilation_errors
    if errors:
        for name, err in errors.items():
            print(f"  WARNING: {name} failed: {err}")

    l_train = evaluator.evaluate_df(train_recs)
    l_test = evaluator.evaluate_df(test_recs)
    llm_names = [r.name for r in rules]

    if all_recs is None:
        l_all = pd.DataFrame(index=range(len(train_recs) + len(test_recs)))
    else:
        l_all = evaluator.evaluate_df(all_recs)

    ok = len(rules) - len(errors)
    print(f"  Generated {len(rules)} rules ({ok} compiled OK)")
    for i, r in enumerate(rules, 1):
        print(f"    {i}. {r.name}: {r.description}")

    return l_all, l_train, l_test, llm_names
