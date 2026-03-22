"""LLM feature generation for VCBench (mirrors example script behavior)."""

from __future__ import annotations

from typing import Any, Iterable
import os
from pathlib import Path

import pandas as pd

from think_reason_learn.core.llms import OpenAIChoice, GoogleChoice
from think_reason_learn.datasets import VCBENCH_HELPERS, VCBENCH_SCHEMA
from think_reason_learn.features import FeatureEvaluator, FeatureGenerator
from lib.paths import PROJECT_ROOT


async def generate_llm_features(
    train_recs: list[dict[str, Any]],
    y_train: Iterable[int],
    test_recs: list[dict[str, Any]],
    model: str,
    n_features: int,
    all_recs: list[dict[str, Any]] | None = None,
    providers: dict[str, bool] | None = None,
    google_model: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[dict[str, str]]]:
    """Generate LLM features exactly like the example script.

    Returns (l_all, l_train, l_test, llm_names, llm_rules).
    """
    _load_env_if_present()
    try:
        from think_reason_learn.core._config import settings as trl_settings
        if os.getenv("OPENAI_API_KEY"):
            trl_settings.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        # Reset LLM singleton so it re-reads updated settings
        from think_reason_learn.core.llms._ask import LLM
        from think_reason_learn.core._singleton import SingletonMeta
        SingletonMeta._instances.pop(LLM, None)
        import think_reason_learn.core.llms as trl_llms
        trl_llms.llm = trl_llms.LLM()
        import think_reason_learn.features._generator as gen_mod
        gen_mod.llm = trl_llms.llm
        print(f"  TRL OPENAI_API_KEY set: {bool(trl_settings.OPENAI_API_KEY)}")
    except Exception:
        pass
    if providers is None:
        providers = {"openai": True, "google": False}
    llm_priority = []
    if providers.get("openai", False):
        llm_priority.append(OpenAIChoice(model=model))
    if providers.get("google", False):
        llm_priority.append(GoogleChoice(model=google_model or "gemini-2.0-flash"))
    if not llm_priority:
        raise RuntimeError("No LLM providers enabled for feature generation.")

    generator = FeatureGenerator(
        schema=VCBENCH_SCHEMA,
        helpers=VCBENCH_HELPERS,
        llm_priority=llm_priority,
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
    llm_rules = [
        {
            "name": str(r.name),
            "description": str(r.description),
            "expression": str(r.expression),
        }
        for r in rules
    ]

    if all_recs is None:
        l_all = pd.DataFrame(index=range(len(train_recs) + len(test_recs)))
    else:
        l_all = evaluator.evaluate_df(all_recs)

    ok = len(rules) - len(errors)
    print(f"  Generated {len(rules)} rules ({ok} compiled OK)")
    for i, r in enumerate(rules, 1):
        def _sanitize(text: str) -> str:
            return "".join(ch if ch.isprintable() else " " for ch in text)

        name = _sanitize(str(r.name))
        desc = _sanitize(str(r.description))
        print(f"    {i}. {name}: {desc}")

    return l_all, l_train, l_test, llm_names, llm_rules


def _load_env_if_present() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from think_reason_learn.core import _config as trl_config
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip().lstrip("\ufeff")
            val = val.strip().strip('"').strip("'")
            if key and (key not in os.environ or not os.environ.get(key)):
                os.environ[key] = val
            if key in ("OPENAI_API_KEY", "GOOGLE_AI_API_KEY", "XAI_API_KEY", "ANTHROPIC_API_KEY"):
                if hasattr(trl_config, "settings") and not getattr(trl_config.settings, key, ""):
                    setattr(trl_config.settings, key, val)
    except Exception:
        return
