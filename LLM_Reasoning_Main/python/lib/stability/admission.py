from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from lib.stability.diagnostics import evaluate_logistic_cv
from lib.stability.family_registry import AlignedFamilyData


def run_sequential_admission(
    data: AlignedFamilyData,
    fold_ids: np.ndarray,
    *,
    family_order: list[str],
    min_delta_f0_5: float,
    max_delta_f0_5_std: float,
    min_delta_precision_at_10: float,
) -> dict[str, Any]:
    admitted: list[str] = []
    decisions: list[dict[str, Any]] = []
    current_df = data.baseline.copy()
    current_eval = evaluate_logistic_cv(current_df, data.labels, fold_ids)["summary"]

    for family_id in family_order:
        family_df = data.candidate_frames[family_id]
        candidate_df = pd.concat([current_df, family_df], axis=1)
        candidate_eval = evaluate_logistic_cv(candidate_df, data.labels, fold_ids)["summary"]
        delta_f0_5 = float(candidate_eval["f0_5_mean"] - current_eval["f0_5_mean"])
        delta_precision = float(
            candidate_eval["precision_at_10_mean"] - current_eval["precision_at_10_mean"]
        )
        decision = {
            "family_id": family_id,
            "delta_f0_5_mean": delta_f0_5,
            "delta_f0_5_std": float(candidate_eval["f0_5_std"]),
            "delta_precision_at_10_mean": delta_precision,
            "accepted": False,
        }
        if (
            delta_f0_5 >= min_delta_f0_5
            and float(candidate_eval["f0_5_std"]) <= max_delta_f0_5_std
            and delta_precision >= min_delta_precision_at_10
        ):
            admitted.append(family_id)
            current_df = candidate_df
            current_eval = candidate_eval
            decision["accepted"] = True
        decisions.append(decision)

    return {
        "admitted_families": admitted,
        "decisions": decisions,
        "final_metrics": current_eval,
        "baseline_metrics": evaluate_logistic_cv(data.baseline, data.labels, fold_ids)["summary"],
    }
