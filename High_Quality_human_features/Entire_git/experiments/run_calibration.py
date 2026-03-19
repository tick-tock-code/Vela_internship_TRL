# run_calibration.py
"""Step 6b: Platt scaling on existing XGBoost model output (Karpathy priority #1)."""
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from features.extract_structured import extract_features
from features.high_precision_rules import apply_rules
from evaluate import sweep_thresholds, best_threshold

FEATURE_COLS = [
    "has_prior_ipo", "has_prior_acquisition", "exit_count",
    "max_company_size_before_founding", "prestige_sacrifice_score",
    "years_in_large_company", "comfort_index", "founding_timing",
    "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
    "degree_level", "stem_flag", "best_degree_prestige",
    "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
    "restlessness_score", "founding_role_count", "longest_founding_tenure",
    "industry_pivot_count", "industry_alignment", "total_inferred_experience",
]


def calibrate():
    val = extract_features(pd.read_csv("data/public_val.csv"))
    model = joblib.load("model.pkl")

    X_val = val[FEATURE_COLS].fillna(0)
    y_val = val["success"]
    raw_probs = model.predict_proba(X_val)[:, 1].copy()

    # Apply rule layer
    for i, (_, row) in enumerate(val.iterrows()):
        rule_pred, _ = apply_rules(row)
        if rule_pred == 1:
            raw_probs[i] = 1.0

    # Platt scaling
    platt = LogisticRegression()
    platt.fit(raw_probs.reshape(-1, 1), y_val)
    calibrated_probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    print("=== Raw probabilities ===")
    raw_result = best_threshold(y_val.tolist(), raw_probs.tolist())
    print(raw_result)

    print("\n=== Calibrated probabilities ===")
    cal_result = best_threshold(y_val.tolist(), calibrated_probs.tolist())
    print(cal_result)

    delta = cal_result['f05'] - raw_result['f05']
    print(f"\nDelta F0.5 from calibration: {delta:+.4f}")

    # Show probability distribution comparison
    print(f"\n=== Raw prob distribution ===")
    print(f"  min={raw_probs.min():.3f} p25={np.percentile(raw_probs, 25):.3f} "
          f"median={np.median(raw_probs):.3f} p75={np.percentile(raw_probs, 75):.3f} "
          f"max={raw_probs.max():.3f}")

    print(f"\n=== Calibrated prob distribution ===")
    print(f"  min={calibrated_probs.min():.3f} p25={np.percentile(calibrated_probs, 25):.3f} "
          f"median={np.median(calibrated_probs):.3f} p75={np.percentile(calibrated_probs, 75):.3f} "
          f"max={calibrated_probs.max():.3f}")

    joblib.dump(platt, "platt_scaler.pkl")
    print("\nSaved platt_scaler.pkl")


if __name__ == "__main__":
    calibrate()
