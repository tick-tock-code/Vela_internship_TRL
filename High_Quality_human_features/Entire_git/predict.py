# predict.py
"""Generate submission on private test set. Run only once."""
import pandas as pd
import numpy as np
import joblib
from features.extract_structured import extract_features
from features.high_precision_rules import apply_rules

FINAL_THRESHOLD = 0.738

FEATURE_COLS = [
    "has_prior_ipo", "has_prior_acquisition", "exit_count",
    "max_company_size_before_founding", "prestige_sacrifice_score",
    "years_in_large_company", "comfort_index", "founding_timing",
    "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
    "degree_level", "stem_flag", "best_degree_prestige",
    "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
    "restlessness_score", "founding_role_count", "longest_founding_tenure",
    "industry_pivot_count", "industry_alignment", "total_inferred_experience",
    "is_serial_founder", "exit_x_serial", "sacrifice_x_serial",
    "industry_prestige_penalty", "persistence_score",
]


PRIVATE_DATA_PATH = "data/vcbench_final_private.csv"
OUTPUT_PATH = "submissions/submission_v1.csv"


def generate_submission():
    """Load the trained model and generate binary predictions on the test set.

    Applies the full inference pipeline:
    1. Extract structured features from the test CSV.
    2. Score rows with the trained XGBoost model (loaded from model.pkl).
    3. Override probabilities to 1.0 where the rule layer fires (exit_count > 0).
    4. Apply FINAL_THRESHOLD (0.738) to produce binary predictions.
    5. Validate that the positive rate is in the expected 4–15% range.
    6. Save ``founder_uuid, success`` to ``submissions/submission_v1.csv``.

    Returns
    -------
    str or None
        Output file path if saved successfully; None if the sanity check fails.
    """
    test = extract_features(pd.read_csv(PRIVATE_DATA_PATH))
    model = joblib.load("model.pkl")

    X_test = test[FEATURE_COLS].fillna(0)
    probs = model.predict_proba(X_test)[:, 1].tolist()

    # Apply rule layer
    rule_stats = {}
    for i, (_, row) in enumerate(test.iterrows()):
        rule_pred, rule_name = apply_rules(row)
        if rule_pred == 1:
            probs[i] = 1.0
            if rule_name not in rule_stats:
                rule_stats[rule_name] = 0
            rule_stats[rule_name] += 1

    print(f"Rule layer overrides: {rule_stats}")

    preds = (np.array(probs) >= FINAL_THRESHOLD).astype(int)

    total = len(preds)
    pos = int(preds.sum())
    rate = pos / total

    print(f"\n=== Pre-save check ===")
    print(f"Total rows: {total}")
    print(f"Predicted positive: {pos} / {total} ({rate:.1%})")

    if rate < 0.04 or rate > 0.15:
        print(f"WARNING: Positive rate {rate:.1%} is OUTSIDE expected 5-15% range.")
        print("STOPPING. Review before saving.")
        return None

    print("Positive rate within expected range. Saving.")

    submission = pd.DataFrame({
        "founder_uuid": test["founder_uuid"],
        "success": preds,
    })
    submission.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate_submission()
