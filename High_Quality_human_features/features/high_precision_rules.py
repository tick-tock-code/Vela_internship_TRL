# features/high_precision_rules.py
def apply_rules(row) -> tuple[int | None, str]:
    """
    Returns (prediction, rule_name) if a high-confidence rule fires.
    Returns (None, None) if no rule fires — falls through to classifier.

    Calibration log:
    - 2026-03-08: Rule 2 disabled. FPs have higher edu_prestige than TPs.
    - 2026-03-08 (v2): Rule 3 (clevel_serial_founder) disabled — precision 11.1% on train.
    - prior_exit: precision 24.5% on train (2.7x base rate). KEEP — strongest signal.
    """
    # Rule 1: Prior exit — strongest TP signal (exit_count: TP avg 0.58 vs FP avg 0.11)
    if row.get('exit_count', 0) > 0:
        return 1, "prior_exit"

    # Rule 2: DISABLED — education prestige is a FP amplifier
    # if (row.get('edu_prestige_tier', 0) >= 4 and row.get('stem_flag', 0) == 1
    #         and row.get('founding_role_count', 0) >= 2):
    #     return 1, "top10_stem_serial_founder"

    # Rule 3: DISABLED — precision 11.1% on train (barely above 9% base rate)
    # if (row.get('max_seniority_reached', 0) >= 4 and
    #     row.get('max_company_size_before_founding', 0) >= 6 and
    #     row.get('founding_role_count', 0) >= 2):
    #     return 1, "clevel_serial_founder"

    return None, None
