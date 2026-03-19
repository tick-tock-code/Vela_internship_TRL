# features/extract_structured.py
"""
Parse educations_json and jobs_json into flat feature columns.
Implements Tier 1–4 features from plan.md.
"""
import pandas as pd
import numpy as np
import json
import re


# === Encoding constants ===

COMPANY_SIZE_MAP = {
    "myself only": 1, "1": 1, "1 employee": 1,
    "1-10": 2, "1-10 employees": 2,
    "2-10": 2, "2-10 employees": 2,
    "11-50": 3, "11-50 employees": 3,
    "51-200": 4, "51-200 employees": 4,
    "201-500": 5, "201-500 employees": 5,
    "501-1000": 6, "501-1000 employees": 6,
    "1001-5000": 7, "1001-5000 employees": 7,
    "5001-10000": 8, "5001-10000 employees": 8,
    "10001+": 9, "10001+ employees": 9,
    "myself only employees": 1,
}

DURATION_MIDPOINT = {
    "<2": 1.0, "2-3": 2.5, "4-5": 4.5, "6-9": 7.5, ">9": 10.0,
}

HIGH_COMFORT_INDUSTRIES = {
    "financial services", "consulting", "investment banking",
    "government", "law", "accounting", "insurance",
    "banking", "venture capital", "private equity",
}

STEM_KEYWORDS = {
    "computer", "engineering", "math", "physics", "chemistry", "biology",
    "science", "electrical", "mechanical", "software", "data",
    "statistics", "biotech", "nanotechnology", "information technology",
    "aerospace", "chemical", "biomedical", "neuroscience", "genomics",
    "bioinformatics", "robotics", "artificial intelligence", "machine learning",
    "electronics", "telecommunications", "materials", "nuclear", "industrial",
    "civil", "environmental engineering", "applied math", "computational",
    "informatics", "cybersecurity", "operations research",
}

BUSINESS_KEYWORDS = {
    "business", "mba", "management", "marketing", "finance", "economics",
    "accounting", "entrepreneurship", "strategy", "commerce", "administration",
    "consulting", "leadership", "organizational",
}

BIOTECH_VC_KEYWORDS = {
    "biotech", "biotechnology", "venture capital", "private equity",
    "life sciences", "pharmaceutical", "research",
}


# === Helper functions ===

def _parse_json_safe(val):
    """Parse a JSON string, returning empty list on failure.
    Handles both JSON (double quotes) and Python literal (single quotes) formats.
    """
    if pd.isna(val) or val == "":
        return []
    val_str = str(val)
    # Try standard JSON first
    try:
        parsed = json.loads(val_str)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        pass
    # Fall back to Python literal eval (handles single-quoted dicts)
    try:
        import ast
        parsed = ast.literal_eval(val_str)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _normalize_company_size(raw):
    """Normalize company_size string to ordinal value."""
    if not raw or pd.isna(raw):
        return 0
    cleaned = str(raw).strip().strip("()")
    return COMPANY_SIZE_MAP.get(cleaned, 0)


def _get_duration_midpoint(raw):
    """Convert duration bucket string to numeric midpoint."""
    if not raw or pd.isna(raw):
        return 0.0
    return DURATION_MIDPOINT.get(str(raw).strip(), 0.0)


def _get_seniority(role_str):
    """Map a role string to a 0-5 seniority score using keyword matching.
    Uses word-boundary-aware matching to avoid substring false positives
    (e.g. 'cto' matching inside 'director').
    """
    if not role_str:
        return 0
    role = role_str.lower()

    def _has_word(keywords):
        """Check if any keyword appears as a whole word (or clear token) in role."""
        for kw in keywords:
            pattern = r'(?:^|[\s,/(\-])' + re.escape(kw) + r'(?:$|[\s,/)\-])'
            if re.search(pattern, role):
                return True
        return False

    # Level 0: Intern / Junior / Entry — check FIRST to avoid IC match on "Junior Developer"
    if _has_word([
        "intern", "junior", "trainee", "apprentice",
        "volunteer", "student",
    ]):
        return 0

    # Level 5: Founder / C-suite
    if _has_word([
        "founder", "co-founder", "cofounder",
        "ceo", "cto", "cfo", "coo", "cmo", "cpo", "cio", "cso",
        "chief executive", "chief technology", "chief financial",
        "chief operating", "chief marketing", "chief product",
        "chief information", "chief scientific", "chief medical",
    ]):
        return 5

    # Level 4: VP / President / Managing Director
    if _has_word([
        "vp", "vice president", "svp", "evp",
        "president", "general manager",
        "managing director", "partner", "general partner",
    ]):
        return 4

    # Level 3: Director / Head / Principal
    if _has_word([
        "director", "head of", "head,", "principal",
        "dean", "chair", "professor",
    ]):
        return 3

    # Level 2: Senior / Lead / Staff
    if _has_word([
        "senior", "lead", "staff", "sr.", "sr ",
        "architect", "fellow",
    ]):
        return 2

    # Level 1: IC / Manager / Associate / Analyst
    if _has_word([
        "manager", "engineer", "developer", "analyst",
        "associate", "consultant", "scientist", "researcher",
        "designer", "specialist", "coordinator", "advisor",
        "strategist", "editor", "producer", "planner",
    ]):
        return 1

    # Check for "assistant" last — can be junior or IC depending on context
    if "assistant" in role:
        return 0

    return 1  # default to IC-level


def _get_degree_level(degree_str):
    """Map degree string to ordinal: PhD=4, MBA/JD/MD=3, MS=2, BS/BA=1, other=0."""
    if not degree_str:
        return 0
    d = degree_str.lower().strip()

    if any(kw in d for kw in ["phd", "ph.d", "dphil", "scd", "edd", "psyd", "dba"]):
        return 4
    if any(kw in d for kw in ["postdoc", "postdoctoral"]):
        return 4
    if any(kw in d for kw in ["mba", "jd", "md", "do ", "dds", "dmd", "dvm", "pharmd"]):
        return 3
    if d in ("md", "do", "jd", "mba", "dds", "dmd", "dvm"):
        return 3
    if any(kw in d for kw in [
        "ms", "ma", "msc", "meng", "med", "mfa", "mph", "mpp", "mpa",
        "mfin", "mim", "llm", "mphil", "master", "mdes", "march",
        "mba", "executive mba",
    ]):
        return 2
    if any(kw in d for kw in [
        "bs", "ba", "bsc", "beng", "btech", "bba", "bfa", "barch",
        "bachelor", "bcom", "be ", "undergraduate",
    ]):
        return 1
    if d in ("be", "ba", "bs", "bsc", "bfa", "bba", "bcom"):
        return 1

    return 0


def _is_stem_field(field_str):
    """Check if an education field is STEM."""
    if not field_str:
        return 0
    field = field_str.lower()
    return int(any(kw in field for kw in STEM_KEYWORDS))


def _field_relevance(field_str, industry_str):
    """Score field relevance to startup industry: 1-5."""
    if not field_str:
        return 1
    field = field_str.lower()
    industry = _safe_str(industry_str).lower()

    is_stem = any(kw in field for kw in STEM_KEYWORDS)
    is_biz = any(kw in field for kw in BUSINESS_KEYWORDS)
    is_tech_industry = any(kw in industry for kw in [
        "software", "technology", "internet", "it services", "digital",
        "biotech", "nanotech",
    ])

    if is_stem and is_tech_industry:
        return 5
    if is_stem:
        return 4
    if is_biz:
        return 3
    if is_tech_industry:
        return 2
    return 1


def _qs_to_prestige_tier(qs_str):
    """Convert QS ranking string to prestige tier: top-10=4, top-50=3, top-100=2, ranked=1, null=0."""
    if not qs_str or pd.isna(qs_str) or qs_str == "200+":
        if qs_str == "200+":
            return 1
        return 0
    try:
        rank = int(qs_str)
        if rank <= 10:
            return 4
        elif rank <= 50:
            return 3
        elif rank <= 100:
            return 2
        else:
            return 1
    except (ValueError, TypeError):
        return 0


def _safe_str(val):
    """Convert a value to string safely, returning '' for NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val)


def _is_biotech_or_vc(industry_str):
    """Check if startup industry is biotech/VC/PE — credential-dense, high FP rate."""
    if not industry_str:
        return 0
    ind = industry_str.lower()
    return int(any(kw in ind for kw in BIOTECH_VC_KEYWORDS))


def _is_founding_size(size_ordinal):
    """Check if company size indicates a founding-stage company (myself only or 2-10)."""
    return size_ordinal in (1, 2)


def _is_comfort_industry(industry_str):
    """Check if a job industry is a 'high comfort' stable industry."""
    if not industry_str:
        return False
    ind = industry_str.lower()
    return any(kw in ind for kw in HIGH_COMFORT_INDUSTRIES)


# === Main extraction function ===

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract all 28 structured features from a VCBench dataframe.

    Parses the ``educations_json``, ``jobs_json``, ``ipos``, and ``acquisitions``
    columns and appends one column per feature to a copy of the input dataframe.
    The original columns are preserved; four temporary ``_*_parsed`` columns are
    dropped before returning.

    Feature tiers
    -------------
    Tier 1 — Direct exit signals (3):
        has_prior_ipo, has_prior_acquisition, exit_count
    Tier 2 — Sacrifice signal (5):
        max_company_size_before_founding, prestige_sacrifice_score,
        years_in_large_company, comfort_index, founding_timing
    Tier 3 — Education × QS interaction (6):
        edu_prestige_tier, field_relevance_score, prestige_x_relevance,
        degree_level, stem_flag, best_degree_prestige
    Tier 4 — Career trajectory (9):
        max_seniority_reached, seniority_is_monotone, company_size_is_growing,
        restlessness_score, founding_role_count, longest_founding_tenure,
        industry_pivot_count, industry_alignment, total_inferred_experience
    v2 interaction features (5):
        is_serial_founder, exit_x_serial, sacrifice_x_serial,
        industry_prestige_penalty, persistence_score

    Note: ``repeat_founding_gap`` is also computed but excluded from the
    classifier feature set due to a 73.8% null rate on the training split.

    Parameters
    ----------
    df : pd.DataFrame
        Raw VCBench dataframe. Must contain ``educations_json``, ``jobs_json``,
        ``ipos``, ``acquisitions``, and ``industry`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with 28 feature columns appended. Null rates are 0% for
        all classifier features (NaN → 0 fill applied in classifier.py).
    """
    df = df.copy()

    # Pre-parse JSON columns
    df["_edu_parsed"] = df["educations_json"].apply(_parse_json_safe)
    df["_jobs_parsed"] = df["jobs_json"].apply(_parse_json_safe)

    # --- Tier 1: Direct Exit Signals ---
    df["_ipos_parsed"] = df["ipos"].apply(_parse_json_safe)
    df["_acq_parsed"] = df["acquisitions"].apply(_parse_json_safe)
    df["has_prior_ipo"] = df["_ipos_parsed"].apply(lambda x: int(len(x) > 0))
    df["has_prior_acquisition"] = df["_acq_parsed"].apply(lambda x: int(len(x) > 0))
    df["exit_count"] = df["has_prior_ipo"] + df["has_prior_acquisition"]

    # --- Tier 2: Sacrifice Signal ---
    sacrifice_records = []
    for _, row in df.iterrows():
        jobs = row["_jobs_parsed"]
        industry = _safe_str(row.get("industry", ""))

        # Find first founding role index
        first_founding_idx = None
        for idx, job in enumerate(jobs):
            size_ord = _normalize_company_size(job.get("company_size", ""))
            seniority = _get_seniority(job.get("role", ""))
            if _is_founding_size(size_ord) or seniority == 5:
                first_founding_idx = idx
                break

        # Pre-founding jobs
        pre_founding_jobs = jobs[first_founding_idx + 1:] if first_founding_idx is not None else jobs

        max_size_before = 0
        max_seniority_before = 0
        years_in_large = 0.0
        comfort_score = 0.0
        total_pre_exp = 0.0

        for job in pre_founding_jobs:
            size_ord = _normalize_company_size(job.get("company_size", ""))
            seniority = _get_seniority(job.get("role", ""))
            dur = _get_duration_midpoint(job.get("duration", ""))

            max_size_before = max(max_size_before, size_ord)
            max_seniority_before = max(max_seniority_before, seniority)

            if size_ord >= 6:  # 501+ employees
                years_in_large += dur

            if _is_comfort_industry(job.get("industry", "")):
                comfort_score += dur * size_ord * seniority

            total_pre_exp += dur

        sacrifice_records.append({
            "max_company_size_before_founding": max_size_before,
            "prestige_sacrifice_score": max_size_before * max_seniority_before,
            "years_in_large_company": years_in_large,
            "comfort_index": comfort_score,
            "founding_timing": total_pre_exp,
        })

    sacrifice_df = pd.DataFrame(sacrifice_records, index=df.index)
    df = pd.concat([df, sacrifice_df], axis=1)

    # --- Tier 3: Education × QS Interaction ---
    edu_records = []
    for _, row in df.iterrows():
        edus = row["_edu_parsed"]
        industry = _safe_str(row.get("industry", ""))

        best_prestige = 0
        best_degree_level = 0
        best_field_relevance = 1
        any_stem = 0

        for edu in edus:
            prestige = _qs_to_prestige_tier(edu.get("qs_ranking", ""))
            degree_lev = _get_degree_level(edu.get("degree", ""))
            field_rel = _field_relevance(edu.get("field", ""), industry)
            stem = _is_stem_field(edu.get("field", ""))

            if prestige > best_prestige:
                best_prestige = prestige
            if degree_lev > best_degree_level:
                best_degree_level = degree_lev
            if field_rel > best_field_relevance:
                best_field_relevance = field_rel
            if stem:
                any_stem = 1

        edu_records.append({
            "edu_prestige_tier": best_prestige,
            "field_relevance_score": best_field_relevance,
            "prestige_x_relevance": best_prestige * best_field_relevance,
            "degree_level": best_degree_level,
            "stem_flag": any_stem,
            "best_degree_prestige": best_prestige,
        })

    edu_df = pd.DataFrame(edu_records, index=df.index)
    df = pd.concat([df, edu_df], axis=1)

    # --- Tier 4: Career Trajectory Features ---
    traj_records = []
    for _, row in df.iterrows():
        jobs = row["_jobs_parsed"]
        industry = _safe_str(row.get("industry", ""))

        seniorities = []
        sizes = []
        founding_count = 0
        founding_tenures = []
        founding_indices = []
        all_durations = []
        restless_count = 0
        job_industries = set()
        total_exp = 0.0

        for job_i, job in enumerate(jobs):
            size_ord = _normalize_company_size(job.get("company_size", ""))
            seniority = _get_seniority(job.get("role", ""))
            dur = _get_duration_midpoint(job.get("duration", ""))

            seniorities.append(seniority)
            sizes.append(size_ord)
            all_durations.append(dur)
            total_exp += dur

            if dur < 2:
                restless_count += 1

            if _is_founding_size(size_ord) or seniority == 5:
                founding_count += 1
                founding_tenures.append(dur)
                founding_indices.append(job_i)

            job_ind = job.get("industry", "")
            if job_ind:
                job_industries.add(job_ind.lower().strip())

        # Monotone seniority (non-decreasing from last to first, since jobs are newest-first)
        reversed_sen = list(reversed(seniorities))
        seniority_mono = int(all(
            reversed_sen[i] <= reversed_sen[i + 1]
            for i in range(len(reversed_sen) - 1)
        )) if len(reversed_sen) > 1 else 0

        # Growing company size
        reversed_sizes = list(reversed(sizes))
        size_growing = int(all(
            reversed_sizes[i] <= reversed_sizes[i + 1]
            for i in range(len(reversed_sizes) - 1)
        )) if len(reversed_sizes) > 1 else 0

        # Industry alignment
        startup_industry = (industry or "").lower().strip()
        industry_aligned = int(any(
            startup_industry and startup_industry in ji
            for ji in job_industries
        )) if startup_industry else 0

        # repeat_founding_gap: years between 1st and 2nd founding role (chronologically)
        # Jobs are newest-first, so chronologically 1st = largest index, 2nd = second-largest
        repeat_gap = np.nan
        if len(founding_indices) >= 2:
            idx_first = founding_indices[-1]   # chronologically 1st (oldest in list)
            idx_second = founding_indices[-2]  # chronologically 2nd
            repeat_gap = sum(all_durations[idx_second + 1 : idx_first])

        traj_records.append({
            "max_seniority_reached": max(seniorities) if seniorities else 0,
            "seniority_is_monotone": seniority_mono,
            "company_size_is_growing": size_growing,
            "restlessness_score": restless_count,
            "founding_role_count": founding_count,
            "longest_founding_tenure": max(founding_tenures) if founding_tenures else 0.0,
            "industry_pivot_count": len(job_industries),
            "industry_alignment": industry_aligned,
            "total_inferred_experience": total_exp,
            "repeat_founding_gap": repeat_gap,
        })

    traj_df = pd.DataFrame(traj_records, index=df.index)
    df = pd.concat([df, traj_df], axis=1)

    # --- v2 derived features (from Step 7 + panel synthesis) ---
    df["is_serial_founder"] = (df["founding_role_count"] >= 2).astype(int)
    df["exit_x_serial"] = df["exit_count"] * df["founding_role_count"]
    df["sacrifice_x_serial"] = df["prestige_sacrifice_score"] * df["is_serial_founder"]
    df["industry_prestige_penalty"] = df["edu_prestige_tier"] * df["industry"].apply(
        lambda x: _is_biotech_or_vc(_safe_str(x))
    )
    df["persistence_score"] = df["longest_founding_tenure"] / (df["total_inferred_experience"] + 0.01)

    # Drop temporary columns
    df.drop(columns=["_edu_parsed", "_jobs_parsed", "_ipos_parsed", "_acq_parsed"], inplace=True)

    return df


if __name__ == "__main__":
    train = pd.read_csv("data/public_train.csv")
    result = extract_features(train)

    FEATURE_COLS = [
        "has_prior_ipo", "has_prior_acquisition", "exit_count",
        "max_company_size_before_founding", "prestige_sacrifice_score",
        "years_in_large_company", "comfort_index", "founding_timing",
        "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
        "degree_level", "stem_flag", "best_degree_prestige",
        "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
        "restlessness_score", "founding_role_count", "longest_founding_tenure",
        "industry_pivot_count", "industry_alignment", "total_inferred_experience",
        # v2 features
        "is_serial_founder", "exit_x_serial", "sacrifice_x_serial",
        "industry_prestige_penalty", "persistence_score", "repeat_founding_gap",
    ]

    print(f"Total features created: {len(FEATURE_COLS)}")
    print(f"\n=== Null rates ===")
    for col in FEATURE_COLS:
        null_rate = result[col].isna().mean()
        print(f"  {col}: {null_rate:.1%}")

    # Check repeat_founding_gap null rate — drop if > 35%
    rfg_null = result["repeat_founding_gap"].isna().mean()
    print(f"\n*** repeat_founding_gap null rate: {rfg_null:.1%} ***")
    if rfg_null > 0.35:
        print("  -> ABOVE 35% THRESHOLD — should be DROPPED from feature set")
    else:
        print("  -> Below 35% threshold — KEEP")

    print(f"\n=== v2 feature distributions ===")
    for col in ["is_serial_founder", "exit_x_serial", "sacrifice_x_serial",
                "industry_prestige_penalty", "persistence_score", "repeat_founding_gap"]:
        print(f"\n{col}:")
        print(result[col].describe().round(3))
        if col in ["is_serial_founder"]:
            print(result[col].value_counts().sort_index())

    print(f"\n=== Success rate by is_serial_founder ===")
    print(result.groupby("is_serial_founder")["success"].mean().round(3))

    print(f"\n=== Success rate by exit_count ===")
    print(result.groupby("exit_count")["success"].mean().round(3))
