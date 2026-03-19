# tests/test_extract_structured.py
"""
Unit tests for features/extract_structured.py
Tests each of the 23 engineered features against hand-crafted inputs.
"""
import pytest
import pandas as pd
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.extract_structured import (
    extract_features,
    _parse_json_safe,
    _normalize_company_size,
    _get_duration_midpoint,
    _get_seniority,
    _get_degree_level,
    _is_stem_field,
    _field_relevance,
    _qs_to_prestige_tier,
    _safe_str,
)


# ============================================================
# Helper to build a single-row DataFrame for feature extraction
# ============================================================

def _make_row(
    educations_json="[]",
    jobs_json="[]",
    ipos=None,
    acquisitions=None,
    industry=None,
    success=0,
    founder_uuid="test-uuid",
):
    return pd.DataFrame([{
        "founder_uuid": founder_uuid,
        "success": success,
        "industry": industry,
        "ipos": ipos,
        "acquisitions": acquisitions,
        "educations_json": educations_json,
        "jobs_json": jobs_json,
        "anonymised_prose": "test prose",
    }])


def _extract_one(**kwargs):
    """Build a single-row DF, extract features, return the row as a dict."""
    df = _make_row(**kwargs)
    result = extract_features(df)
    return result.iloc[0].to_dict()


# ============================================================
# Tests for helper functions
# ============================================================

class TestParseJsonSafe:
    def test_valid_json(self):
        assert _parse_json_safe('[{"a": 1}]') == [{"a": 1}]

    def test_empty_list(self):
        assert _parse_json_safe("[]") == []

    def test_nan(self):
        assert _parse_json_safe(float("nan")) == []

    def test_none_string(self):
        assert _parse_json_safe(None) == []

    def test_empty_string(self):
        assert _parse_json_safe("") == []

    def test_python_single_quotes(self):
        """ipos/acquisitions use Python dict syntax with single quotes."""
        val = "[{'price_usd': 'Undisclosed', 'acquired_by_well_known': False}]"
        result = _parse_json_safe(val)
        assert len(result) == 1
        assert result[0]["price_usd"] == "Undisclosed"

    def test_multiple_python_dicts(self):
        val = "[{'a': 1}, {'a': 2}]"
        assert len(_parse_json_safe(val)) == 2


class TestNormalizeCompanySize:
    def test_plain(self):
        assert _normalize_company_size("myself only") == 1
        assert _normalize_company_size("2-10") == 2
        assert _normalize_company_size("11-50") == 3
        assert _normalize_company_size("51-200") == 4
        assert _normalize_company_size("201-500") == 5
        assert _normalize_company_size("501-1000") == 6
        assert _normalize_company_size("1001-5000") == 7
        assert _normalize_company_size("5001-10000") == 8
        assert _normalize_company_size("10001+") == 9

    def test_with_employees_suffix(self):
        assert _normalize_company_size("10001+ employees") == 9
        assert _normalize_company_size("51-200 employees") == 4
        assert _normalize_company_size("myself only employees") == 1

    def test_with_parentheses(self):
        assert _normalize_company_size("(10001+ employees)") == 9
        assert _normalize_company_size("(2-10 employees)") == 2
        assert _normalize_company_size("(myself only employees)") == 1

    def test_empty_and_nan(self):
        assert _normalize_company_size("") == 0
        assert _normalize_company_size(None) == 0
        assert _normalize_company_size(float("nan")) == 0

    def test_single_employee(self):
        assert _normalize_company_size("1") == 1
        assert _normalize_company_size("1 employee") == 1


class TestGetDurationMidpoint:
    def test_all_buckets(self):
        assert _get_duration_midpoint("<2") == 1.0
        assert _get_duration_midpoint("2-3") == 2.5
        assert _get_duration_midpoint("4-5") == 4.5
        assert _get_duration_midpoint("6-9") == 7.5
        assert _get_duration_midpoint(">9") == 10.0

    def test_empty(self):
        assert _get_duration_midpoint("") == 0.0
        assert _get_duration_midpoint(None) == 0.0


class TestGetSeniority:
    def test_founder_level(self):
        assert _get_seniority("Founder, CEO") == 5
        assert _get_seniority("Co-Founder") == 5
        assert _get_seniority("CTO") == 5
        assert _get_seniority("Chief Technology Officer") == 5

    def test_vp_level(self):
        assert _get_seniority("VP of Engineering") == 4
        assert _get_seniority("Senior Vice President") == 4
        assert _get_seniority("President") == 4
        assert _get_seniority("Managing Director") == 4
        assert _get_seniority("General Partner") == 4

    def test_director_level(self):
        assert _get_seniority("Director of Product") == 3
        assert _get_seniority("Head of Engineering") == 3
        assert _get_seniority("Principal Engineer") == 3
        assert _get_seniority("Professor") == 3

    def test_senior_level(self):
        assert _get_seniority("Senior Software Engineer") == 2
        assert _get_seniority("Lead Developer") == 2
        assert _get_seniority("Staff Engineer") == 2
        assert _get_seniority("Solutions Architect") == 2

    def test_ic_level(self):
        assert _get_seniority("Software Engineer") == 1
        assert _get_seniority("Product Manager") == 1
        assert _get_seniority("Data Analyst") == 1
        assert _get_seniority("Consultant") == 1

    def test_junior_level(self):
        assert _get_seniority("Intern") == 0
        assert _get_seniority("Junior Developer") == 0
        assert _get_seniority("Research Assistant") == 0

    def test_empty(self):
        assert _get_seniority("") == 0
        assert _get_seniority(None) == 0


class TestGetDegreeLevel:
    def test_phd(self):
        assert _get_degree_level("PhD") == 4
        assert _get_degree_level("Postdoctoral") == 4
        assert _get_degree_level("ScD") == 4

    def test_professional(self):
        assert _get_degree_level("MBA") == 3
        assert _get_degree_level("JD") == 3
        assert _get_degree_level("MD") == 3

    def test_masters(self):
        assert _get_degree_level("MS") == 2
        assert _get_degree_level("MA") == 2
        assert _get_degree_level("MSc") == 2
        assert _get_degree_level("MEng") == 2
        assert _get_degree_level("Master") == 2

    def test_bachelors(self):
        assert _get_degree_level("BS") == 1
        assert _get_degree_level("BA") == 1
        assert _get_degree_level("BSc") == 1
        assert _get_degree_level("Bachelor") == 1

    def test_other(self):
        assert _get_degree_level("Certificate") == 0
        assert _get_degree_level("") == 0
        assert _get_degree_level(None) == 0


class TestIsStemField:
    def test_stem(self):
        assert _is_stem_field("Computer Science") == 1
        assert _is_stem_field("Electrical Engineering") == 1
        assert _is_stem_field("Mathematics") == 1
        assert _is_stem_field("Physics") == 1
        assert _is_stem_field("Data Science") == 1

    def test_non_stem(self):
        assert _is_stem_field("History") == 0
        assert _is_stem_field("English Literature") == 0
        assert _is_stem_field("Philosophy") == 0

    def test_empty(self):
        assert _is_stem_field("") == 0
        assert _is_stem_field(None) == 0


class TestQsToPrestigeTier:
    def test_top_10(self):
        assert _qs_to_prestige_tier("1") == 4
        assert _qs_to_prestige_tier("10") == 4

    def test_top_50(self):
        assert _qs_to_prestige_tier("11") == 3
        assert _qs_to_prestige_tier("50") == 3

    def test_top_100(self):
        assert _qs_to_prestige_tier("51") == 2
        assert _qs_to_prestige_tier("100") == 2

    def test_ranked(self):
        assert _qs_to_prestige_tier("101") == 1
        assert _qs_to_prestige_tier("200+") == 1

    def test_null(self):
        assert _qs_to_prestige_tier(None) == 0
        assert _qs_to_prestige_tier("") == 0
        assert _qs_to_prestige_tier(float("nan")) == 0


class TestFieldRelevance:
    def test_stem_in_tech(self):
        assert _field_relevance("Computer Science", "Software Development") == 5

    def test_stem_non_tech(self):
        assert _field_relevance("Computer Science", "Food & Beverage") == 4

    def test_business(self):
        assert _field_relevance("Business Administration", "Financial Services") == 3

    def test_non_stem_in_tech(self):
        assert _field_relevance("History", "Software Development") == 2

    def test_non_stem_non_tech(self):
        assert _field_relevance("History", "Food & Beverage") == 1

    def test_nan_industry(self):
        assert _field_relevance("Computer Science", float("nan")) == 4


# ============================================================
# Tier 1: Direct Exit Signals
# ============================================================

class TestTier1ExitSignals:
    def test_no_exits(self):
        row = _extract_one()
        assert row["has_prior_ipo"] == 0
        assert row["has_prior_acquisition"] == 0
        assert row["exit_count"] == 0

    def test_has_ipo(self):
        row = _extract_one(
            ipos="[{'amount_raised_usd': '50M - 150M', 'valuation_usd': '>500M'}]"
        )
        assert row["has_prior_ipo"] == 1
        assert row["exit_count"] >= 1

    def test_has_acquisition(self):
        row = _extract_one(
            acquisitions="[{'price_usd': 'Undisclosed', 'acquired_by_well_known': False}]"
        )
        assert row["has_prior_acquisition"] == 1
        assert row["exit_count"] >= 1

    def test_both_exits(self):
        row = _extract_one(
            ipos="[{'amount_raised_usd': '50M - 150M', 'valuation_usd': '>500M'}]",
            acquisitions="[{'price_usd': '>500M', 'acquired_by_well_known': False}]",
        )
        assert row["has_prior_ipo"] == 1
        assert row["has_prior_acquisition"] == 1
        assert row["exit_count"] == 2

    def test_empty_list_ipos(self):
        row = _extract_one(ipos="[]")
        assert row["has_prior_ipo"] == 0

    def test_nan_ipos(self):
        row = _extract_one(ipos=None)
        assert row["has_prior_ipo"] == 0


# ============================================================
# Tier 2: Sacrifice Signal
# ============================================================

class TestTier2SacrificeSignal:
    def test_no_jobs(self):
        row = _extract_one(jobs_json="[]")
        assert row["max_company_size_before_founding"] == 0
        assert row["prestige_sacrifice_score"] == 0
        assert row["years_in_large_company"] == 0.0
        assert row["comfort_index"] == 0.0
        assert row["founding_timing"] == 0.0

    def test_vp_at_large_company_then_founder(self):
        """VP at 10001+ company for 6-9 years, then founded solo startup."""
        jobs = json.dumps([
            {"role": "Founder, CEO", "company_size": "myself only", "industry": "", "duration": "2-3"},
            {"role": "VP of Engineering", "company_size": "10001+ employees", "industry": "Software Development", "duration": "6-9"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["max_company_size_before_founding"] == 9  # 10001+ = 9
        assert row["prestige_sacrifice_score"] == 9 * 4     # size(9) * seniority(VP=4)
        assert row["years_in_large_company"] == 7.5          # 6-9 → 7.5
        assert row["founding_timing"] == 7.5                 # total pre-founding experience

    def test_engineer_at_small_company(self):
        """Engineer at 11-50 company, no founding role → all jobs are 'pre-founding'."""
        jobs = json.dumps([
            {"role": "Software Engineer", "company_size": "11-50 employees", "industry": "Technology", "duration": "2-3"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["max_company_size_before_founding"] == 3  # 11-50 = 3
        assert row["prestige_sacrifice_score"] == 3 * 1      # size(3) * seniority(IC=1)

    def test_comfort_index_financial_services(self):
        """Director at large financial services company should have high comfort_index."""
        jobs = json.dumps([
            {"role": "Founder", "company_size": "myself only", "industry": "", "duration": "<2"},
            {"role": "Director", "company_size": "5001-10000 employees", "industry": "Financial Services", "duration": "4-5"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["comfort_index"] > 0  # duration(4.5) * size(8) * seniority(3)
        assert row["comfort_index"] == 4.5 * 8 * 3

    def test_founding_timing_multiple_pre_jobs(self):
        """Total pre-founding experience across multiple jobs."""
        jobs = json.dumps([
            {"role": "CEO", "company_size": "2-10", "industry": "", "duration": "4-5"},
            {"role": "Manager", "company_size": "201-500", "industry": "Technology", "duration": "2-3"},
            {"role": "Analyst", "company_size": "1001-5000", "industry": "Financial Services", "duration": "<2"},
        ])
        row = _extract_one(jobs_json=jobs)
        # First founding role is index 0 (CEO at 2-10, seniority=5)
        # Pre-founding jobs: index 1 and 2
        assert row["founding_timing"] == 2.5 + 1.0  # 2-3 + <2


# ============================================================
# Tier 3: Education × QS Interaction
# ============================================================

class TestTier3EducationQS:
    def test_no_education(self):
        row = _extract_one(educations_json="[]")
        assert row["edu_prestige_tier"] == 0
        assert row["field_relevance_score"] == 1
        assert row["prestige_x_relevance"] == 0
        assert row["degree_level"] == 0
        assert row["stem_flag"] == 0
        assert row["best_degree_prestige"] == 0

    def test_null_education(self):
        row = _extract_one(educations_json=None)
        assert row["edu_prestige_tier"] == 0
        assert row["stem_flag"] == 0

    def test_top10_cs_degree(self):
        edu = json.dumps([
            {"degree": "BS", "field": "Computer Science", "qs_ranking": "3"},
        ])
        row = _extract_one(educations_json=edu, industry="Software Development")
        assert row["edu_prestige_tier"] == 4       # top-10
        assert row["stem_flag"] == 1
        assert row["field_relevance_score"] == 5   # STEM + tech industry
        assert row["prestige_x_relevance"] == 20   # 4 * 5
        assert row["degree_level"] == 1            # BS

    def test_mba_at_ranked_school(self):
        edu = json.dumps([
            {"degree": "MBA", "field": "Entrepreneurship", "qs_ranking": "6"},
        ])
        row = _extract_one(educations_json=edu)
        assert row["edu_prestige_tier"] == 4     # top-10
        assert row["degree_level"] == 3          # MBA
        assert row["stem_flag"] == 0

    def test_multiple_degrees_takes_best(self):
        """Should take the best prestige tier across multiple degrees."""
        edu = json.dumps([
            {"degree": "BA", "field": "History", "qs_ranking": "200+"},
            {"degree": "PhD", "field": "Computer Science", "qs_ranking": "8"},
        ])
        row = _extract_one(educations_json=edu, industry="Software Development")
        assert row["edu_prestige_tier"] == 4     # top-10 from PhD
        assert row["best_degree_prestige"] == 4
        assert row["degree_level"] == 4          # PhD
        assert row["stem_flag"] == 1             # CS from PhD
        assert row["field_relevance_score"] == 5 # STEM + tech

    def test_qs_200plus_is_tier_1(self):
        edu = json.dumps([
            {"degree": "BS", "field": "Art", "qs_ranking": "200+"},
        ])
        row = _extract_one(educations_json=edu)
        assert row["edu_prestige_tier"] == 1  # ranked but >200

    def test_qs_boundary_50(self):
        edu = json.dumps([
            {"degree": "BS", "field": "Math", "qs_ranking": "50"},
        ])
        row = _extract_one(educations_json=edu)
        assert row["edu_prestige_tier"] == 3  # top-50

    def test_qs_boundary_100(self):
        edu = json.dumps([
            {"degree": "BS", "field": "Math", "qs_ranking": "100"},
        ])
        row = _extract_one(educations_json=edu)
        assert row["edu_prestige_tier"] == 2  # top-100

    def test_qs_boundary_101(self):
        edu = json.dumps([
            {"degree": "BS", "field": "Math", "qs_ranking": "101"},
        ])
        row = _extract_one(educations_json=edu)
        assert row["edu_prestige_tier"] == 1  # ranked >100


# ============================================================
# Tier 4: Career Trajectory Features
# ============================================================

class TestTier4CareerTrajectory:
    def test_no_jobs(self):
        row = _extract_one(jobs_json="[]")
        assert row["max_seniority_reached"] == 0
        assert row["seniority_is_monotone"] == 0
        assert row["company_size_is_growing"] == 0
        assert row["restlessness_score"] == 0
        assert row["founding_role_count"] == 0
        assert row["longest_founding_tenure"] == 0.0
        assert row["industry_pivot_count"] == 0
        assert row["industry_alignment"] == 0
        assert row["total_inferred_experience"] == 0.0

    def test_max_seniority_reached(self):
        jobs = json.dumps([
            {"role": "CEO", "company_size": "", "industry": "", "duration": "2-3"},
            {"role": "Engineer", "company_size": "", "industry": "", "duration": "2-3"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["max_seniority_reached"] == 5  # CEO = founder level

    def test_seniority_monotone_true(self):
        """Jobs listed newest-first. Reversed: Intern → Engineer → VP → monotone."""
        jobs = json.dumps([
            {"role": "VP of Engineering", "company_size": "", "industry": "", "duration": "2-3"},
            {"role": "Software Engineer", "company_size": "", "industry": "", "duration": "2-3"},
            {"role": "Intern", "company_size": "", "industry": "", "duration": "<2"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["seniority_is_monotone"] == 1

    def test_seniority_monotone_false(self):
        """VP → Engineer → Intern (newest first). Reversed: Intern → Engineer → VP.
        Wait, that IS monotone. Let me make it non-monotone."""
        jobs = json.dumps([
            {"role": "Software Engineer", "company_size": "", "industry": "", "duration": "2-3"},
            {"role": "VP of Product", "company_size": "", "industry": "", "duration": "2-3"},
            {"role": "Intern", "company_size": "", "industry": "", "duration": "<2"},
        ])
        row = _extract_one(jobs_json=jobs)
        # Reversed: Intern(0) → VP(4) → Engineer(1) → NOT monotone
        assert row["seniority_is_monotone"] == 0

    def test_company_size_growing_true(self):
        """Newest first: 10001+ → 51-200 → 2-10. Reversed: 2-10 → 51-200 → 10001+."""
        jobs = json.dumps([
            {"role": "Manager", "company_size": "10001+", "industry": "", "duration": "2-3"},
            {"role": "Engineer", "company_size": "51-200", "industry": "", "duration": "2-3"},
            {"role": "Intern", "company_size": "2-10", "industry": "", "duration": "<2"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["company_size_is_growing"] == 1

    def test_company_size_growing_false(self):
        """Newest first: 2-10 → 10001+. Reversed: 10001+ → 2-10 → shrinking."""
        jobs = json.dumps([
            {"role": "Founder", "company_size": "2-10", "industry": "", "duration": "2-3"},
            {"role": "Manager", "company_size": "10001+", "industry": "", "duration": "2-3"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["company_size_is_growing"] == 0

    def test_restlessness_score(self):
        """Count of roles with duration < 2 years."""
        jobs = json.dumps([
            {"role": "CTO", "company_size": "", "industry": "", "duration": "<2"},
            {"role": "Engineer", "company_size": "", "industry": "", "duration": "4-5"},
            {"role": "Intern", "company_size": "", "industry": "", "duration": "<2"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["restlessness_score"] == 2  # two <2 duration roles

    def test_founding_role_count(self):
        """Count founding-stage roles (myself only, 2-10, or seniority=5)."""
        jobs = json.dumps([
            {"role": "Founder", "company_size": "myself only", "industry": "", "duration": "4-5"},
            {"role": "Co-Founder", "company_size": "2-10", "industry": "", "duration": "2-3"},
            {"role": "Engineer", "company_size": "10001+", "industry": "", "duration": "2-3"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["founding_role_count"] == 2

    def test_longest_founding_tenure(self):
        """Max duration in a founding-stage company."""
        jobs = json.dumps([
            {"role": "Founder", "company_size": "myself only", "industry": "", "duration": "2-3"},
            {"role": "Co-Founder", "company_size": "2-10", "industry": "", "duration": "6-9"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["longest_founding_tenure"] == 7.5  # 6-9 → 7.5

    def test_industry_pivot_count(self):
        """Count distinct industries across all jobs."""
        jobs = json.dumps([
            {"role": "Engineer", "company_size": "", "industry": "Software Development", "duration": "2-3"},
            {"role": "Analyst", "company_size": "", "industry": "Financial Services", "duration": "2-3"},
            {"role": "Manager", "company_size": "", "industry": "Software Development", "duration": "2-3"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["industry_pivot_count"] == 2  # Software Dev + Financial Services

    def test_industry_alignment_true(self):
        """Prior job industry matches current startup industry."""
        jobs = json.dumps([
            {"role": "Engineer", "company_size": "", "industry": "Software Development", "duration": "2-3"},
        ])
        row = _extract_one(jobs_json=jobs, industry="Software Development")
        assert row["industry_alignment"] == 1

    def test_industry_alignment_false(self):
        """Prior job industry does NOT match current startup industry."""
        jobs = json.dumps([
            {"role": "Engineer", "company_size": "", "industry": "Healthcare", "duration": "2-3"},
        ])
        row = _extract_one(jobs_json=jobs, industry="Software Development")
        assert row["industry_alignment"] == 0

    def test_industry_alignment_nan_industry(self):
        """NaN startup industry should not crash and should return 0."""
        jobs = json.dumps([
            {"role": "Engineer", "company_size": "", "industry": "Software Development", "duration": "2-3"},
        ])
        row = _extract_one(jobs_json=jobs, industry=float("nan"))
        assert row["industry_alignment"] == 0

    def test_total_inferred_experience(self):
        """Sum of all duration midpoints."""
        jobs = json.dumps([
            {"role": "CTO", "company_size": "", "industry": "", "duration": "4-5"},
            {"role": "Engineer", "company_size": "", "industry": "", "duration": "2-3"},
            {"role": "Intern", "company_size": "", "industry": "", "duration": "<2"},
        ])
        row = _extract_one(jobs_json=jobs)
        assert row["total_inferred_experience"] == 4.5 + 2.5 + 1.0


# ============================================================
# Integration tests on real data
# ============================================================

class TestIntegrationRealData:
    @pytest.fixture(scope="class")
    def train_features(self):
        train = pd.read_csv("data/public_train.csv")
        return extract_features(train)

    def test_all_23_features_present(self, train_features):
        expected = [
            "has_prior_ipo", "has_prior_acquisition", "exit_count",
            "max_company_size_before_founding", "prestige_sacrifice_score",
            "years_in_large_company", "comfort_index", "founding_timing",
            "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
            "degree_level", "stem_flag", "best_degree_prestige",
            "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
            "restlessness_score", "founding_role_count", "longest_founding_tenure",
            "industry_pivot_count", "industry_alignment", "total_inferred_experience",
        ]
        for col in expected:
            assert col in train_features.columns, f"Missing feature: {col}"

    def test_no_nulls(self, train_features):
        feature_cols = [
            "has_prior_ipo", "has_prior_acquisition", "exit_count",
            "max_company_size_before_founding", "prestige_sacrifice_score",
            "years_in_large_company", "comfort_index", "founding_timing",
            "edu_prestige_tier", "field_relevance_score", "prestige_x_relevance",
            "degree_level", "stem_flag", "best_degree_prestige",
            "max_seniority_reached", "seniority_is_monotone", "company_size_is_growing",
            "restlessness_score", "founding_role_count", "longest_founding_tenure",
            "industry_pivot_count", "industry_alignment", "total_inferred_experience",
        ]
        for col in feature_cols:
            assert train_features[col].isna().sum() == 0, f"Null values in {col}"

    def test_row_count_preserved(self, train_features):
        assert len(train_features) == 3600

    def test_binary_features_are_binary(self, train_features):
        binary_cols = [
            "has_prior_ipo", "has_prior_acquisition",
            "stem_flag", "seniority_is_monotone",
            "company_size_is_growing", "industry_alignment",
        ]
        for col in binary_cols:
            vals = set(train_features[col].unique())
            assert vals.issubset({0, 1}), f"{col} has non-binary values: {vals}"

    def test_exit_count_bounds(self, train_features):
        assert train_features["exit_count"].min() >= 0
        assert train_features["exit_count"].max() <= 10

    def test_exit_count_is_sum(self, train_features):
        computed = train_features["has_prior_ipo"] + train_features["has_prior_acquisition"]
        pd.testing.assert_series_equal(
            train_features["exit_count"], computed, check_names=False
        )

    def test_prestige_tier_range(self, train_features):
        assert train_features["edu_prestige_tier"].min() >= 0
        assert train_features["edu_prestige_tier"].max() <= 4

    def test_seniority_range(self, train_features):
        assert train_features["max_seniority_reached"].min() >= 0
        assert train_features["max_seniority_reached"].max() <= 5

    def test_degree_level_range(self, train_features):
        assert train_features["degree_level"].min() >= 0
        assert train_features["degree_level"].max() <= 4

    def test_field_relevance_range(self, train_features):
        assert train_features["field_relevance_score"].min() >= 1
        assert train_features["field_relevance_score"].max() <= 5

    def test_company_size_range(self, train_features):
        assert train_features["max_company_size_before_founding"].min() >= 0
        assert train_features["max_company_size_before_founding"].max() <= 9

    def test_non_negative_features(self, train_features):
        non_neg = [
            "years_in_large_company", "comfort_index", "founding_timing",
            "restlessness_score", "founding_role_count",
            "longest_founding_tenure", "industry_pivot_count",
            "total_inferred_experience", "prestige_sacrifice_score",
        ]
        for col in non_neg:
            assert train_features[col].min() >= 0, f"{col} has negative values"

    def test_prestige_x_relevance_is_product(self, train_features):
        computed = train_features["edu_prestige_tier"] * train_features["field_relevance_score"]
        pd.testing.assert_series_equal(
            train_features["prestige_x_relevance"], computed, check_names=False
        )

    def test_best_degree_prestige_equals_edu_prestige_tier(self, train_features):
        """In current implementation, best_degree_prestige == edu_prestige_tier."""
        pd.testing.assert_series_equal(
            train_features["best_degree_prestige"],
            train_features["edu_prestige_tier"],
            check_names=False,
        )

    def test_success_rate_higher_with_exits(self, train_features):
        """Founders with exits should have higher success rate than those without."""
        rate_with = train_features[train_features["exit_count"] > 0]["success"].mean()
        rate_without = train_features[train_features["exit_count"] == 0]["success"].mean()
        assert rate_with > rate_without

    def test_success_rate_higher_top10_edu(self, train_features):
        """Top-10 QS education should have higher success rate than unranked."""
        rate_top10 = train_features[train_features["edu_prestige_tier"] == 4]["success"].mean()
        rate_unranked = train_features[train_features["edu_prestige_tier"] <= 1]["success"].mean()
        assert rate_top10 > rate_unranked
