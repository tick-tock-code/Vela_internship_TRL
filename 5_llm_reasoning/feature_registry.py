"""Custom feature registry for VCBench experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Any

from think_reason_learn.datasets import parse_company_size, parse_duration, parse_qs


FeatureType = Literal["binary", "continuous"]


@dataclass(frozen=True)
class FeatureSpec:
    """A feature spec with name, callable, and type."""

    name: str
    func: Callable[[dict[str, Any]], float]
    feature_type: FeatureType


def _qs_in_range(record: dict[str, Any], upper: int) -> float:
    educations = record.get("educations", [])
    return float(
        any(parse_qs(e.get("qs_ranking", "")) <= upper for e in educations)
    )


def _prior_ipos(record: dict[str, Any]) -> float:
    return float(len(record.get("ipos", [])) >= 1)


def _prior_acquisitions(record: dict[str, Any]) -> float:
    return float(len(record.get("acquisitions", [])) >= 1)


def _large_company_years(record: dict[str, Any]) -> float:
    jobs = record.get("jobs", [])
    return float(
        sum(
            parse_duration(j.get("duration", ""))
            for j in jobs
            if parse_company_size(j.get("company_size", "")) >= 1000
        )
    )

def _experience_duration(record: dict[str, Any]) -> float:
    jobs = record.get("jobs", [])
    return float(sum(parse_duration(j.get("duration", "")) for j in jobs))


_TECH_ROLE_KEYWORDS = {"engineer", "developer", "cto", "technical"}


def _technical_experience_duration(record: dict[str, Any]) -> float:
    jobs = record.get("jobs", [])
    return float(
        sum(
            parse_duration(j.get("duration", ""))
            for j in jobs
            if any(kw in j.get("role", "").lower() for kw in _TECH_ROLE_KEYWORDS)
        )
    )


def _qs_inverse_best(record: dict[str, Any]) -> float:
    educations = record.get("educations", [])
    ranks = [parse_qs(e.get("qs_ranking", "")) for e in educations]
    ranks = [r for r in ranks if r > 0]
    if not ranks:
        return 0.0
    best = min(ranks)
    return float(1.0 / best)


FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "qs_top_25": FeatureSpec("qs_top_25", lambda r: _qs_in_range(r, 25), "binary"),
    "qs_top_50": FeatureSpec("qs_top_50", lambda r: _qs_in_range(r, 50), "binary"),
    "qs_top_100": FeatureSpec("qs_top_100", lambda r: _qs_in_range(r, 100), "binary"),
    "qs_top_200": FeatureSpec("qs_top_200", lambda r: _qs_in_range(r, 200), "binary"),
    "qs_inverse_best": FeatureSpec("qs_inverse_best", _qs_inverse_best, "continuous"),
    "prior_ipos": FeatureSpec("prior_ipos", _prior_ipos, "binary"),
    "prior_acquisitions": FeatureSpec(
        "prior_acquisitions", _prior_acquisitions, "binary"
    ),
    "large_company_years": FeatureSpec(
        "large_company_years", _large_company_years, "continuous"
    ),
    "experience_duration": FeatureSpec(
        "experience_duration", _experience_duration, "continuous"
    ),
    "technical_experience_duration": FeatureSpec(
        "technical_experience_duration", _technical_experience_duration, "continuous"
    ),
}


FEATURE_SETS: dict[str, list[str]] = {
    "base_only": [],
    "custom_only": list(FEATURE_REGISTRY.keys()),
    "base_plus_custom": list(FEATURE_REGISTRY.keys()),
}
