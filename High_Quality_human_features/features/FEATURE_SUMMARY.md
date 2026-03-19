# Feature Summary — 23 Structured Features (Train Set, n=3,600)

Generated from `features/extract_structured.py` on 2026-03-08.

## Tier 1: Direct Exit Signals (3 features)

| Feature | Type | Null% | Distribution | Success Rate by Value |
|---|---|---|---|---|
| `has_prior_ipo` | binary | 0% | 0: 99.7%, 1: 0.3% | 0→8.9%, 1→41.7% |
| `has_prior_acquisition` | binary | 0% | 0: 97.2%, 1: 2.8% | 0→8.6%, 1→24.2% |
| `exit_count` | int 0–2 | 0% | 0: 97.1%, 1: 2.8%, 2: 0.1% | 0→8.5%, 1→22.8%, 2→60.0% |

**Takeaway:** Exit signals are rare (2.9%) but extremely predictive. IPO holders have 41.7% success rate (4.7x baseline). This is the foundation for the high-precision rule layer.

---

## Tier 2: Sacrifice Signal (5 features)

| Feature | Type | Null% | Min | Max | Mean | Median | Std |
|---|---|---|---|---|---|---|---|
| `max_company_size_before_founding` | int 0–9 | 0% | 0 | 9 | 5.3 | 7.0 | 3.8 |
| `prestige_sacrifice_score` | int 0–45 | 0% | 0 | 45 | 17.4 | 12.0 | 16.6 |
| `years_in_large_company` | float | 0% | 0.0 | 70.0 | 3.4 | 1.0 | 5.8 |
| `comfort_index` | float | 0% | 0.0 | 787.5 | 9.6 | 0.0 | 46.1 |
| `founding_timing` | float | 0% | 0.0 | 100.0 | 8.6 | 6.5 | 8.9 |

**Distribution highlights:**
- `max_company_size_before_founding`: bimodal — 27.3% at 0 (no pre-founding job), 37.4% at 9 (10001+). Success rate: 0→7.4%, 5→12.3%, 9→11.5%.
- `prestige_sacrifice_score`: 28% at 0 (no sacrifice measurable), 16% at 45 (max). Heavy right tail.
- `comfort_index`: 85.2% at 0.0 (no high-comfort industry experience). Sparse but potentially discriminative.
- `founding_timing`: 13.7% at 0 (founded immediately), median 6.5 years of prior experience.

---

## Tier 3: Education × QS Interaction (6 features)

| Feature | Type | Null% | Distribution | Success Rate by Value |
|---|---|---|---|---|
| `edu_prestige_tier` | int 0–4 | 0% | 0: 11.5%, 1: 57.4%, 2: 6.1%, 3: 13.3%, 4: 11.7% | 0→8.4%, 1→6.5%, 2→8.7%, 3→11.3%, 4→19.5% |
| `field_relevance_score` | int 1–5 | 0% | 1: 27.8%, 2: 3.8%, 3: 23.5%, 4: 23.8%, 5: 21.2% | 1→7.9%, 3→7.0%, 5→13.5% |
| `prestige_x_relevance` | int 0–20 | 0% | 14 unique values, range 0–20 | 0→8.4%, 1→4.4%, 15→15.4%, 20→24.2% |
| `degree_level` | int 0–4 | 0% | 0: 22.2%, 1: 32.7%, 2: 14.2%, 3: 18.9%, 4: 11.9% | 0→8.5%, 1→8.1%, 3→10.0%, 4→12.6% |
| `stem_flag` | binary | 0% | 0: 55.1%, 1: 44.9% | 0→7.6%, 1→10.8% |
| `best_degree_prestige` | int 0–4 | 0% | same as edu_prestige_tier | same as edu_prestige_tier |

**Takeaway:** `prestige_x_relevance` is the strongest education signal. Score of 20 (top-10 QS × STEM in tech) → 24.2% success (2.7x baseline). The interaction term captures more than either dimension alone.

---

## Tier 4: Career Trajectory (9 features)

| Feature | Type | Null% | Distribution | Key Stats |
|---|---|---|---|---|
| `max_seniority_reached` | int 0–5 | 0% | 0: 0.8%, 1: 15.1%, 2: 6.4%, 3: 4.2%, 4: 6.7%, 5: 66.9% | 3→11.9%, 4→11.3%, 5→9.3% |
| `seniority_is_monotone` | binary | 0% | 0: 63.6%, 1: 36.4% | 0→9.5%, 1→8.1% |
| `company_size_is_growing` | binary | 0% | 0: 75.4%, 1: 24.6% | 0→9.6%, 1→7.2% |
| `restlessness_score` | int 0–17 | 0% | mean=2.3, median=2 | 0→(n=627), 1→(n=975) |
| `founding_role_count` | int 0–11 | 0% | 0: 29.8%, 1: 29.1%, 2: 19.0%, 3: 11.3% | 4→11.3%, 6→17.5% |
| `longest_founding_tenure` | float 0–10 | 0% | 0: 32.9%, 1: 19.4%, 2.5: 15.8% | 4.5→11.3% |
| `industry_pivot_count` | int 0–9 | 0% | mean=2.1, median=2 | 4→12.4% |
| `industry_alignment` | binary | 0% | 0: 64.2%, 1: 35.8% | 0→7.6%, 1→11.5% |
| `total_inferred_experience` | float 0–100 | 0% | mean=11.8, median=10.0, std=10.2 | — |

**Takeaway:**
- 66.9% of founders have max seniority 5 (founder/C-level) — expected for a VC dataset.
- `industry_alignment` shows signal: founders whose prior job industry matches their startup have 11.5% success rate vs 7.6%.
- `founding_role_count` of 6+ is rare but has 17.5% success rate (serial founders).

---

## Cross-Tier Signal Strength Summary

| Signal | Success Rate (high) | Success Rate (low) | Lift |
|---|---|---|---|
| `exit_count` ≥ 1 | 23.6% | 8.5% | 2.8x |
| `has_prior_ipo` = 1 | 41.7% | 8.9% | 4.7x |
| `prestige_x_relevance` = 20 | 24.2% | 4.4% (score=1) | 5.5x |
| `edu_prestige_tier` = 4 (top-10) | 19.5% | 6.5% (tier 1) | 3.0x |
| `industry_alignment` = 1 | 11.5% | 7.6% | 1.5x |
| `stem_flag` = 1 | 10.8% | 7.6% | 1.4x |
| `field_relevance_score` = 5 | 13.5% | 7.9% (score=1) | 1.7x |

**Overall:** Exit signals dominate (Tier 1), followed by the education×prestige interaction term (Tier 3). Sacrifice signal (Tier 2) and trajectory features (Tier 4) provide moderate but consistent lift.
