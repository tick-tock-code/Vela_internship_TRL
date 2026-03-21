# LLM Field Similarity Map (Snippet vs Experiments A/B/E - Numeric Only, Good-Match Threshold)

Purpose: Provide a supervisor-friendly summary of (1) the experiment prompts in plain language, (2) the numeric fields and their definitions, (3) the strict matching between experiment fields and the example snippet fields, and (4) the best-run feature weights.

**Experiment prompts in plain language (A and E)**
- Experiment A asks the model to score four traits from 1 to 5 and briefly justify each: career progression strength, ownership/responsibility, career coherence, and scrappiness (resourcefulness). The output is a JSON object with one numeric score per trait (plus justifications, which are not used in this document).
- Experiment E asks the model to identify what traditional metrics might miss or misread, then score hidden upside and hidden risk from 1 to 5, with justifications (not used here). The output is a JSON object with numeric scores for hidden upside and hidden risk.

**Snippet fields (example) - short descriptions**
- `prior_founding_attempt`: whether the founder previously founded or co-founded any company.
- `domain_expertise_depth`: depth of specialized expertise in a specific domain (1-5).
- `highest_seniority_reached`: highest role level prior to current founding (categorical).
- `evidence_of_prior_exit`: explicit evidence of IPO/acquisition/company sale.
- `career_narrative_type`: narrative archetype (builder/climber/academic/hybrid/unclear).
- `domain_focus_consistency`: consistency of domain focus over career (1-5).
- `conviction_indicator`: strength of commitment/conviction signals (1-5).

**Experiment numeric fields (A/B/E) - short descriptions**
- `trajectory_strength`: perceived strength of career progression (1-5).
- `ownership_signal`: perceived ownership and responsibility taken (1-5).
- `career_coherence`: perceived coherence/consistency of the career path (1-5).
- `scrappiness`: perceived resourcefulness/grit (1-5).
- `rubric_score`: overall score against a founder-quality rubric (1-5).
- `hidden_upside`: upside not captured by traditional metrics (1-5).
- `hidden_risk`: risks not captured by traditional metrics (1-5).
- `evidence_support_rating`: degree to which the profile provides concrete evidence for the judgments (1-5).

**Good matches**
- Snippet `conviction_indicator` <-> Experiments `ownership_signal`
- Snippet `domain_focus_consistency` <-> Experiments `career_coherence`

**No snippet counterpart (numeric experiments)**
- `trajectory_strength`
- `scrappiness`
- `rubric_score`
- `hidden_upside`
- `hidden_risk`
- `evidence_support_rating`

**Snippet fields with no experiment counterpart**
- `prior_founding_attempt`
- `evidence_of_prior_exit`
- `highest_seniority_reached`
- `domain_expertise_depth`
- `career_narrative_type`

**Best run summary (reasoning-only features)**
- Run: `set_05` | LLM Engineered + Reasoning [A+E] | F0.5=0.288 +/- 0.078
- Weightings (reasoning-based parameters only):
  feature: A_career_coherence | coef_mean: +0.6015 | coef_std: 0.0296
  feature: E_hidden_upside | coef_mean: -0.3594 | coef_std: 0.0517
  feature: A_ownership_signal | coef_mean: -0.2824 | coef_std: 0.0376
  feature: A_scrappiness | coef_mean: -0.2774 | coef_std: 0.0478
  feature: A_evidence_support_rating | coef_mean: +0.2308 | coef_std: 0.0468
  feature: E_hidden_risk | coef_mean: +0.2249 | coef_std: 0.0577
  feature: A_trajectory_strength | coef_mean: +0.0191 | coef_std: 0.0718
  feature: E_evidence_support_rating | coef_mean: +0.0101 | coef_std: 0.1009

**Best run summary (full feature list)**
- Set: `set_05` | Combo: A+E
- feature: has_exit_event | coef_mean: +1.1998 | coef_std: 0.0642
- feature: industry_bio_nano | coef_mean: +0.7495 | coef_std: 0.0712
- feature: education_high_qs | coef_mean: +0.7174 | coef_std: 0.0638
- feature: has_multiple_exits | coef_mean: +0.6902 | coef_std: 0.1193
- feature: A_career_coherence | coef_mean: +0.6015 | coef_std: 0.0296
- feature: industry_consumer_or_media | coef_mean: -0.4822 | coef_std: 0.0549
- feature: has_experience_in_fortune_500_companies | coef_mean: +0.4691 | coef_std: 0.1099
- feature: industry_tech_or_internet | coef_mean: +0.4647 | coef_std: 0.0498
- feature: education_in_top_qs | coef_mean: +0.4401 | coef_std: 0.0727
- feature: industry_services_or_health | coef_mean: -0.4360 | coef_std: 0.1133
- feature: E_hidden_upside | coef_mean: -0.3594 | coef_std: 0.0517
- feature: prior_ipo | coef_mean: +0.3359 | coef_std: 0.1606
- feature: A_ownership_signal | coef_mean: -0.2824 | coef_std: 0.0376
- feature: A_scrappiness | coef_mean: -0.2774 | coef_std: 0.0478
- feature: A_evidence_support_rating | coef_mean: +0.2308 | coef_std: 0.0468
- feature: experience_in_startups | coef_mean: +0.2264 | coef_std: 0.0438
- feature: E_hidden_risk | coef_mean: +0.2249 | coef_std: 0.0577
- feature: has_experience_in_high_growth_companies | coef_mean: -0.1675 | coef_std: 0.0868
- feature: has_experience_in_fortune_1000 | coef_mean: +0.1484 | coef_std: 0.0478
- feature: has_advanced_degree | coef_mean: -0.1340 | coef_std: 0.0439
- feature: has_phd | coef_mean: +0.0444 | coef_std: 0.0580
- feature: has_experience_in_nano_or_bio | coef_mean: -0.0394 | coef_std: 0.1334
- feature: A_trajectory_strength | coef_mean: +0.0191 | coef_std: 0.0718
- feature: E_evidence_support_rating | coef_mean: +0.0101 | coef_std: 0.1009

Notes:
- This comparison excludes all text justifications by design.
