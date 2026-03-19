# Feature Weight Summary (CV fold coefficients)

*CV: 10-fold stratified on 4400 founders (seed excluded).*

## Human Features
| feature | coef_mean | coef_std |
|---|---:|---:|
| prior_ipos | +0.9065 | 0.1859 |
| prior_acquisitions | +0.8975 | 0.0752 |
| multiple_exits | +0.5464 | 0.0979 |
| qs_top_200 | +0.5340 | 0.0622 |
| long_experience | -0.4552 | 0.0465 |
| industry_match | +0.4441 | 0.0433 |
| qs_top_25 | +0.4388 | 0.0571 |
| technical_role | +0.2507 | 0.0460 |
| senior_leadership | +0.1934 | 0.0462 |
| short_tenure_pattern | -0.1524 | 0.0544 |
| large_company_years | +0.1016 | 0.0101 |
| stem_degree | +0.0931 | 0.0293 |
| has_phd | +0.0911 | 0.0445 |
| startup_exp | -0.0682 | 0.0437 |
| qs_inverse_best | +0.0664 | 0.0179 |
| has_mba | -0.0511 | 0.0409 |

## Best Engineered-Only Set
Set: set_09
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exited_before | +1.1559 | 0.0483 |
| education_in_top_qs | +0.6581 | 0.0582 |
| has_experience_in_industry | +0.3884 | 0.0432 |
| has_experience_in_large_company | +0.3519 | 0.0404 |
| has_startup_experience | -0.3088 | 0.1018 |
| has_education_in_field | +0.2979 | 0.0533 |
| has_multiple_educations | -0.2967 | 0.0353 |
| has_high_qs_rank | +0.2344 | 0.0322 |
| industry_has_high_qs_rank | +0.2344 | 0.0322 |
| experience_long_duration | -0.1995 | 0.0511 |
| has_leadership_role | +0.1318 | 0.0916 |
| has_tech_experience | +0.1285 | 0.0272 |
| has_international_experience | +0.0781 | 0.2026 |
| has_education_bachelor_or_higher | -0.0467 | 0.0203 |
| industry_known | -0.0407 | 0.0568 |
| has_education_in_science_or_engineering | -0.0104 | 0.0218 |

## Best Engineered + Reasoning
Set: set_05  Combo: A+E
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exit_event | +1.1998 | 0.0642 |
| industry_bio_nano | +0.7495 | 0.0712 |
| education_high_qs | +0.7174 | 0.0638 |
| has_multiple_exits | +0.6902 | 0.1193 |
| A_career_coherence | +0.6015 | 0.0296 |
| industry_consumer_or_media | -0.4822 | 0.0549 |
| has_experience_in_fortune_500_companies | +0.4691 | 0.1099 |
| industry_tech_or_internet | +0.4647 | 0.0498 |
| education_in_top_qs | +0.4401 | 0.0727 |
| industry_services_or_health | -0.4360 | 0.1133 |
| E_hidden_upside | -0.3594 | 0.0517 |
| prior_ipo | +0.3359 | 0.1606 |
| A_ownership_signal | -0.2824 | 0.0376 |
| A_scrappiness | -0.2774 | 0.0478 |
| A_evidence_support_rating | +0.2308 | 0.0468 |
| experience_in_startups | +0.2264 | 0.0438 |
| E_hidden_risk | +0.2249 | 0.0577 |
| has_experience_in_high_growth_companies | -0.1675 | 0.0868 |
| has_experience_in_fortune_1000 | +0.1484 | 0.0478 |
| has_advanced_degree | -0.1340 | 0.0439 |
| has_phd | +0.0444 | 0.0580 |
| has_experience_in_nano_or_bio | -0.0394 | 0.1334 |
| A_trajectory_strength | +0.0191 | 0.0718 |
| E_evidence_support_rating | +0.0101 | 0.1009 |

## Best Engineered + Reasoning per Combo
### Combo: A
Set: set_01
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exited | +1.3642 | 0.0569 |
| industry_experience_in_current | +0.5146 | 0.0331 |
| A_career_coherence | +0.4791 | 0.0249 |
| has_education_in_top_ranked_institution | +0.4630 | 0.0169 |
| has_high_qs_rank | +0.4630 | 0.0169 |
| industry_known | +0.4580 | 0.0374 |
| A_ownership_signal | -0.3387 | 0.0409 |
| has_bachelor_degree | -0.3346 | 0.0400 |
| A_scrappiness | -0.3144 | 0.0513 |
| industry_experience_years | -0.2531 | 0.0662 |
| has_executive_experience | +0.1793 | 0.0534 |
| has_startup_experience | +0.1527 | 0.0387 |
| has_international_experience | +0.1505 | 0.0176 |
| has_role_in_high_growth_company | +0.1505 | 0.0176 |
| A_evidence_support_rating | +0.1468 | 0.0479 |
| has_advanced_degree | -0.1311 | 0.0506 |
| industry_specific_experience | -0.1193 | 0.0328 |
| industry_experience_in_relevant_role | -0.1193 | 0.0328 |
| has_higher_education_degree | +0.0859 | 0.0529 |
| has_professional_network | -0.0450 | 0.0539 |
| education_field_relevant | +0.0128 | 0.0362 |
| A_trajectory_strength | -0.0111 | 0.0679 |

### Combo: A+B
Set: set_05
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exit_event | +1.2467 | 0.0567 |
| industry_bio_nano | +0.7707 | 0.0720 |
| education_high_qs | +0.6927 | 0.0619 |
| has_multiple_exits | +0.5896 | 0.1335 |
| A_career_coherence | +0.5530 | 0.0234 |
| has_experience_in_fortune_500_companies | +0.5051 | 0.1197 |
| industry_services_or_health | -0.4683 | 0.1097 |
| industry_consumer_or_media | -0.4655 | 0.0494 |
| prior_ipo | +0.4517 | 0.1337 |
| industry_tech_or_internet | +0.4240 | 0.0475 |
| education_in_top_qs | +0.3846 | 0.0698 |
| A_scrappiness | -0.2958 | 0.0496 |
| A_ownership_signal | -0.2841 | 0.0385 |
| experience_in_startups | +0.2348 | 0.0413 |
| B_evidence_support_rating | -0.1988 | 0.1027 |
| has_advanced_degree | -0.1802 | 0.0408 |
| has_experience_in_high_growth_companies | -0.1802 | 0.0945 |
| A_evidence_support_rating | +0.1798 | 0.0474 |
| B_rubric_score | -0.1132 | 0.1011 |
| has_experience_in_fortune_1000 | +0.1055 | 0.0493 |
| has_phd | +0.0157 | 0.0599 |
| has_experience_in_nano_or_bio | +0.0091 | 0.0930 |
| A_trajectory_strength | -0.0073 | 0.0736 |

### Combo: A+B+E
Set: set_05
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exit_event | +1.2027 | 0.0693 |
| industry_bio_nano | +0.7521 | 0.0724 |
| education_high_qs | +0.7231 | 0.0638 |
| has_multiple_exits | +0.6981 | 0.1228 |
| A_career_coherence | +0.6058 | 0.0285 |
| industry_consumer_or_media | -0.4824 | 0.0546 |
| has_experience_in_fortune_500_companies | +0.4734 | 0.1111 |
| industry_tech_or_internet | +0.4668 | 0.0491 |
| education_in_top_qs | +0.4413 | 0.0742 |
| industry_services_or_health | -0.4363 | 0.1127 |
| prior_ipo | +0.3565 | 0.1651 |
| E_hidden_upside | -0.3436 | 0.0516 |
| A_ownership_signal | -0.2777 | 0.0382 |
| A_scrappiness | -0.2768 | 0.0489 |
| A_evidence_support_rating | +0.2327 | 0.0458 |
| experience_in_startups | +0.2273 | 0.0424 |
| E_hidden_risk | +0.2220 | 0.0570 |
| has_experience_in_high_growth_companies | -0.1718 | 0.0878 |
| has_experience_in_fortune_1000 | +0.1546 | 0.0469 |
| has_advanced_degree | -0.1293 | 0.0430 |
| has_phd | +0.0525 | 0.0569 |
| has_experience_in_nano_or_bio | -0.0405 | 0.1308 |
| B_rubric_score | -0.0332 | 0.1025 |
| B_evidence_support_rating | -0.0329 | 0.1124 |
| E_evidence_support_rating | +0.0304 | 0.1053 |
| A_trajectory_strength | +0.0253 | 0.0740 |

### Combo: A+E
Set: set_05
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exit_event | +1.1998 | 0.0642 |
| industry_bio_nano | +0.7495 | 0.0712 |
| education_high_qs | +0.7174 | 0.0638 |
| has_multiple_exits | +0.6902 | 0.1193 |
| A_career_coherence | +0.6015 | 0.0296 |
| industry_consumer_or_media | -0.4822 | 0.0549 |
| has_experience_in_fortune_500_companies | +0.4691 | 0.1099 |
| industry_tech_or_internet | +0.4647 | 0.0498 |
| education_in_top_qs | +0.4401 | 0.0727 |
| industry_services_or_health | -0.4360 | 0.1133 |
| E_hidden_upside | -0.3594 | 0.0517 |
| prior_ipo | +0.3359 | 0.1606 |
| A_ownership_signal | -0.2824 | 0.0376 |
| A_scrappiness | -0.2774 | 0.0478 |
| A_evidence_support_rating | +0.2308 | 0.0468 |
| experience_in_startups | +0.2264 | 0.0438 |
| E_hidden_risk | +0.2249 | 0.0577 |
| has_experience_in_high_growth_companies | -0.1675 | 0.0868 |
| has_experience_in_fortune_1000 | +0.1484 | 0.0478 |
| has_advanced_degree | -0.1340 | 0.0439 |
| has_phd | +0.0444 | 0.0580 |
| has_experience_in_nano_or_bio | -0.0394 | 0.1334 |
| A_trajectory_strength | +0.0191 | 0.0718 |
| E_evidence_support_rating | +0.0101 | 0.1009 |

### Combo: B
Set: set_05
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exit_event | +1.0581 | 0.0570 |
| industry_bio_nano | +0.8856 | 0.0633 |
| education_high_qs | +0.6840 | 0.0543 |
| has_multiple_exits | +0.6113 | 0.1396 |
| has_experience_in_fortune_500_companies | +0.5186 | 0.1120 |
| industry_consumer_or_media | -0.4398 | 0.0599 |
| industry_services_or_health | -0.4120 | 0.1029 |
| education_in_top_qs | +0.3976 | 0.0713 |
| prior_ipo | +0.3883 | 0.1222 |
| industry_tech_or_internet | +0.3576 | 0.0463 |
| has_experience_in_fortune_1000 | +0.2335 | 0.0464 |
| has_phd | +0.1728 | 0.0579 |
| has_experience_in_high_growth_companies | -0.1662 | 0.0912 |
| B_rubric_score | -0.1569 | 0.1024 |
| has_advanced_degree | -0.1203 | 0.0404 |
| has_experience_in_nano_or_bio | +0.0921 | 0.1275 |
| B_evidence_support_rating | -0.0677 | 0.0930 |
| experience_in_startups | +0.0033 | 0.0408 |

### Combo: B+E
Set: set_05
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exit_event | +1.0584 | 0.0683 |
| industry_bio_nano | +0.8821 | 0.0688 |
| education_high_qs | +0.7003 | 0.0561 |
| has_multiple_exits | +0.6972 | 0.1244 |
| has_experience_in_fortune_500_companies | +0.5126 | 0.1091 |
| industry_consumer_or_media | -0.4689 | 0.0558 |
| education_in_top_qs | +0.4347 | 0.0742 |
| industry_services_or_health | -0.3929 | 0.1054 |
| industry_tech_or_internet | +0.3821 | 0.0505 |
| E_hidden_upside | -0.3708 | 0.0511 |
| prior_ipo | +0.2836 | 0.1442 |
| has_experience_in_fortune_1000 | +0.2669 | 0.0464 |
| E_evidence_support_rating | +0.2111 | 0.1006 |
| has_phd | +0.1906 | 0.0576 |
| has_experience_in_high_growth_companies | -0.1673 | 0.0909 |
| E_hidden_risk | +0.1563 | 0.0564 |
| has_experience_in_nano_or_bio | +0.0990 | 0.1731 |
| has_advanced_degree | -0.0739 | 0.0388 |
| B_rubric_score | -0.0677 | 0.1023 |
| B_evidence_support_rating | +0.0553 | 0.1086 |
| experience_in_startups | +0.0077 | 0.0421 |

### Combo: E
Set: set_05
| feature | coef_mean | coef_std |
|---|---:|---:|
| has_exit_event | +1.0603 | 0.0555 |
| industry_bio_nano | +0.8833 | 0.0672 |
| education_high_qs | +0.6970 | 0.0578 |
| has_multiple_exits | +0.6897 | 0.1167 |
| has_experience_in_fortune_500_companies | +0.5127 | 0.1027 |
| industry_consumer_or_media | -0.4693 | 0.0623 |
| education_in_top_qs | +0.4334 | 0.0734 |
| industry_services_or_health | -0.3922 | 0.1077 |
| industry_tech_or_internet | +0.3813 | 0.0488 |
| E_hidden_upside | -0.3805 | 0.0533 |
| prior_ipo | +0.2779 | 0.1447 |
| has_experience_in_fortune_1000 | +0.2664 | 0.0485 |
| E_evidence_support_rating | +0.2119 | 0.0930 |
| has_phd | +0.1895 | 0.0588 |
| has_experience_in_high_growth_companies | -0.1670 | 0.0851 |
| E_hidden_risk | +0.1559 | 0.0582 |
| has_experience_in_nano_or_bio | +0.0952 | 0.1713 |
| has_advanced_degree | -0.0768 | 0.0396 |
| experience_in_startups | +0.0079 | 0.0422 |
