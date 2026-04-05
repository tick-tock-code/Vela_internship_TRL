# Next Steps: Stability-Aware Integration of LLM-Derived Feature Families

This note turns the proposed follow-on direction into a concrete plan against the current codebase.

## Project Shift

The main change in direction is that the project should no longer ask only:

- Can LLM-derived features beat the baseline?

It should now ask:

- How can LLM-derived feature families be admitted without causing validation-to-holdout collapse?

That is a different project framing.

Your current project has mostly been an empirical model-comparison project:

- generate human, LLM-engineered, and reasoning features
- compare feature combos directly in LR/XGB/MLP models
- look for higher CV or test F0.5
- observe that richer reasoning combos often raise CV but overfit on private test
- test partial fixes such as pruning, ElasticNet, and PLS compression

The proposed project reframes this as a methodology paper about controlling instability in correlated feature families:

- the unit of analysis becomes feature families, not just individual feature sets
- the goal becomes stable integration, not just higher validation scores
- families should be diagnosed for redundancy, conditional value over baseline, fold stability, calibration, and precision@k
- families should be added sequentially and retained only if they remain stable across resamples
- naive concatenation should be treated as a negative control, not the main route
- the model ladder should include group-aware and compression-aware routes
- residual and borderline-case analysis should test orthogonal value, not just score inflation

This is not a reset. It is a formalisation of what the existing results are already pointing toward.

## What You Already Have

### Core pipeline and experiment infrastructure

- `vcbench_pipeline.py` already supports the main feature sources: human, LLM-engineered, and reasoning-derived features.
  File: [vcbench_pipeline.py](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/python/pipelines/vcbench_pipeline.py)
- `paper_pipeline.py` already runs paper-style evaluation blocks and produces validation and private-test outputs.
  File: [paper_pipeline.py](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/python/pipelines/paper_pipeline.py)
- `model_testing_pipeline.py` already contains the newer stability-oriented experiment machinery, including aggressive pruning, ElasticNet, PLS, collinearity reports, and final-model comparison tables.
  File: [model_testing_pipeline.py](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/python/pipelines/model_testing_pipeline.py)

### Evidence that the instability problem is real

- The earlier paper summary already shows the core failure mode: reasoning features often improved validation but failed to deliver comparable private-test gains.
  File: [paper_experiment_summary.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/Summaries_to_build_report/paper_experiment_summary.md)
- The current docs already describe the project as combining human, LLM-engineered, and reasoning features.
  File: [README.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/README.md)

### Early stability-control work is already underway

- Aggressive pruning plus ElasticNet has already been run on reduced reasoning combos.
  File: [summary.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/Summaries_to_build_report/model_Checkpoint_summaries/summary.md)
- PLS sweeps have already been run, and `n_components = 6` was selected as the default stable choice.
  File: [summary_pls_n_components.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/Summaries_to_build_report/model_Checkpoint_summaries/summary_pls_n_components.md)
- You already have a shortlist of final models and a test-side comparison table showing which compressed variants held up better.
  Files:
  [summary_final_models.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/Summaries_to_build_report/model_Checkpoint_summaries/summary_final_models.md)
  [summary_final_table.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/Summaries_to_build_report/model_Checkpoint_summaries/summary_final_table.md)

### The current project already hints at family-level thinking

- Reasoning prompts are already grouped into experiments A-F.
- Some summaries already discuss certain families as more stable than others, especially `A`, `D`, and `F`.
- The code already supports combo-based evaluation, which is close to family admission testing.

## What Is Missing

The missing pieces are not mostly about raw modelling capacity. They are about making the project explicitly family-aware and stability-aware.

### 1. A formal family-diagnostics layer

Right now you have collinearity reports and model results, but not one unified family diagnostics table that answers:

- how redundant is each family relative to the baseline and to other families?
- what is the conditional gain over the structured baseline?
- how stable is that gain across folds or repeated resamples?
- what happens to calibration, precision@k, and threshold sensitivity when the family is added?

### 2. A controlled-admission framework

Right now you have many combo comparisons, but not yet one explicit admission protocol:

- start from the structured baseline
- add one family at a time
- retain it only if inner-CV improves and the gain is stable across folds or repeated splits
- compare this against naive concatenation as a negative control

### 3. Group-aware methods beyond plain pruning and PLS

You already have:

- pruning
- ElasticNet
- PLS

You do not yet have a dedicated grouped route such as:

- stability selection
- sparse-group-lasso style family selection
- supervised grouping over correlated predictors

This is one of the clearest gaps between the current project and the proposal.

### 4. Residual and borderline-case analysis

The proposal explicitly asks whether added families explain residual error or improve ranking among borderline cases. That analysis is not yet a first-class output.

Missing outputs:

- which founders does HQ get wrong that family `A` or `F` fixes?
- do added families help only on easy cases, or on borderline cases where ranking matters?
- do they improve top-k precision even when global threshold metrics move only slightly?

### 5. A compact final representation of the deterministic LLM-engineered bank

The proposal asks for one interpretable reduced representation of the engineered bank:

- 10-30 retained features, or
- grouped summaries, or
- labelled components

You already have engineered feature families and PLS routes, but not yet one clearly documented final reduced representation built for the paper.

### 6. A single paper-ready comparison focused on CV-to-holdout gap

You have the ingredients, but not yet one final comparison table built around the new thesis:

- structured baseline
- naive concatenation control
- one sparse/stability-aware admission route
- one compression route

The ranking criterion should emphasise reduction in CV-to-holdout collapse, not just highest CV.

## What To Do Next

Use the next week to convert the existing experiment stack into a stability-aware workflow.

### Immediate project objective

By the end of the week, we want a decision-grade answer to:

- Which LLM-derived families add robust out-of-sample value beyond the strongest structured baseline?
- Which integration route best preserves holdout performance while staying interpretable?

### One-week execution checklist

#### Day 1: lock the baseline and family definitions

- [ ] Freeze the anchor baseline.
  Suggested anchor: `HQ` or the strongest structured baseline currently used in `model_testing_pipeline.py`.
- [ ] Freeze the family definitions.
  Minimum families:
  - `HQ`
  - deterministic LLM-engineered family or sub-families
  - reasoning families `A` through `F`
- [ ] Write down the exact family map in `docs/instability_control/`.
- [ ] Decide whether the deterministic engineered bank will be treated as one family or clustered into sub-families.

#### Day 2: build the family diagnostics table

- [ ] Add one script or report path that outputs, for each family:
  - feature count
  - overlap/correlation burden
  - incremental CV gain over HQ
  - fold-to-fold variance
  - CV-to-test gap where available
  - calibration or threshold drift
  - precision@k impact
- [ ] Save this as a single CSV and markdown summary under `LLM_Reasoning_Main/docs/instability_control/`.

#### Day 3: implement controlled admission

- [ ] Create a sequential family-admission experiment:
  - start with HQ
  - test adding one family at a time
  - keep only families that improve inner-CV and remain stable across folds/resamples
- [ ] Add naive concatenation as the negative control.
- [ ] Produce a compact comparison between:
  - HQ only
  - HQ + naive full family concat
  - HQ + selected stable families

#### Day 4: add one grouped route and one compression route

- [ ] Keep the compression route based on PLS because that is already working.
- [ ] Add one grouped/stability-aware route.
  Strong candidates:
  - stability selection over repeated subsamples
  - family-level grouped selection
- [ ] Keep scope tight: one grouped route is enough for this week if it is cleanly implemented and tested.

#### Day 5: run residual and borderline-case analysis

- [ ] Compare errors made by HQ vs the best stability-aware route.
- [ ] Identify whether added families improve:
  - borderline ranking
  - top-k precision
  - founders near the classification threshold
- [ ] Write one short diagnostic note with examples of where added families helped and where they destabilised the model.

#### Day 6: produce paper-facing outputs

- [ ] Create one final comparison table with:
  - baseline
  - naive concatenation control
  - selected stability-aware route
  - selected compression route
- [ ] Create one compact figure showing:
  - family gains
  - instability burden
  - CV-to-holdout gap
- [ ] Write a one-page interpretation note answering whether any reasoning family adds robust orthogonal value.

#### Day 7: final decision and paper framing

- [ ] Choose the main paper story:
  - positive result: one stability-aware route reduces collapse and preserves holdout value
  - null result: richer families mostly inflate validation unless tightly filtered
- [ ] Lock the fallback framing if gains are mixed:
  - benchmark/protocol paper on imbalance, hidden-test shift, and validation inflation
- [ ] Update the docs to reflect the final direction and archive dead-end branches cleanly.

## Suggested Codebase Work Order

- [ ] Start from [model_testing_pipeline.py](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/python/pipelines/model_testing_pipeline.py), not `paper_pipeline.py`.
  Reason: it already contains pruning, ElasticNet, PLS, collinearity reporting, and finalising logic closer to the proposed direction.
- [ ] Keep [paper_pipeline.py](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/python/pipelines/paper_pipeline.py) as the paper-output layer, not the main experimental sandbox.
- [ ] Add new outputs under `LLM_Reasoning_Main/docs/instability_control/`.
- [ ] Only move logic into `paper_pipeline.py` once the admission and diagnostics workflow is stable.

## Working Thesis For The Week

LLM-derived feature families contain real signal, but the main challenge is not generating richer features. The main challenge is admitting correlated, low-support families in a way that preserves holdout performance under hidden-test shift.

## Success Criteria

At the end of the week, this direction is successful if we have:

- one clear family diagnostics artifact
- one controlled admission workflow
- one grouped or stability-aware route
- one compression route
- one final table centred on CV-to-holdout gap
- one written answer on whether any reasoning family adds robust orthogonal value beyond HQ

If the method gains are weak, the week is still successful if we can show, cleanly and convincingly, that naive addition of richer families inflates validation more than true holdout performance.
