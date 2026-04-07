# Readings and Recommended Order

This note condenses the suggested readings from section 5 of the proposal, assesses the ChatGPT interpretation, and turns them into a practical experiment order for this repo.

Update:

- this note predates the evidence-first rewrite
- where it says `admission`, read `Step 1 evidence mapping / triage`
- the current canonical Step 1 logic now lives in [step1/02_protocol.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step1/02_protocol.md), [step1/05_hq_anchor_and_route_protocol.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step1/05_hq_anchor_and_route_protocol.md), and [step1/06_step1_family_diagnostics_and_admission.md](C:/Users/joelb/OneDrive/Vela_partnerships_project/Project_folder/LLM_Reasoning_Main/docs/instability_control/step1/06_step1_family_diagnostics_and_admission.md)

## Short Summary

The readings point to four different levers for the instability problem:

- stability across resamples
- explicit structure over correlated feature families
- compression of dense correlated banks
- optional use of semantic or domain priors

That is useful, because those levers match the failure mode already seen in this project:

- richer LLM-derived families can raise CV
- but correlated or low-support additions can collapse on holdout

The key implication is that the project should not jump straight to a new fancy selector. It should build a ladder:

1. diagnose family instability clearly
2. map raw failure versus transform-sensitive signal
3. compare one grouped route and one compression route
4. treat LLM-guided selection as a later extension, not the first fix

## Reading Themes

### 1. Stability Selection

Main value:

- repeatedly subsample
- measure how often a feature or family survives
- prefer selections that recur reliably

Why it matters here:

- directly attacks the CV-to-holdout collapse problem
- gives a principled way to say a family is unstable even if it looks good on one split

Best use in this project:

- as a family evidence or triage filter
- as a robustness check on selected engineered or reasoning features

### 2. Sparse-Group Lasso

Main value:

- selects at the group level and the feature level
- can drop entire families while keeping only a few useful members in retained families

Why it matters here:

- your features already come in meaningful families
- pure lasso is too willing to arbitrarily choose one member of a correlated set

Best use in this project:

- as the main structured-selection route once the family map is fixed

### 3. Sparse PLS

Main value:

- compresses correlated features
- keeps only signal-bearing directions
- blends dimension reduction with sparsity

Why it matters here:

- this repo already shows that compression helps more than naive concatenation
- PLS is already partly validated in your existing work

Best use in this project:

- as the low-risk compression baseline
- probably the first nontrivial route to keep in the paper

### 4. Supervised Feature Grouping

Main value:

- learns groups from data rather than relying only on hand-defined groups
- stabilises selection among correlated predictors

Why it matters here:

- some of your current family definitions are natural, but some banks may still be too coarse
- the engineered bank in particular may benefit from sub-grouping

Best use in this project:

- after the first family-level results
- especially if one family looks useful but internally unstable

### 5. LLM-Select

Main value:

- uses semantic understanding to suggest subsets or groupings

Why it matters here:

- could help organise engineered features or reasoning outputs when names/descriptions contain meaning

Best use in this project:

- later-stage grouping aid
- not as the first decision-maker for what enters the final model

### 6. LLM-Lasso

Main value:

- uses LLM knowledge to modify regularisation or feature penalties

Why it matters here:

- can inject domain priors
- may help when purely statistical selection is underpowered or unstable

Best use in this project:

- optional extension or ablation
- only after a strong data-driven baseline exists

## View on the ChatGPT Response

The response is useful as a high-level orientation note. It captures the broad logic correctly:

- stability selection for robustness
- grouped penalties for family structure
- sparse PLS for correlated high-dimensional banks
- LLM-guided methods as knowledge-aware extensions

But it is still too smooth and too optimistic in three ways.

### 1. It treats all methods as if they belong in the same phase

They do not.

- stability selection, grouped penalties, and sparse PLS address your immediate empirical problem
- LLM-guided selection is much later-stage and much easier to misuse

### 2. It jumps too quickly from literature summary to method stack

The suggested stack:

- use LLM or clustering to define groups
- apply sparse-group lasso or SPLS
- validate with stability selection

is plausible, but not the best first move for this repo.

For this project, the safer order is:

- define the family map from the existing pipeline outputs first
- run stability-aware diagnostics and evidence mapping next
- only then introduce grouped or semantic refinements

### 3. It underplays the value of your existing results

This repo is not starting from zero.

- you already know naive feature accumulation is risky
- you already have a promising compression route
- you already have meaningful family candidates such as `A`, `D`, and `F`

So the best next step is not "try everything." It is "build a controlled experiment ladder around what already looks real."

## Recommended Order for This Repo

This is the order I would use.

### Phase 1: Family Diagnostics and Evidence Mapping

Start here.

Goal:

- turn the current intuition into a hard artifact

Deliverables:

- one diagnostics table by family
- one clean evidence map from `HQ`
- one naive concatenation control

Why first:

- this is the shortest path from current evidence to a paper-worthy claim
- it does not require committing to a new modelling framework too early

Methods to use:

- repeated CV or repeated subsampling
- family-level incremental lift
- selection frequency or evidence frequency across resamples

Decision rule:

- classify units as `raw_fail`, `transform_sensitive`, or `no_reproducible_evidence`

### Phase 2: Compression Route

Do this second.

Goal:

- test whether correlated families are more useful when compressed than when added raw

Why second:

- PLS-like routes are already partly supported by your current results
- this gives you a strong low-risk benchmark against naive concatenation

Methods to use:

- current PLS route
- optionally sparse PLS if you want one cleaner extension beyond plain PLS

Decision rule:

- keep the compression route if it reduces CV-to-holdout gap while preserving ranking quality

### Phase 3: Structured Group Selection

Do this third.

Goal:

- test whether explicit grouped sparsity outperforms plain evidence-mapped raw routes plus compression

Why third:

- by this point you will already know whether the problem is mostly "too many raw correlated features" or "some families are not real"

Methods to use:

- sparse-group lasso or a practical approximation at family level

Decision rule:

- keep this route if it beats or matches the compression route with better interpretability or better stability

### Phase 4: Supervised Group Refinement

Do this only if needed.

Goal:

- refine groups inside banks that still look internally unstable

Likely targets:

- deterministic engineered bank
- possibly reasoning bank if family-level groups are still too blunt

Why later:

- you should not learn new group definitions until you know which current families are actually worth saving

### Phase 5: LLM-Guided Grouping or Penalty Design

Treat this as an extension, not the core paper path.

Goal:

- test whether semantic priors improve grouped selection or penalty assignment

Why last:

- this is the easiest part to overclaim
- it adds another layer of uncertainty to a project already about instability

Best framing:

- optional ablation
- semantic assistance layered on top of a stable statistical core

## Concrete Recommendation

If I were choosing only three things for the next week, I would do this:

1. family diagnostics plus evidence mapping with repeated resampling
2. compression route built around the existing PLS work
3. one grouped selection route, preferably sparse-group-lasso style or a family-level approximation

I would not prioritise LLM-Select or LLM-Lasso this week.

Reason:

- they are interesting
- they are not the shortest path to answering your actual problem
- they also create an avoidable risk that the paper turns into "LLMs all the way down" instead of a defensible instability-control paper

## Recommended Working Thesis

The strongest paper story is probably:

- LLM-derived families do contain signal
- but naive accumulation inflates validation
- stability-aware evidence mapping and compression are more reliable than raw concatenation
- family structure matters more than simply generating more features

That is cleaner and more defensible than jumping immediately to LLM-guided regularisation.

## Immediate Next Actions

1. Freeze the family registry and anchor baseline.
2. Build the diagnostics table and Step 1 evidence-map output.
3. Treat naive concatenation as the explicit negative control.
4. Keep PLS as the first compression benchmark.
5. Add one grouped route only after the Step 1 evidence artifact exists.

That order gives you a coherent experimental narrative and avoids spending the week on methods that are interesting but not yet decision-relevant.
