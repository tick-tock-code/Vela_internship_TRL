# Think Reason Learn Example Use Cases

These examples are adapted from the local README and code patterns.

## 1. GPTree for explainable decision paths

Use GPTree to build a decision tree where each split question is generated and
answered by LLMs. This is useful when the raw input is text or semi-structured
and you need explicit, human-readable reasoning at each step.

Example domains:
- Venture capital founder success prediction
- Legal case triage with transparent paths
- Healthcare intake routing with explainable rules

## 2. Random Rule Forest for transparent ensembles

RRF is ideal when you want a large set of simple YES/NO rules, each generated
by an LLM, then scored and aggregated into a final prediction.

Example domains:
- Early-stage startup screening with rule transparency
- Customer support ticket classification with auditability
- Compliance risk checks with a clear list of triggering rules

## 3. Policy Induction for weighted policy sets

Policy Induction generates policies (natural language rules) and learns a
weighting with logistic regression. The result is a model that can be inspected
as a weighted policy ensemble.

Example domains:
- Risk scoring where policies must be reviewed by analysts
- Credit or underwriting models with transparent thresholds
- Internal decision playbooks with measurable predictive strength

## 4. Feature generation and safe evaluation

The `features` module generates binary lambda rules from LLMs and evaluates
them deterministically with a restricted builtins set. This gives you
LLM-generated features that are cheap to run at scale.

Example domains:
- Form parsing into boolean features
- Structured signal extraction from CRM records
- Rapid prototyping of rule-based features

## 5. Hybrid flows

You can combine components:
- Use Feature Generation to create candidate features.
- Feed features into GPTree or RRF for explainable classification.
- Use Policy Induction as an interpretable fallback model.

