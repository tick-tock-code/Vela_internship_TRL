# LLM-Engineered Feature Classification Summary

This summary explains how the LLM is used to engineer lambda-based features in
the VCBench example and what context it does and does not receive.

## What the LLM does
The LLM generates **binary feature rules** expressed as Python lambdas. Each
lambda takes a single record (a dict) and returns True or False. These rules
become features in a downstream classifier.

## Where the instructions live
The rules and instructions are defined in:
`think_reason_learn/features/_prompts.py`

Key instruction blocks:
- Task: generate exactly N binary rules as Python lambdas.
- Rules: handle missing keys with `.get()`, handle empty lists safely, use
  `.lower()` for string comparisons, return booleans, and keep rules diverse.
- Constraints: use only standard Python and provided helper functions, and
  use the schema-defined parameter name.

## What context the LLM receives
The LLM sees:
- A **schema description** of the record structure (`VCBENCH_SCHEMA`), including
  field names and types.
- **Helper functions** (`VCBENCH_HELPERS`) it can call inside lambdas, with
  signatures and docstrings.
- **Labeled sample records** tagged `[SUCCESS]` or `[FAIL]`, which implicitly
  convey the target label.

This provides field-level context and outcome labels but relies on labeled
examples rather than an explicit natural-language objective.

## What the LLM does not receive in the minimal script
The LLM does not see:
- The **human-engineered features** (those are computed separately).
- An explicit task sentence like “predict VC success,” unless the schema’s
  description includes it.

## How the features are evaluated
The generated lambdas are compiled and executed locally:
- `FeatureEvaluator` compiles lambdas with a restricted set of builtins.
- Each lambda is run deterministically on each record to produce a 0/1 feature.
- Any lambda that fails compilation or execution yields 0.

## Summary in one sentence
The LLM only proposes the lambda logic, while the project enforces safety and
deterministic evaluation of those lambdas against structured records.

