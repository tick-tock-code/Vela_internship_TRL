# Think Reason Learn Documentation Notes

This is a compact, practical guide based on the local docs and code.

## Installation

From PyPI:
```bash
pip install think-reason-learn
```

From source:
```bash
git clone https://github.com/vela-research/think-reason-learn.git
cd think-reason-learn
poetry install
```

Prerequisites noted in the repository:
- Python 3.13+
- Graphviz system package for tree visualization

## Configuration and API keys

`think_reason_learn.core._config.Settings` reads keys from `.env` and
environment variables:

- `OPENAI_API_KEY`
- `GOOGLE_AI_API_KEY`
- `XAI_API_KEY`
- `ANTHROPIC_API_KEY`

The LLM router (`think_reason_learn.core.llms.LLM`) validates that the
provider key is set before it attempts a call.

## LLM selection

You pass a list of provider choices in priority order, for example:

```python
from think_reason_learn.core.llms import OpenAIChoice, GoogleChoice

llm_priority = [
    GoogleChoice(model="gemini-1.5-flash-latest"),
    OpenAIChoice(model="gpt-4o-mini"),
]
```

If a provider fails, the router falls back to the next choice.

## Core algorithms

### GPTree
LLM-guided decision tree classifier.

Key methods:
- `set_task(...)` or `set_tasks(...)` to build question generation prompts
- `fit(...)` to build or resume the tree (async generator)
- `predict(...)` to label new samples (async generator)
- `save(...)` and `load(...)` for persistence

### RRF (Random Rule Forest)
Binary classifier with LLM-generated YES/NO questions.

Key methods:
- `set_tasks(...)`
- `fit(...)`
- `predict(...)`
- `save(...)` and `load(...)`

### Policy Induction
LLM-generated policies with logistic regression weighting.

Key methods:
- `set_task(...)`
- `fit(...)`
- `predict(...)`
- `save(...)` and `load(...)`

### Feature Generation
LLM-generated lambda rules with safe evaluation.

Key methods:
- `FeatureGenerator.generate(...)`
- `FeatureEvaluator.evaluate(...)` and `evaluate_df(...)`

## Data formats

Most algorithms expect a `pandas.DataFrame` and class labels as strings.
Some algorithms enforce labels to be "YES" or "NO".

## Persistence format

Artifacts are stored in a directory:
- JSON manifest with parameters and metadata
- Parquet files for questions, answers, and data
- Optional model files (Policy Induction uses `lr.joblib`)

