# Data Flow (Plain Text + Diagram)

## Plain-Text Flow
1. **Load dataset** (`vcbench_final_public.csv` or sample).
2. **Split train/test** once with a fixed random seed.
3. **Extract features**:
   - Baseline human features from the example script.
   - Custom features from the registry.
   - LLM-engineered rules (train-only generation, evaluated on all).
   - LLM-reasoning features (per-founder prompt outputs).
4. **Assemble feature matrix** for the selected mode.
5. **Standardize continuous features** (train stats only).
6. **Save feature datasets** to `features_storage/`.
7. **Train logistic regression** and tune threshold on training set.
8. **Evaluate metrics** on test set.
9. **Write logs and run reports** to `training_logs/`.

## Data Flow Diagram (Mermaid)
```mermaid
flowchart TD
  A["Raw CSV"] --> B["Load records + labels"]
  B --> C["Train/Test Split"]
  C --> D["Train Records"]
  C --> E["Test Records"]

  D --> F["Baseline Feature Extractor"]
  D --> G["Custom Feature Registry"]
  D --> H["LLM Rule Generation\n(train only)"]
  D --> I["LLM Reasoning Prompting\n(per founder)"]

  F --> J["Train Feature Matrix"]
  G --> J
  H --> J
  I --> J

  E --> K["Test Feature Matrix\n(LLM rules evaluated)"]
  E --> L["Reasoning Features\n(per founder)"]
  L --> K

  J --> M["Standardize (train stats)"]
  M --> N["Train Logistic Regression"]
  K --> O["Test Evaluation"]

  J --> P["features_storage/ (parquet)"]
  K --> P
  O --> Q["training_logs/ (txt + json)"]
```
