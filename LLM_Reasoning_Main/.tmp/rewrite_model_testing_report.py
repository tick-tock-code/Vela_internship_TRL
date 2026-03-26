import pandas as pd
from pathlib import Path

base = Path(r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder\LLM_Reasoning_Main")
results_path = base / "docs" / "model_testing" / "model_testing_results.csv"
report_path = base / "docs" / "model_testing" / "model_testing_report.md"
results = pd.read_csv(results_path)

allowed_combos = [
    "HQ",
    "A",
    "B",
    "D",
    "A+E",
    "A+F",
    "A+D+E+F",
    "A+B+C+D+E+F",
]
model_order = ["logistic", "xgb1", "xgb3", "mlp32"]

engineered_family = next((f for f in results["family"].unique() if str(f).startswith("engineered_")), "engineered_set_05")
engineered_set_id = engineered_family.replace("engineered_", "")

def fmt(mean, std):
    return f"{mean:.3f}+/-{std:.3f}"

def build_matrix(family_key, title):
    lines = [title, "| Combo | " + " | ".join(m.upper() for m in model_order) + " |", "|---|---:|---:|---:|---:|"]
    for combo in allowed_combos:
        row = []
        for model in model_order:
            sub = results[(results.family == family_key) & (results.reasoning_combo == combo) & (results.model_type == model)]
            if sub.empty:
                row.append("--")
            else:
                row.append(fmt(float(sub.iloc[0]['f0.5_mean']), float(sub.iloc[0]['f0.5_std'])))
        lines.append(f"| {combo} | " + " | ".join(row) + " |")
    return lines

def build_cv_full(family_key, title):
    header_cols = []
    for model in model_order:
        header_cols.extend([f"{model.upper()} CV", f"{model.upper()} Full"])
    lines = [title, "| Combo | " + " | ".join(header_cols) + " |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for combo in allowed_combos:
        row = []
        for model in model_order:
            sub = results[(results.family == family_key) & (results.reasoning_combo == combo) & (results.model_type == model)]
            if sub.empty:
                row.extend(["--", "--"])
            else:
                row.append(fmt(float(sub.iloc[0]['f0.5_mean']), float(sub.iloc[0]['f0.5_std'])))
                row.append(f"{float(sub.iloc[0]['full_train_f0.5']):.3f}")
        lines.append(f"| {combo} | " + " | ".join(row) + " |")
    return lines

lines = [
    "# Model Testing Report",
    f"Generated: {pd.Timestamp.utcnow().isoformat()}Z",
    "",
    "Model variants: logistic, xgb1 (stump), xgb3 (depth=3), mlp32 (1 hidden layer).",
    "",
]
lines += build_matrix("hq_mirror", "## HQ Mirror + Reasoning (rule layer)")
lines += [""]
lines += build_matrix(engineered_family, f"## Engineered {engineered_set_id} + Reasoning (no rule layer)")
lines += ["", "## CV vs Full-Train (per combo, same model)"]
lines += build_cv_full("hq_mirror", "### HQ Mirror + Reasoning")
lines += [""]
lines += build_cv_full(engineered_family, f"### Engineered {engineered_set_id} + Reasoning")

report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Rewrote report: {report_path}")
