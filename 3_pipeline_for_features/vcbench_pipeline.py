"""VCBench in-depth pipeline (human + optional LLM features).

Pipeline steps:
1. Load VCBench data (sample or full).
2. Extract baseline features (15) from example script.
3. Extract custom features from registry.
4. Optionally add LLM-derived features.
5. Save full feature dataset (founder_uuid, label, features) to Parquet.
6. Train PyTorch logistic regression and evaluate metrics.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from think_reason_learn.core.llms import OpenAIChoice
from think_reason_learn.datasets import (
    VCBENCH_HELPERS,
    VCBENCH_SCHEMA,
    load_vcbench,
)
from think_reason_learn.features import FeatureEvaluator, FeatureGenerator

from feature_registry import FEATURE_REGISTRY, FEATURE_SETS


def _load_base_feature_extractor(script_path: Path):
    spec = importlib.util.spec_from_file_location("vcbench_base_features", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load base feature script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[assignment]
    if not hasattr(module, "_extract_human_features"):
        raise AttributeError(
            f"Base script missing _extract_human_features: {script_path}"
        )
    return module._extract_human_features  # type: ignore[attr-defined]


def _custom_feature_df(
    records: Iterable[dict[str, Any]], feature_names: list[str]
) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {}
        for name in feature_names:
            spec = FEATURE_REGISTRY.get(name)
            if spec is None:
                raise KeyError(f"Unknown feature: {name}")
            row[name] = spec.func(r)
        rows.append(row)
    return pd.DataFrame(rows)


def _standardize_continuous(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cont_names = [
        n
        for n in feature_names
        if FEATURE_REGISTRY.get(n, None) is not None
        and FEATURE_REGISTRY[n].feature_type == "continuous"
    ]
    if not cont_names:
        return train_df, test_df
    train = train_df.copy()
    test = test_df.copy()
    for name in cont_names:
        mean = float(train[name].mean())
        std = float(train[name].std())
        if std <= 0:
            std = 1.0
        train[name] = (train[name] - mean) / std
        test[name] = (test[name] - mean) / std
    return train, test


def _precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, pct: float) -> float:
    n = len(y_true)
    k = max(1, int(math.ceil(pct * n)))
    order = np.argsort(y_scores)[::-1]
    top_k = order[:k]
    return float(np.sum(y_true[top_k]) / k)


def _train_pytorch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
) -> tuple[np.ndarray, np.ndarray, "torch.nn.Linear"]:
    import torch
    device = torch.device("cpu")
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=device)
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)

    model = torch.nn.Linear(Xtr.shape[1], 1, bias=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(Xtr)
        loss = loss_fn(logits, ytr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_logits = model(Xtr).cpu().numpy().reshape(-1)
        test_logits = model(Xte).cpu().numpy().reshape(-1)
    return train_logits, test_logits, model


def _report_metrics(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float]:
    # Threshold tuning on training set for F0.5
    best_t = 0.5
    best_f = 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        f = fbeta_score(
            y_train,
            (train_scores >= t).astype(int),
            beta=0.5,
            zero_division=0.0,  # type: ignore[arg-type]
        )
        if f > best_f:
            best_f, best_t = f, float(t)

    y_pred = (test_scores >= best_t).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y_test, test_scores),
        "pr_auc": average_precision_score(y_test, test_scores),
        "precision": precision_score(y_test, y_pred, zero_division=0.0),  # type: ignore[arg-type]
        "recall": recall_score(y_test, y_pred, zero_division=0.0),  # type: ignore[arg-type]
        "f0.5": fbeta_score(y_test, y_pred, beta=0.5, zero_division=0.0),  # type: ignore[arg-type]
        "precision@1%": _precision_at_k(y_test, test_scores, 0.01),
        "precision@5%": _precision_at_k(y_test, test_scores, 0.05),
        "precision@10%": _precision_at_k(y_test, test_scores, 0.10),
        "threshold": best_t,
    }
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VCBench in-depth pipeline.")
    p.add_argument("--dataset", choices=["sample", "full"], default="sample")
    p.add_argument("--input_csv", default="", help="Optional override CSV path.")
    p.add_argument("--label_column", default="success")
    p.add_argument("--test_size", type=float, default=0.20)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--feature_set",
        choices=sorted(FEATURE_SETS.keys()),
        default="base_plus_custom",
    )
    p.add_argument(
        "--features",
        default="",
        help="Optional comma-separated custom feature names (overrides feature_set)",
    )
    p.add_argument(
        "--feature_config",
        default=str(Path(__file__).parent / "features.json"),
        help="Optional JSON file with {\"features\": [...]} to select custom features.",
    )
    p.add_argument(
        "--extract_only",
        action="store_true",
        help="Extract features and save Parquet, then exit before training.",
    )
    p.add_argument(
        "--mode",
        choices=["human", "llm", "hybrid"],
        default="human",
        help="human=base+custom, llm=LLM only, hybrid=base+custom+LLM",
    )
    p.add_argument(
        "--llm_features",
        action="store_true",
        help="(Deprecated) Use --mode llm or --mode hybrid instead.",
    )
    p.add_argument("--llm_model", default="gpt-4.1-nano")
    p.add_argument("--llm_n_features", type=int, default=8)
    p.add_argument(
        "--base_script",
        default=str(
            Path(
                r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
            )
            / "2_Human_features_running_example_script"
            / "vcbench_lambda_features_minimal.py"
        ),
        help="Path to base feature script with _extract_human_features.",
    )
    p.add_argument(
        "--output_parquet",
        default="",
        help="Optional output parquet path for extracted features.",
    )
    return p.parse_args()


def _resolve_input_csv(dataset: str, override: str) -> str:
    if override:
        return override
    base = Path(
        r"C:\Users\joelb\OneDrive\Vela_partnerships_project\Project_folder"
    ) / "VCBench-Starter-Kit"
    if dataset == "sample":
        return str(base / "vcbench_final_public_sample100.csv")
    return str(base / "vcbench_final_public.csv")


def main() -> None:
    args = _parse_args()
    rs = args.random_state
    input_csv = _resolve_input_csv(args.dataset, args.input_csv)

    print(f"\n{'=' * 60}")
    print("  VCBench In-Depth Pipeline")
    print(f"{'=' * 60}\n")
    print(f"  Dataset: {args.dataset}")
    print(f"  Input CSV: {input_csv}")
    print(f"  Mode: {args.mode}\n")

    records, labels = load_vcbench(
        input_csv,
        args.label_column,
        0,
        rs,
    )

    idx = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=labels,
        random_state=rs,
    )
    train_recs = [records[i] for i in train_idx]
    test_recs = [records[i] for i in test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    base_script = Path(args.base_script)
    extract_base = _load_base_feature_extractor(base_script)
    base_all = pd.DataFrame([extract_base(r) for r in records])
    base_train = pd.DataFrame([extract_base(r) for r in train_recs])
    base_test = pd.DataFrame([extract_base(r) for r in test_recs])
    base_feature_names = list(base_train.columns)

    selected_features: list[str] = []
    cfg_path = Path(args.feature_config) if args.feature_config else None
    if cfg_path is not None and cfg_path.exists():
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        selected_features = [f for f in data.get("features", []) if isinstance(f, str)]

    if selected_features:
        base_selected = [f for f in selected_features if f in base_feature_names]
        custom_features = [f for f in selected_features if f in FEATURE_REGISTRY]
        unknown = [
            f
            for f in selected_features
            if f not in base_feature_names and f not in FEATURE_REGISTRY
        ]
        if unknown:
            print(f"  WARNING: Unknown features in config: {', '.join(unknown)}")
    elif args.features.strip():
        base_selected = base_feature_names
        custom_features = [f.strip() for f in args.features.split(",") if f.strip()]
    else:
        base_selected = base_feature_names
        custom_features = FEATURE_SETS[args.feature_set]

    # Apply baseline selection if provided
    if selected_features:
        base_all = base_all[base_selected]
        base_train = base_train[base_selected]
        base_test = base_test[base_selected]
        base_feature_names = list(base_train.columns)

    custom_all = (
        _custom_feature_df(records, custom_features)
        if custom_features
        else pd.DataFrame(index=range(len(records)))
    )
    custom_train = (
        _custom_feature_df(train_recs, custom_features)
        if custom_features
        else pd.DataFrame(index=range(len(train_recs)))
    )
    custom_test = (
        _custom_feature_df(test_recs, custom_features)
        if custom_features
        else pd.DataFrame(index=range(len(test_recs)))
    )

    llm_feature_names: list[str] = []
    llm_all = pd.DataFrame(index=range(len(records)))
    llm_train = pd.DataFrame(index=range(len(train_recs)))
    llm_test = pd.DataFrame(index=range(len(test_recs)))

    use_llm = args.mode in ("llm", "hybrid") or args.llm_features
    if use_llm:
        # NOTE: LLM generation is async; we run a simple synchronous wrapper.
        # This keeps the option available without forcing LLM usage.
        async def _gen_llm():
            generator = FeatureGenerator(
                schema=VCBENCH_SCHEMA,
                helpers=VCBENCH_HELPERS,
                llm_priority=[OpenAIChoice(model=args.llm_model)],
                temperature=0.7,
            )
            rules = await generator.generate(
                samples=train_recs,
                labels=y_train.tolist(),
                n_rules=args.llm_n_features,
                n_samples=min(60, len(train_recs)),
            )
            if not rules:
                raise RuntimeError("No LLM rules generated. Check API key.")
            evaluator = FeatureEvaluator(rules=rules, helpers=VCBENCH_HELPERS)
            names = [r.name for r in rules]
            return (
                evaluator.evaluate_df(records),
                evaluator.evaluate_df(train_recs),
                evaluator.evaluate_df(test_recs),
                names,
            )

        import asyncio

        llm_all, llm_train, llm_test, llm_feature_names = asyncio.run(_gen_llm())

    if args.mode == "human":
        full_all = pd.concat([base_all, custom_all], axis=1)
        full_train = pd.concat([base_train, custom_train], axis=1)
        full_test = pd.concat([base_test, custom_test], axis=1)
        feature_names = base_feature_names + custom_features
        mode_label = "Human Only"
    elif args.mode == "llm":
        full_all = llm_all
        full_train = llm_train
        full_test = llm_test
        feature_names = llm_feature_names
        mode_label = "LLM Only"
    else:
        full_all = pd.concat([base_all, custom_all, llm_all], axis=1)
        full_train = pd.concat([base_train, custom_train, llm_train], axis=1)
        full_test = pd.concat([base_test, custom_test, llm_test], axis=1)
        feature_names = base_feature_names + custom_features + llm_feature_names
        mode_label = "Hybrid"

    # Standardize continuous custom features only (training stats)
    if args.mode in ("human", "hybrid"):
        full_train, full_test = _standardize_continuous(
            full_train, full_test, custom_features
        )

    # Save feature dataset (founder_uuid, label, features)
    founder_ids = [r.get("founder_uuid") for r in records]
    save_df = pd.concat(
        [
            pd.Series(founder_ids, name="founder_uuid"),
            pd.Series(labels, name="success"),
            full_all,
        ],
        axis=1,
    )
    if args.output_parquet:
        out_path = Path(args.output_parquet)
    else:
        out_path = Path(__file__).parent / "features_full.parquet"
    save_df.to_parquet(out_path, index=False)
    print(f"  Saved features to: {out_path}")

    if args.extract_only:
        print("  extract_only enabled; skipping training.")
        return

    # Placeholder for future multiple training loops over feature subsets.
    # TODO: add loop over named feature sets and aggregate metrics.

    X_train = full_train.values.astype(float)
    X_test = full_test.values.astype(float)

    train_scores, test_scores, model = _train_pytorch(
        X_train, y_train, X_test, y_test, args.epochs, args.lr
    )

    metrics = _report_metrics(y_train, train_scores, y_test, test_scores)

    # Output format
    print(f"\n  Features used: {', '.join(feature_names)}")
    print(
        f"\n  [{mode_label}]   {len(feature_names)} features, threshold={metrics['threshold']:.2f}"
    )
    coef = model.weight.detach().cpu().numpy().reshape(-1)
    ranked = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)
    for name, c in ranked:
        sign = "+" if c >= 0 else "-"
        print(f"    {sign}{abs(c):.3f}  {name}")

    acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y_test))
    print(
        f"\n  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
        f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
        f"F0.5={metrics['f0.5']:.3f}  Acc={acc:.3f}"
    )


if __name__ == "__main__":
    main()
