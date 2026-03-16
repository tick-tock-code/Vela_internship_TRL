"""Train PyTorch logistic regression from a Parquet feature file.

This script is TRL-free and intended to run in torch_env2.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from datetime import datetime
from typing import Any, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


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
    epochs: int,
    lr: float,
    device_choice: str,
) -> tuple[np.ndarray, np.ndarray, torch.nn.Linear]:
    if device_choice == "cpu":
        device = torch.device("cpu")
    elif device_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        train_logits = model(Xtr).detach().cpu().numpy().reshape(-1)
        test_logits = model(Xte).detach().cpu().numpy().reshape(-1)
    return train_logits, test_logits, model


def _report_metrics(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
) -> dict[str, float]:
    return _threshold_and_metrics(
        y_train=y_train,
        train_scores=train_scores,
        y_test=y_test,
        test_scores=test_scores,
        optimize_for="f0.5",
        t_min=0.01,
        t_max=0.99,
        t_step=0.01,
    )


def _threshold_and_metrics(
    y_train: np.ndarray,
    train_scores: np.ndarray,
    y_test: np.ndarray,
    test_scores: np.ndarray,
    optimize_for: str,
    t_min: float,
    t_max: float,
    t_step: float,
) -> dict[str, float]:
    best_t = 0.5
    best_score = -1.0
    best_rec = -1.0
    best_prec = -1.0
    for t in np.arange(t_min, t_max + 1e-9, t_step):
        y_hat = (train_scores >= t).astype(int)
        rec = recall_score(y_train, y_hat, zero_division=0.0)  # type: ignore[arg-type]
        prec = precision_score(y_train, y_hat, zero_division=0.0)  # type: ignore[arg-type]
        f05 = fbeta_score(y_train, y_hat, beta=0.5, zero_division=0.0)  # type: ignore[arg-type]
        if optimize_for == "recall":
            score = rec
        elif optimize_for == "precision":
            score = prec
        elif optimize_for == "f1":
            score = fbeta_score(y_train, y_hat, beta=1.0, zero_division=0.0)  # type: ignore[arg-type]
        else:
            score = f05
        if score > best_score or (score == best_score and rec > best_rec) or (
            score == best_score and rec == best_rec and prec > best_prec
        ):
            best_score = score
            best_rec = rec
            best_prec = prec
            best_t = float(t)

    y_pred = (test_scores >= best_t).astype(int)
    # Confusion components
    tp = float(np.sum((y_test == 1) & (y_pred == 1)))
    fn = float(np.sum((y_test == 1) & (y_pred == 0)))
    tn = float(np.sum((y_test == 0) & (y_pred == 0)))
    fp = float(np.sum((y_test == 0) & (y_pred == 1)))
    fnt = fn / (tp + fn) if (tp + fn) > 0 else 0.0

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
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "fnr": fnt,
    }
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train from Parquet features.")
    p.add_argument(
        "--input_parquet",
        default=str(Path(__file__).parent / "features_full.parquet"),
    )
    p.add_argument(
        "--log_path",
        default="",
        help="Optional log file path. Defaults to training_log_<timestamp>.txt",
    )
    p.add_argument("--label_column", default="success")
    p.add_argument("--test_size", type=float, default=0.40)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--optimize_for",
        choices=["f0.5", "recall", "precision", "f1"],
        default="f0.5",
        help="Threshold optimization target.",
    )
    p.add_argument("--threshold_min", type=float, default=1e-6)
    p.add_argument("--threshold_max", type=float, default=0.99)
    p.add_argument("--threshold_step", type=float, default=0.005)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use for training. Use cpu to avoid CUDA issues.",
    )
    p.add_argument(
        "--self_test",
        action="store_true",
        help="Run internal sanity tests and exit.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        run_self_tests()
        return
    df = pd.read_parquet(args.input_parquet)

    y = df[args.label_column].to_numpy().astype(int)
    feature_cols = [c for c in df.columns if c not in ("founder_uuid", args.label_column)]
    X = df[feature_cols].to_numpy(dtype=float)

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_scores, test_scores, model = _train_pytorch(
        X_train, y_train, X_test, args.epochs, args.lr, args.device
    )
    metrics = _threshold_and_metrics(
        y_train=y_train,
        train_scores=train_scores,
        y_test=y_test,
        test_scores=test_scores,
        optimize_for=args.optimize_for,
        t_min=args.threshold_min,
        t_max=args.threshold_max,
        t_step=args.threshold_step,
    )

    log_lines: List[str] = []
    def _log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    _log(f"\n  Features used: {', '.join(feature_cols)}")
    _log(
        f"\n  [Train from Parquet]   {len(feature_cols)} features, threshold={metrics['threshold']:.2f}"
    )
    coef = model.weight.detach().cpu().numpy().reshape(-1)
    ranked = sorted(zip(feature_cols, coef), key=lambda x: abs(x[1]), reverse=True)
    for name, c in ranked:
        sign = "+" if c >= 0 else "-"
        _log(f"    {sign}{abs(c):.3f}  {name}")

    acc = float(np.mean((test_scores >= metrics["threshold"]).astype(int) == y_test))
    _log(
        f"\n  ROC-AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
        f"Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  "
        f"F0.5={metrics['f0.5']:.3f}  Acc={acc:.3f}  FNR={metrics['fnr']:.3f}  "
        f"TP={int(metrics['tp'])}  FN={int(metrics['fn'])}"
    )

    if args.log_path:
        log_path = Path(args.log_path)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(__file__).parent / f"training_log_{ts}.txt"

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n  Training log saved to: {log_path}")


def run_self_tests() -> None:
    # Test 1: recall optimization should pick low threshold to get recall=1
    y_train = np.array([1, 1, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.4, 0.2, 0.1])
    m = _threshold_and_metrics(
        y_train=y_train,
        train_scores=scores,
        y_test=y_train,
        test_scores=scores,
        optimize_for="recall",
        t_min=0.0,
        t_max=1.0,
        t_step=0.1,
    )
    assert m["recall"] >= 1.0 - 1e-9, "Recall optimization failed"

    # Test 2: f0.5 optimization returns a valid threshold in range
    m = _threshold_and_metrics(
        y_train=y_train,
        train_scores=scores,
        y_test=y_train,
        test_scores=scores,
        optimize_for="f0.5",
        t_min=0.0,
        t_max=1.0,
        t_step=0.1,
    )
    assert 0.0 <= m["threshold"] <= 1.0, "Threshold out of range"

    # Test 3: confusion consistency
    tp = int(m["tp"])
    fn = int(m["fn"])
    assert tp + fn == int(np.sum(y_train)), "TP+FN != positives"

    print("Self-tests passed.")


if __name__ == "__main__":
    main()
