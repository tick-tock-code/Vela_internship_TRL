# evaluate.py
# LOCKED — do not modify. Changing this file alters the evaluation metric and
# makes results incomparable with reported numbers.
import numpy as np
from sklearn.metrics import precision_score, recall_score, fbeta_score


def evaluate(y_true, y_prob, threshold=0.5):
    """Apply threshold and compute F₀.₅, precision, recall, and positive rate.

    Parameters
    ----------
    y_true : list[int]
        Ground truth binary labels (0 or 1).
    y_prob : list[float]
        Predicted probabilities for class 1.
    threshold : float
        Decision threshold (default 0.5).

    Returns
    -------
    dict
        Keys: f05, precision, recall, positive_rate, n_predicted_positive, threshold.
    """
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    return {
        "f05": round(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "positive_rate": round(float(y_pred.mean()), 4),
        "n_predicted_positive": int(y_pred.sum()),
        "threshold": threshold,
    }

def sweep_thresholds(y_true, y_prob, steps=50):
    """Sweep thresholds from 0.30 to 0.95 and return results sorted by F₀.₅ descending."""
    results = []
    for t in np.linspace(0.3, 0.95, steps):
        results.append(evaluate(y_true, y_prob, threshold=t))
    return sorted(results, key=lambda x: x["f05"], reverse=True)

def best_threshold(y_true, y_prob):
    """Return the threshold configuration that maximises F₀.₅."""
    return sweep_thresholds(y_true, y_prob)[0]

if __name__ == "__main__":
    # Smoke test
    import numpy as np
    y_true = [0]*91 + [1]*9
    y_prob = [0.1]*85 + [0.8]*6 + [0.9]*9
    result = best_threshold(y_true, y_prob)
    print("Smoke test passed:", result)
