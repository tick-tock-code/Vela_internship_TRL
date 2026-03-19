# baselines/zero_shot_baseline.py
"""
Zero-shot Claude Sonnet baseline using anonymised_prose only.
This establishes the ~25% F0.5 baseline that all structured approaches must beat.
Run once, cache predictions, never re-run.
"""
import anthropic
import pandas as pd
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluate import evaluate, sweep_thresholds

CACHE_FILE = Path(__file__).parent / "zero_shot_predictions.json"

SYSTEM_PROMPT = """You are a venture capital analyst predicting startup success.
You will be given a founder profile. Respond with ONLY a JSON object:
{"probability": <float between 0 and 1>, "reasoning": "<one sentence>"}
A probability > 0.5 means you predict success (IPO, major acquisition, or high-tier funding).
Be calibrated: only ~9% of founders in this dataset are successful."""


def predict_one(client, prose: str) -> float:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=100,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prose}],
    )
    try:
        return json.loads(response.content[0].text)["probability"]
    except Exception:
        return 0.5


def run_baseline():
    val = pd.read_csv(Path(__file__).parent.parent / "data" / "public_val.csv")

    if CACHE_FILE.exists():
        print("Loading cached predictions...")
        cache = json.loads(CACHE_FILE.read_text())
        probs = cache["probs"]
    else:
        client = anthropic.Anthropic()
        probs = []
        for i, (_, row) in enumerate(val.iterrows()):
            prob = predict_one(client, row["anonymised_prose"])
            probs.append(prob)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(val)} complete...")
                # Save partial progress
                CACHE_FILE.write_text(json.dumps({"probs": probs, "complete": False}))
            # Small delay to avoid rate limits
            time.sleep(0.1)
        CACHE_FILE.write_text(json.dumps({"probs": probs, "complete": True}))
        print(f"  {len(val)}/{len(val)} complete. Cached to {CACHE_FILE}")

    results = sweep_thresholds(val["success"].tolist(), probs)
    print("\n=== Zero-shot baseline (top 5 thresholds) ===")
    for r in results[:5]:
        print(r)
    print(f"\nBest F0.5: {results[0]['f05']} at threshold {results[0]['threshold']:.3f}")
    print(f"Precision: {results[0]['precision']}, Recall: {results[0]['recall']}")
    return results[0]


if __name__ == "__main__":
    run_baseline()
