# features/extract_llm.py
"""Phase 3: LLM feature extraction from anonymised_prose.
Extracts 7 structured fields → 9 classifier features.
Cache to features/llm_features_cache.json. One-time cost (~$4 with Haiku).
"""
import json
import os
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_FILE = Path("features/llm_features_cache.json")

EXTRACTION_PROMPT = """Given this founder profile, respond ONLY with valid JSON. No preamble, no explanation, no markdown fences.

{
  "prior_founding_attempt": true or false,
  "domain_expertise_depth": integer 1-5,
  "highest_seniority_reached": "founder"|"C-level"|"VP"|"senior-IC"|"IC"|"junior",
  "evidence_of_prior_exit": true or false,
  "career_narrative_type": "builder"|"climber"|"academic"|"hybrid"|"unclear",
  "domain_focus_consistency": integer 1-5,
  "conviction_indicator": integer 1-5
}

Field definitions:
- prior_founding_attempt: true if the profile shows founding or co-founding ANY company before the current startup.
- domain_expertise_depth: 1=no visible domain expertise, 5=deep specialized expertise in one technical/business area.
- highest_seniority_reached: the most senior role BEFORE the current founding. "founder" if they founded another company previously.
- evidence_of_prior_exit: true ONLY if profile explicitly mentions a prior IPO, acquisition, or company sale.
- career_narrative_type: "builder"=founded/built products, "climber"=rose through corporate ranks, "academic"=research/PhD-track, "hybrid"=mix, "unclear"=insufficient info.
- domain_focus_consistency: 1=scattered across many unrelated domains, 5=laser-focused in one domain throughout career.
- conviction_indicator: 1=low personal commitment signals, 5=high commitment (left stable career, early founding, multiple founding attempts, bootstrapped)."""


def extract_single(client, uuid, prose):
    """Extract features for one founder."""
    try:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": EXTRACTION_PROMPT + "\n\nProfile:\n" + str(prose)
            }]
        )
        text = r.content[0].text.strip()
        # Handle potential markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return uuid, json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON substring
        text = r.content[0].text
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            try:
                return uuid, json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        print(f"  JSON parse error for {uuid}: {text[:100]}")
        return uuid, None
    except anthropic.RateLimitError:
        time.sleep(5)
        return extract_single(client, uuid, prose)
    except Exception as e:
        print(f"  Error for {uuid}: {type(e).__name__}: {e}")
        return uuid, None


def run_extraction():
    """Run LLM extraction on all public rows, with checkpointing."""
    if CACHE_FILE.exists():
        cache = json.loads(CACHE_FILE.read_text())
        print(f"Loaded cache with {len(cache)} entries")
    else:
        cache = {}

    df = pd.read_csv("data/vcbench_final_public.csv")
    # Skip rows with valid (non-None) cache entries; re-extract failed ones
    remaining = [(row['founder_uuid'], row['anonymised_prose'])
                  for _, row in df.iterrows()
                  if cache.get(row['founder_uuid']) is None]

    valid_count = sum(1 for v in cache.values() if v is not None)
    print(f"Total rows: {len(df)}, Valid cached: {valid_count}, Remaining: {len(remaining)}")

    if not remaining:
        print("All rows already cached.")
        return cache

    client = anthropic.Anthropic()

    # Process in batches with concurrent workers
    workers = 10
    checkpoint_interval = 50  # save every 50 batches
    for i in range(0, len(remaining), workers):
        batch = remaining[i:i + workers]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(extract_single, client, uuid, prose): uuid
                for uuid, prose in batch
            }
            for future in as_completed(futures):
                uuid, result = future.result()
                cache[uuid] = result

        done = len(cache)
        batch_num = i // workers
        # Checkpoint periodically
        if batch_num % checkpoint_interval == 0 or done == len(df):
            CACHE_FILE.write_text(json.dumps(cache))
            print(f"  Checkpoint: {done}/{len(df)} ({done / len(df) * 100:.1f}%)")

    # Final save
    CACHE_FILE.write_text(json.dumps(cache))
    null_count = sum(1 for v in cache.values() if v is None)
    print(f"\nExtraction complete. {len(cache)} entries cached. {null_count} nulls.")
    return cache


def add_llm_features(df):
    """Add 9 LLM features to dataframe from cache."""
    if not CACHE_FILE.exists():
        raise FileNotFoundError(f"Cache not found at {CACHE_FILE}. Run extraction first.")

    cache = json.loads(CACHE_FILE.read_text())

    records = []
    for uuid in df['founder_uuid']:
        feat = cache.get(uuid) or {}
        seniority = feat.get("highest_seniority_reached", "junior")
        narrative = feat.get("career_narrative_type", "unclear")
        records.append({
            "founder_uuid": uuid,
            "llm_prior_founding": int(bool(feat.get("prior_founding_attempt", False))),
            "llm_domain_expertise": int(feat.get("domain_expertise_depth", 3)),
            "llm_prior_exit": int(bool(feat.get("evidence_of_prior_exit", False))),
            "llm_narrative_builder": int(narrative == "builder"),
            "llm_narrative_climber": int(narrative == "climber"),
            "llm_seniority_founder": int(seniority == "founder"),
            "llm_seniority_clevel": int(seniority == "C-level"),
            "llm_domain_focus": int(feat.get("domain_focus_consistency", 3)),
            "llm_conviction": int(feat.get("conviction_indicator", 3)),
        })

    return df.merge(pd.DataFrame(records), on="founder_uuid", how="left")


if __name__ == "__main__":
    run_extraction()
