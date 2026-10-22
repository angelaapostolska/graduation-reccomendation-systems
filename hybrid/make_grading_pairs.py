"""Builds the pairs for the human check: for broad queries and different user profiles, plain ada-002 against the hybrid result.
Only pairs where the two top 10 differ are kept. Usage: python hybrid/make_grading_pairs.py [--weight 0.2] [--out data/search_runs/hybrid_pairs.json]"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_boost import MODEL, PERSONAS, load_features  # noqa: E402
from try_search import run  # noqa: E402

# broad queries, where many events tie and a profile could decide
BROAD_QUERIES = ["party", "live music", "night out", "dance night", "concert", "bar with music", "dinner and drinks", "DJ night",
                 "something fun to do", "chill evening with music"]
GRADING_PERSONAS = ["house and techno fan", "hip-hop and R&B fan", "pop and balkan fan", "rock and indie fan", "jazz and soul fan", "classical and folk fan"]


def build_pairs(scores_per_query: list[np.ndarray], features: dict, weight: float) -> dict:
    pairs, unchanged = [], 0
    for persona in GRADING_PERSONAS:
        profile = PERSONAS[persona]
        for query, scores in zip(BROAD_QUERIES, scores_per_query):
            result = run(query, profile, features, scores, weight, dt.date(2026, 1, 16), use_dates=False, use_guard=True)
            plain, hybrid = {(r["name"], r["date"]) for r in result["plain"]}, {(r["name"], r["date"]) for r in result["hybrid"]}
            changed = len(hybrid - plain)
            if changed == 0:
                unchanged += 1
                continue
            pairs.append({"persona": persona, "music": profile["music"], "types": profile["types"], "query": query, "notes": result["notes"],
                          "changed_slots": changed, "plain": result["plain"], "hybrid": result["hybrid"]})
    return {"weight": weight, "queries": BROAD_QUERIES, "personas": GRADING_PERSONAS, "pairs": pairs, "unchanged_pairs": unchanged,
            "all_pairs": len(GRADING_PERSONAS) * len(BROAD_QUERIES)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--weight", type=float, default=0.2)
    parser.add_argument("--out", default=str(ROOT / "data" / "search_runs" / "hybrid_pairs.json"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in .env")
    from openai import OpenAI

    features = load_features(args.tables)
    cities = pd.read_csv(f"{args.tables}/cities.csv").set_index("city_id")["name"]
    venues = pd.read_csv(f"{args.tables}/venues.csv").set_index("venue_id")
    features["events"]["city"] = features["events"]["venue_id"].map(venues["city_id"]).map(cities)
    features["events"]["venue"] = features["events"]["venue_id"].map(venues["name"])
    vectors = np.load(ROOT / "data" / "graph" / "event_text.npy").astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert len(vectors) == len(features["events"]), "events.csv and the saved vectors do not match"
    queries = np.array([d.embedding for d in OpenAI().embeddings.create(model=MODEL, input=BROAD_QUERIES).data], dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    report = build_pairs([vectors @ q for q in queries], features, args.weight)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"weight {args.weight}: {len(report['pairs'])} of {report['all_pairs']} (profile, query) pairs differ from plain ada-002; {report['unchanged_pairs']} are identical")
    for query in BROAD_QUERIES:
        changed = [p["changed_slots"] for p in report["pairs"] if p["query"] == query]
        print(f'  "{query}": {len(changed)} of {len(GRADING_PERSONAS)} profiles differ, changed slots {changed}')
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
