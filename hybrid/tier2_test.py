"""Hybrid search test with the most common real choices of the app's users (genre and party type pairs), for several boost weights.
Each profile is one pair from the Tier 2 table, for example Hip-Hop + Bar. Every profile is run on the broad queries and, separately,
on queries that contain a time word, which switch on the date filter.
Usage: python hybrid/tier2_test.py --profiles data/tier2_profiles.csv [--top 12] [--weights 0.1 0.2 0.3 0.5] [--today 2026-01-16]
The profiles file has the columns music_genre, party_type, users_count."""

from __future__ import annotations

import argparse
import csv
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
from profile_boost import MODEL, load_features  # noqa: E402
from try_search import run  # noqa: E402

BROAD_QUERIES = ["party", "live music", "night out", "dance night", "concert", "bar with music", "dinner and drinks", "DJ night",
                 "something fun to do", "chill evening with music"]
TIME_QUERIES = ["party tonight", "live music this weekend", "something fun tomorrow"]
MAIN_WEIGHT = 0.2


def read_profiles(path: str, top: int) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    rows.sort(key=lambda r: -int(r["users_count"]))
    return [{"name": f"{r['music_genre']} + {r['party_type']}", "music": [r["music_genre"]], "types": [r["party_type"]],
             "users": int(r["users_count"])} for r in rows[:top]]


def key(row: dict) -> tuple:
    return row["name"], row["date"]


def compare(result: dict, profile: dict) -> dict:
    plain = {key(r) for r in result["plain"]}
    promoted = [r for r in result["hybrid"] if key(r) not in plain]
    fits = [r for r in promoted if set(profile["music"]) & set(r["tags"]) or set(profile["types"]) & set(r["types"])]
    return {"changed": len(promoted), "deepest_promoted_rank": max([r["plain_rank"] for r in promoted], default=0),
            "promoted_fitting_profile": len(fits), "promoted": promoted,
            "dropped": [r for r in result["plain"] if key(r) not in {key(h) for h in result["hybrid"]}], "notes": result["notes"]}


def summarise(rows: list[dict]) -> dict:
    changed = [r["changed"] for r in rows]
    promoted = sum(changed)
    return {"lists": len(rows), "share_changed": float(np.mean([c > 0 for c in changed])), "mean_changed_slots": float(np.mean(changed)),
            "deepest_promoted_rank": max(r["deepest_promoted_rank"] for r in rows),
            "share_promoted_fitting_profile": (sum(r["promoted_fitting_profile"] for r in rows) / promoted) if promoted else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default=str(ROOT / "data" / "tier2_profiles.csv"))
    parser.add_argument("--top", type=int, default=12, help="number of most common genre and party-type pairs to use")
    parser.add_argument("--weights", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--today", default="2026-01-16", help="the day 'tonight' and 'tomorrow' refer to")
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--out", default=str(ROOT / "data" / "search_runs" / "tier2_test.json"))
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
    queries = BROAD_QUERIES + TIME_QUERIES
    embedded = np.array([d.embedding for d in OpenAI().embeddings.create(model=MODEL, input=queries).data], dtype=np.float32)
    embedded /= np.linalg.norm(embedded, axis=1, keepdims=True)
    scores = {q: vectors @ v for q, v in zip(queries, embedded)}
    run_tier2(args, features, scores)


def run_tier2(args, features: dict, scores: dict) -> None:
    profiles = read_profiles(args.profiles, args.top)
    today = dt.date.fromisoformat(args.today)
    report = {"profiles": [{k: p[k] for k in ("name", "music", "types", "users")} for p in profiles], "weights": args.weights,
              "broad_queries": BROAD_QUERIES, "time_queries": TIME_QUERIES, "today": args.today, "broad": {}, "time": {}, "per_profile": {}, "examples": []}
    for weight in args.weights:
        broad, timed = [], []
        for profile in profiles:
            for query in BROAD_QUERIES:
                result = run(query, profile, features, scores[query], weight, today, use_dates=False, use_guard=True)
                row = {"profile": profile["name"], "query": query, **compare(result, profile)}
                broad.append(row)
                if weight == MAIN_WEIGHT and row["changed"]:
                    report["examples"].append({k: row[k] for k in ("profile", "query", "changed", "promoted", "dropped", "notes")})
            for query in TIME_QUERIES:
                # the date filter alone (weight 0) is the reference, so that only the effect of the profile boost is counted
                filtered = run(query, profile, features, scores[query], 0.0, today, use_dates=True, use_guard=True)
                result = run(query, profile, features, scores[query], weight, today, use_dates=True, use_guard=True)
                result["plain"] = filtered["hybrid"]
                row = {"profile": profile["name"], "query": query, **compare(result, profile)}
                timed.append(row)
        report["broad"][str(weight)] = summarise(broad)
        report["time"][str(weight)] = summarise(timed)
        report["per_profile"][str(weight)] = {p["name"]: summarise([r for r in broad if r["profile"] == p["name"]]) for p in profiles}
    print(f"profiles: {', '.join(p['name'] for p in profiles)}")
    print(f"broad queries ({len(BROAD_QUERIES)}), plain ada-002 against hybrid:")
    for w in args.weights:
        s = report["broad"][str(w)]
        fit = "n/a" if s["share_promoted_fitting_profile"] is None else f"{s['share_promoted_fitting_profile']:.0%}"
        print(f"  weight {w}: top 10 changed in {s['share_changed']:.0%} of {s['lists']} lists, {s['mean_changed_slots']:.2f} slots on average, "
              f"deepest promoted event was at plain rank {s['deepest_promoted_rank']}, {fit} of the promoted events carry the profile's genre or party type")
    print(f"queries with a time word ({', '.join(TIME_QUERIES)}), 'today' = {args.today}:")
    for w in args.weights:
        s = report["time"][str(w)]
        print(f"  weight {w}: the profile changes the date-filtered top 10 in {s['share_changed']:.0%} of {s['lists']} lists, {s['mean_changed_slots']:.2f} slots on average")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"saved {args.out} ({len(report['examples'])} changed lists at weight {MAIN_WEIGHT} for the human check)")


if __name__ == "__main__":
    main()
