"""Stage 2 of the hybrid idea: rescales the ada-002 scores per query, adds a small boost for events that match a user profile,
and reports how many of the top 10 results change for each boost weight.
Usage: python hybrid/profile_boost.py [--users path/to/users_with_activity.csv] [--out data/search_runs/profile_boost.json]"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "search"))
from testing_query import QUERIES  # noqa: E402

MODEL = "text-embedding-ada-002"
POOL = 50
TOP = 10
WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.3)
PERFORMER_ONLY_WEIGHT = 0.5
# hand-made profiles in the shape of the real onboarding choices: music genres and party types
PERSONAS = {
    "house and techno fan": {"music": ["House", "Techno", "Electronic"], "types": ["Dj Set", "Disco", "Bar"]},
    "hip-hop and R&B fan": {"music": ["Hip-Hop", "R&B", "Rap", "Trap"], "types": ["Bar", "Dj Set"]},
    "pop and balkan fan": {"music": ["Pop", "Balkan Party", "Ex-Yu Hits", "Macedonian"], "types": ["Disco", "Bar"]},
    "rock and indie fan": {"music": ["Rock", "Indie", "Alternative"], "types": ["Live Music", "Bar", "Concert"]},
    "jazz and soul fan": {"music": ["Jazz", "Soul", "Blues", "Funk"], "types": ["Live Music", "Restaurant", "Caffe Bar"]},
    "classical and folk fan": {"music": ["Classical", "Folk", "Opera"], "types": ["Concert", "Restaurant"]},
    "all-round party goer": {"music": ["Pop", "House", "Hip-Hop", "Techno"], "types": ["Disco", "Dj Set", "Festival"]},
    "no preferences (must change nothing)": {"music": [], "types": []},
}


def load_features(tables: str) -> dict:
    events = pd.read_csv(f"{tables}/events.csv")
    genre_name = pd.read_csv(f"{tables}/music_genres.csv").set_index("music_genre_id")["name"]
    type_name = pd.read_csv(f"{tables}/event_types.csv").set_index("event_type_id")["name"]
    index = {e: i for i, e in enumerate(events["event_id"])}
    tag = defaultdict(set)
    for r in pd.read_csv(f"{tables}/event_music_genres.csv").itertuples():
        tag[r.event_id].add(genre_name[r.music_genre_id])
    performer_genres = defaultdict(set)
    for r in pd.read_csv(f"{tables}/performer_music_genres.csv").itertuples():
        performer_genres[r.performer_id].add(genre_name[r.music_genre_id])
    performer_only = defaultdict(set)
    for r in pd.read_csv(f"{tables}/event_performers.csv").itertuples():
        performer_only[r.event_id] |= performer_genres[r.performer_id]
    types = defaultdict(set)
    for r in pd.read_csv(f"{tables}/event_event_types.csv").itertuples():
        types[r.event_id].add(type_name[r.event_type_id])
    genres, kinds = sorted(genre_name.index), sorted(type_name.index)
    genre_names, type_names = [genre_name[g] for g in genres], [type_name[t] for t in kinds]
    genre_index, type_index = {n: i for i, n in enumerate(genre_names)}, {n: i for i, n in enumerate(type_names)}
    event_genre = np.zeros((len(events), len(genre_names)), dtype=np.float32)
    event_type = np.zeros((len(events), len(type_names)), dtype=np.float32)
    for event_id, i in index.items():
        for g in performer_only[event_id] - tag[event_id]:
            event_genre[i, genre_index[g]] = PERFORMER_ONLY_WEIGHT
        for g in tag[event_id]:
            event_genre[i, genre_index[g]] = 1.0
        for t in types[event_id]:
            event_type[i, type_index[t]] = 1.0
    # rare genres and types count more: inverse of how many events carry them
    idf_genre = np.log(len(events) / (1 + (event_genre > 0).sum(0)))
    idf_type = np.log(len(events) / (1 + (event_type > 0).sum(0)))
    return {"events": events, "event_genre": event_genre, "event_type": event_type, "genre_index": genre_index, "type_index": type_index,
            "idf_genre": idf_genre, "idf_type": idf_type}


def profile_vector(names: list[str], index: dict, idf: np.ndarray) -> np.ndarray | None:
    """Weights of the profile's choices, summing to 1, so that choosing many items dilutes each one."""
    picked = [index[n] for n in names if n in index]
    if not picked:
        return None
    vector = np.zeros(len(index), dtype=np.float32)
    vector[picked] = idf[picked]
    return vector / vector.sum()


def boost(features: dict, profile: dict, type_weight: float = 0.4) -> np.ndarray:
    """0 to 1 for every event: how much of the profile it matches."""
    g = profile_vector(profile["music"], features["genre_index"], features["idf_genre"])
    t = profile_vector(profile["types"], features["type_index"], features["idf_type"])
    parts = []
    if g is not None:
        parts.append(((1 - type_weight) if t is not None else 1.0, features["event_genre"] @ g))
    if t is not None:
        parts.append((type_weight if g is not None else 1.0, features["event_type"] @ t))
    return sum(w * b for w, b in parts) if parts else np.zeros(len(features["events"]), dtype=np.float32)


def rerank(scores: np.ndarray, boost_values: np.ndarray, weight: float) -> list[int]:
    """Takes the top pool by cosine, rescales those scores to 0-1, adds weight times the boost, returns the new top 10."""
    pool = np.argsort(-scores)[:POOL]
    low, high = scores[pool].min(), scores[pool].max()
    scaled = (scores[pool] - low) / (high - low)
    order = np.argsort(-(scaled + weight * boost_values[pool]), kind="stable")
    return [int(pool[i]) for i in order[:TOP]]


def analyse(scores_per_query: list[np.ndarray], profiles: dict[str, dict], features: dict, type_weight: float = 0.4) -> dict:
    report = {"weights": list(WEIGHTS), "profiles": {}}
    names = features["events"]["name"].tolist()
    for profile_name, profile in profiles.items():
        values = boost(features, profile, type_weight)
        per_query = []
        for query, scores in zip(QUERIES, scores_per_query):
            base = rerank(scores, values, 0.0)
            rank_of = {int(e): r + 1 for r, e in enumerate(np.argsort(-scores)[:POOL])}
            by_weight = {}
            for w in WEIGHTS:
                new = rerank(scores, values, w)
                promoted = [e for e in new if e not in base]
                by_weight[str(w)] = {"changed_slots": len(promoted), "deepest_promoted_rank": max([rank_of[e] for e in promoted], default=0),
                                     "mean_boost_top10": float(values[new].mean()),
                                     "top10": [{"name": names[e], "boost": float(values[e]), "cosine_rank": rank_of[e]} for e in new]}
            per_query.append({"query": query, "by_weight": by_weight})
        report["profiles"][profile_name] = {"music": profile["music"], "types": profile["types"], "queries": per_query}
    return report


def summarise(report: dict) -> dict:
    out = {}
    for w in WEIGHTS:
        key = str(w)
        changed = [q["by_weight"][key]["changed_slots"] for p in report["profiles"].values() for q in p["queries"]]
        deepest = [q["by_weight"][key]["deepest_promoted_rank"] for p in report["profiles"].values() for q in p["queries"]]
        out[key] = {"mean_changed_slots": float(np.mean(changed)), "share_with_any_change": float(np.mean([c > 0 for c in changed])),
                    "deepest_promoted_rank": int(max(deepest))}
    return out


def read_users(path: str, limit: int, max_profile: int, seed: int) -> tuple[dict, int]:
    """Real onboarding choices only (no user ids are kept): music genres and party types."""
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    split = lambda s: [x for x in (s or "").split("; ") if x]
    usable, skipped = [], 0
    for r in rows:
        profile = {"music": split(r.get("music_genres")), "types": split(r.get("party_types"))}
        if len(profile["music"]) + len(profile["types"]) == 0 or len(profile["music"]) + len(profile["types"]) > max_profile:
            skipped += 1
            continue
        usable.append(profile)
    random.Random(seed).shuffle(usable)
    return {f"real profile {i + 1}": p for i, p in enumerate(usable[:limit])}, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--users", default="", help="optional CSV of real users (music_genres and party_types columns, items separated by '; ')")
    parser.add_argument("--max-users", type=int, default=100)
    parser.add_argument("--max-profile", type=int, default=15, help="skip real profiles that chose more items than this (they picked almost everything)")
    parser.add_argument("--type-weight", type=float, default=0.4, help="share of the boost that comes from party types instead of genres")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(ROOT / "data" / "search_runs" / "profile_boost.json"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in .env")
    from openai import OpenAI

    features = load_features(args.tables)
    vectors = np.load(ROOT / "data" / "graph" / "event_text.npy").astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert len(vectors) == len(features["events"]), "events.csv and the saved vectors do not match"
    queries = np.array([d.embedding for d in OpenAI().embeddings.create(model=MODEL, input=QUERIES).data], dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    scores = [vectors @ q for q in queries]

    personas = analyse(scores, PERSONAS, features, args.type_weight)
    print("hand-made profiles: mean number of top-10 slots that change, for each boost weight")
    for key, row in summarise(personas).items():
        print(f"  weight {key}: {row['mean_changed_slots']:.1f} of 10 change, any change in {row['share_with_any_change']:.0%} of (profile, query) pairs, deepest promoted event was at cosine rank {row['deepest_promoted_rank']}")
    print("per profile at weight 0.1:")
    for name, p in personas["profiles"].items():
        print(f"  {name}: {[q['by_weight']['0.1']['changed_slots'] for q in p['queries']]} changed slots for the 8 queries")
    report = {"personas": personas, "personas_summary": summarise(personas)}

    if args.users:
        users, skipped = read_users(args.users, args.max_users, args.max_profile, args.seed)
        real = analyse(scores, users, features, args.type_weight)
        report["real_summary"] = summarise(real)
        report["real_profiles_used"], report["real_profiles_skipped"] = len(users), skipped
        print(f"\nreal onboarding profiles: {len(users)} used, {skipped} skipped (empty or more than {args.max_profile} choices)")
        for key, row in report["real_summary"].items():
            print(f"  weight {key}: {row['mean_changed_slots']:.1f} of 10 change, any change in {row['share_with_any_change']:.0%} of (profile, query) pairs, deepest promoted event at rank {row['deepest_promoted_rank']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
