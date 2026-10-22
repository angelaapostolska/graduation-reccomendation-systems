"""Cold-start test: queries aimed at real events that have no genre tag, a performer with known genres elsewhere, and no genre word in their text.
Compares the top results and cosine scores of ada-002 and the GNN. Usage: python gnn/cold_start.py [--out data/search_runs/cold_start.json]"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from common import GRAPH_DIR, ROOT

MODEL = "text-embedding-ada-002"
TOP = 10
DEPTHS = (50, 100)
# (query, genre the query asks for, or None for a control that has no genre truth)
QUERIES = [
    ("pop party night", "Pop"),
    ("ex-yu hits night", "Ex-Yu Hits"),
    ("balkan party with live music", "Balkan Party"),
    ("house music DJ night", "House"),
    ("hip-hop night", "Hip-Hop"),
    ("techno DJ night", "Techno"),
    ("R&B night", "R&B"),
    ("indie live music", "Indie"),
    ("rock band live", "Rock"),
    ("electronic music night", "Electronic"),
    ("jazz evening", "Jazz"),
    ("macedonian music night", "Macedonian"),
    ("watch the football match", None),
    ("workshop for beginners", None),
]


def unit(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def load_tables(tables: str) -> dict:
    read = lambda name: pd.read_csv(f"{tables}/{name}.csv")
    events, tags, performers = read("events"), read("event_music_genres"), read("event_performers")
    genre_name = read("music_genres").set_index("music_genre_id")["name"]
    performer_name = read("performers").set_index("performer_id")["name"]
    performer_genres = defaultdict(set)
    for r in read("performer_music_genres").itertuples():
        performer_genres[r.performer_id].add(genre_name[r.music_genre_id])
    event_tags, event_performers = defaultdict(set), defaultdict(list)
    for r in tags.itertuples():
        event_tags[r.event_id].add(genre_name[r.music_genre_id])
    for r in performers.itertuples():
        event_performers[r.event_id].append(r.performer_id)
    types = read("event_event_types").merge(read("event_types"), on="event_type_id")
    event_types = types.groupby("event_id")["name"].apply(list).to_dict()
    venue = read("venues").set_index("venue_id")["name"]
    patterns = [re.compile(r"\b" + re.escape(n.lower()) + r"\b") for n in genre_name]
    rows = []
    for e in events.itertuples():
        genres_by_performer = set().union(*[performer_genres[p] for p in event_performers[e.event_id]]) if event_performers[e.event_id] else set()
        text = str(e.description).lower() if isinstance(e.description, str) else ""
        in_text = any(p.search(text) for p in patterns)
        cold = e.origin == "real" and not event_tags[e.event_id] and bool(genres_by_performer) and not in_text
        rows.append({"event_id": e.event_id, "name": e.name, "date": e.event_date, "venue": venue[e.venue_id], "types": event_types.get(e.event_id, []),
                     "tags": sorted(event_tags[e.event_id]), "performers": [performer_name[p] for p in event_performers[e.event_id]],
                     "performer_genres": sorted(genres_by_performer), "cold": bool(cold)})
    return {"events": rows}


def analyse(text: np.ndarray, gnn: np.ndarray, query_vectors: np.ndarray, events: list[dict]) -> dict:
    report = {"cold_events": int(sum(e["cold"] for e in events)), "queries": []}
    for (query, genre), vector in zip(QUERIES, query_vectors):
        relevant = np.array([genre is not None and (genre in e["tags"] or genre in e["performer_genres"]) for e in events])
        cold_relevant = np.array([genre is not None and e["cold"] and genre in e["performer_genres"] for e in events])
        scores = {"ada002": text @ vector, "gnn": gnn @ vector}
        order = {name: np.argsort(-s) for name, s in scores.items()}
        entry = {"query": query, "genre": genre, "relevant_total": int(relevant.sum()), "cold_relevant_total": int(cold_relevant.sum())}
        top_sets = {name: set(o[:TOP].tolist()) for name, o in order.items()}
        for name in scores:
            top = order[name][:TOP]
            entry[name] = {
                "proxy_precision_at_10": float(relevant[top].mean()) if genre else None,
                "cold_hits_at_10": int(cold_relevant[top].sum()) if genre else None,
                "cold_recall": {str(d): float(cold_relevant[order[name][:d]].sum() / max(1, cold_relevant.sum())) for d in DEPTHS} if genre else None,
                "results": [{"rank": rank + 1, "cosine": float(scores[name][i]), "other_rank": int(np.where(order["gnn" if name == "ada002" else "ada002"] == i)[0][0]) + 1,
                             "in_other_top10": i in top_sets["gnn" if name == "ada002" else "ada002"], "cold": events[i]["cold"], "proxy_hit": bool(relevant[i]), **events[i]}
                            for rank, i in enumerate(top)],
            }
        report["queries"].append(entry)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--out", default=str(ROOT / "data" / "search_runs" / "cold_start.json"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in .env")
    from openai import OpenAI

    events = load_tables(args.tables)["events"]
    text, gnn = unit(np.load(GRAPH_DIR / "event_text.npy")), unit(np.load(GRAPH_DIR / "event_gnn.npy"))
    assert len(events) == len(text), "events.csv and the saved vectors do not match"
    vectors = unit(np.array([d.embedding for d in OpenAI().embeddings.create(model=MODEL, input=[q for q, _ in QUERIES]).data]))
    report = analyse(text, gnn, vectors, events)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"cold-start events (no genre tag, performer with known genres, no genre word in the text): {report['cold_events']}")
    print("query | proxy P@10 ada-002 / network | cold events in top 10 | share of cold events found in top 50 / 100")
    for q in report["queries"]:
        if q["genre"] is None:
            print(f'"{q["query"]}" (control, no genre truth)')
            continue
        a, g = q["ada002"], q["gnn"]
        print(f'"{q["query"]}": {a["proxy_precision_at_10"]:.1f} / {g["proxy_precision_at_10"]:.1f} | {a["cold_hits_at_10"]} / {g["cold_hits_at_10"]} of {q["cold_relevant_total"]} | '
              f'ada-002 {a["cold_recall"]["50"]:.0%} / {a["cold_recall"]["100"]:.0%}, network {g["cold_recall"]["50"]:.0%} / {g["cold_recall"]["100"]:.0%}')
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
