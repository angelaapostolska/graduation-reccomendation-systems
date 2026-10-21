"""Compares which parts of the event data set ada-002 and the GNN vectors reach for the test queries.
Usage: python gnn/coverage.py [--out data/search_runs/coverage.json]"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from dotenv import load_dotenv

from common import GRAPH_DIR, ROOT, load_graph

sys.path.insert(0, str(ROOT / "search"))
from testing_query import QUERIES  # noqa: E402

MODEL = "text-embedding-ada-002"
KS = (10, 50, 100)


def unit(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    out = np.empty(len(scores))
    out[order] = np.arange(len(scores))
    return out


def analyse(text: np.ndarray, gnn: np.ndarray, queries: np.ndarray, has_genre: np.ndarray, venue_of: np.ndarray) -> dict:
    """text, gnn: unit event vectors; queries: unit query vectors; has_genre: bool per event; venue_of: venue index per event."""
    total = len(text)
    report = {"events": total, "per_query": [], "union": {}}
    seen = {name: {k: set() for k in KS} for name in ("ada002", "gnn")}
    for vector in queries:
        ada, net = text @ vector, gnn @ vector
        row = {"events_scored": {"ada002": int(len(ada)), "gnn": int(len(net))}}
        for name, scores in (("ada002", ada), ("gnn", net)):
            order = np.argsort(-scores)
            row[name] = {"score_std": float(scores.std()), "top1_minus_median": float(scores.max() - np.median(scores))}
            for k in KS:
                top = order[:k]
                seen[name][k].update(top.tolist())
                row[name][f"top{k}"] = {"no_genre": float((~has_genre[top]).mean()), "venues": int(len(set(venue_of[top].tolist())))}
        for k in KS:
            shared = len(set(np.argsort(-ada)[:k].tolist()) & set(np.argsort(-net)[:k].tolist()))
            row[f"shared_top{k}"] = shared
        row["rank_correlation"] = float(np.corrcoef(ranks(ada), ranks(net))[0, 1])
        report["per_query"].append(row)
    for k in KS:
        a, g = seen["ada002"][k], seen["gnn"][k]
        report["union"][f"top{k}"] = {"ada002_distinct": len(a), "gnn_distinct": len(g), "shared": len(a & g), "either": len(a | g),
                                      "share_of_data_either": len(a | g) / total}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data" / "search_runs" / "coverage.json"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in .env")
    from openai import OpenAI

    graph = load_graph()
    text = unit(np.load(GRAPH_DIR / "event_text.npy"))
    gnn = unit(np.load(GRAPH_DIR / "event_gnn.npy"))
    total = graph["counts"]["event"]
    has_genre = np.zeros(total, dtype=bool)
    has_genre[graph["edges"]["event__has_genre__music_genre"][0].numpy()] = True
    events, venues = graph["edges"]["event__hosted_at__venue"]
    venue_of = np.full(total, -1)
    venue_of[events.numpy()] = venues.numpy()
    queries = unit(np.array([item.embedding for item in OpenAI().embeddings.create(model=MODEL, input=QUERIES).data]))

    report = analyse(text, gnn, queries, has_genre, venue_of)
    report["queries"] = QUERIES
    report["share_without_genre"] = float((~has_genre).mean())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"events in the data set: {total}; without a genre tag: {report['share_without_genre']:.0%}")
    print("every query is scored against all events by both methods")
    for query, row in zip(QUERIES, report["per_query"]):
        print(f'"{query}": shared top10/50/100 = {row["shared_top10"]}/{row["shared_top50"]}/{row["shared_top100"]}, rank correlation {row["rank_correlation"]:.2f}, '
              f'no-genre share of top 100: ada-002 {row["ada002"]["top100"]["no_genre"]:.0%}, network {row["gnn"]["top100"]["no_genre"]:.0%}')
    for k, values in report["union"].items():
        print(f'across all queries, {k}: ada-002 reaches {values["ada002_distinct"]} events, network {values["gnn_distinct"]}, shared {values["shared"]}, '
              f'either {values["either"]} ({values["share_of_data_either"]:.1%} of the data)')
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
