"""Third network: chooses the final top 10 from the ada-002 and GNN candidates, using both cosine scores.
Evaluated by leaving one query out at a time. Usage: python gnn/combine.py [--grades data/grades.csv]"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv

from common import GRAPH_DIR, ROOT, load_graph

sys.path.insert(0, str(ROOT / "search"))
from testing_query import QUERIES  # noqa: E402

MODEL = "text-embedding-ada-002"
TOP = 10
SEEDS = 10


def unit(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores)
    out = np.empty(len(scores))
    out[order] = np.arange(len(scores))
    return out


def candidate_features(ada: np.ndarray, net: np.ndarray, has_genre: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidates are the union of both top 10s. Features are the two confidences (z-scored over all events), the two ranks and two flags."""
    ra, rn = ranks(ada), ranks(net)
    candidates = np.array(sorted(set(np.argsort(ra)[:TOP].tolist()) | set(np.argsort(rn)[:TOP].tolist())))
    za, zn = (ada - ada.mean()) / ada.std(), (net - net.mean()) / net.std()
    both = ((ra[candidates] < TOP) & (rn[candidates] < TOP)).astype(float)
    features = np.stack([za[candidates], zn[candidates], np.log1p(ra[candidates]), np.log1p(rn[candidates]), both, (~has_genre[candidates]).astype(float)], axis=1)
    return candidates, features


def dcg(grades: list[int]) -> float:
    return sum((2 ** g - 1) / np.log2(rank + 2) for rank, g in enumerate(grades))


def scores(picked: list[int], grades: dict[int, int], pool: list[int]) -> dict:
    got = [grades[i] for i in picked[:TOP]]
    ideal = dcg(sorted(pool, reverse=True)[:TOP])
    return {"strict": sum(g == 2 for g in got) / TOP, "lenient": sum(g >= 1 for g in got) / TOP, "ndcg": dcg(got) / ideal if ideal else 0.0}


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    x_train = torch.tensor((train_x - mean) / std, dtype=torch.float32)
    x_test = torch.tensor((test_x - mean) / std, dtype=torch.float32)
    target = torch.tensor(train_y / 2.0, dtype=torch.float32)
    outputs = []
    for seed in range(SEEDS):
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(x_train.shape[1], 8), nn.ReLU(), nn.Linear(8, 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-3)
        for _ in range(300):
            loss = F.binary_cross_entropy_with_logits(model(x_train).squeeze(1), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            outputs.append(model(x_test).squeeze(1).numpy())
    return np.mean(outputs, axis=0)


def evaluate(ada: list[np.ndarray], net: list[np.ndarray], has_genre: np.ndarray, grades: list[dict[int, int]]) -> dict:
    """ada, net: cosine score of every event per query; grades: event index to grade per query."""
    built = []
    for a, n, g in zip(ada, net, grades):
        candidates, features = candidate_features(a, n, has_genre)
        known = np.array([int(c) in g for c in candidates])
        built.append((candidates, features, known))
    methods = defaultdict(list)
    picks = {"only_ada": 0, "only_network": 0, "both": 0}
    for q, (candidates, features, known) in enumerate(built):
        g, pool = grades[q], list(grades[q].values())
        cand, feat = candidates[known], features[known]
        train_x = np.concatenate([built[o][1][built[o][2]] for o in range(len(built)) if o != q])
        train_y = np.array([grades[o][int(c)] for o in range(len(built)) if o != q for c in built[o][0][built[o][2]]], dtype=float)
        by = {
            "ada-002": np.argsort(-feat[:, 0]),
            "network": np.argsort(-feat[:, 1]),
            "average of both confidences": np.argsort(-(feat[:, 0] + feat[:, 1])),
            "third network": np.argsort(-fit_predict(train_x, train_y, feat)),
            "best possible from these candidates": np.argsort(-np.array([g[int(c)] for c in cand], dtype=float), kind="stable"),
        }
        for name, order in by.items():
            methods[name].append(scores([int(cand[i]) for i in order], g, pool))
        for i in by["third network"][:TOP]:
            f = feat[i]
            picks["both" if f[4] else "only_ada" if f[2] < np.log1p(TOP) else "only_network"] += 1
    summary = {name: {k: float(np.mean([r[k] for r in rows])) for k in ("strict", "lenient", "ndcg")} for name, rows in methods.items()}
    per_query = {name: [r["strict"] for r in rows] for name, rows in methods.items()}
    return {"summary": summary, "per_query_strict": per_query, "third_network_picks": picks, "ungraded_candidates": int(sum((~k).sum() for _, _, k in built))}


def load_grades(path: str, tables: str) -> list[dict[int, int]]:
    venues = {r["venue_id"]: r["name"] for r in csv.DictReader(open(f"{tables}/venues.csv", encoding="utf-8", newline=""))}
    index = defaultdict(list)
    for i, r in enumerate(csv.DictReader(open(f"{tables}/events.csv", encoding="utf-8", newline=""))):
        index[(r["name"], r["event_date"], venues[r["venue_id"]])].append(i)
    out = [dict() for _ in QUERIES]
    for r in csv.DictReader(open(path, encoding="utf-8", newline="")):
        for i in index.get((r["event"], r["date"], r["venue"]), []):
            out[QUERIES.index(r["query"])][i] = int(r["grade"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grades", default=str(ROOT / "data" / "grades.csv"))
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--out", default=str(ROOT / "data" / "search_runs" / "combine.json"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in .env")
    from openai import OpenAI

    graph = load_graph()
    text, gnn = unit(np.load(GRAPH_DIR / "event_text.npy")), unit(np.load(GRAPH_DIR / "event_gnn.npy"))
    has_genre = np.zeros(graph["counts"]["event"], dtype=bool)
    has_genre[graph["edges"]["event__has_genre__music_genre"][0].numpy()] = True
    queries = unit(np.array([item.embedding for item in OpenAI().embeddings.create(model=MODEL, input=QUERIES).data]))
    grades = load_grades(args.grades, args.tables)
    print("graded events per query:", [len(g) for g in grades])

    report = evaluate([text @ q for q in queries], [gnn @ q for q in queries], has_genre, grades)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"candidates without a grade: {report['ungraded_candidates']}")
    print("method (leave one query out), strict P@10 / lenient P@10 / nDCG@10:")
    for name, values in report["summary"].items():
        print(f"  {name}: {values['strict']:.3f} / {values['lenient']:.3f} / {values['ndcg']:.3f}")
    print("strict P@10 per query:")
    for name, values in report["per_query_strict"].items():
        print(f"  {name}: {' '.join(f'{v:.1f}' for v in values)}")
    print("final picks of the third network that came from:", report["third_network_picks"])
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
