"""Embeds the names of the label nodes of the graph (music genres, event types, venue types, cities) with ada-002, so the graph knows
what its labels mean. The files are saved as node_names.npz in the graph folder, in the node order of graph.npz.
Usage: python gnn/node_names.py            (about 250 short names, a fraction of a cent)
       GRAPH_DIR=data/graph_stripped python gnn/node_names.py"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from common import GRAPH_DIR, ROOT

sys.path.insert(0, str(ROOT / "search"))
NAME_TABLES = {"music_genre": ("music_genres", "music_genre_id"), "event_type": ("event_types", "event_type_id"),
               "business_type": ("business_types", "type_id"), "city": ("cities", "city_id")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--graph", default=str(GRAPH_DIR / "graph.npz"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    graph = np.load(args.graph, allow_pickle=False)
    names = {}
    for kind, (table, key) in NAME_TABLES.items():
        frame = pd.read_csv(f"{args.tables}/{table}.csv", dtype={key: str}).set_index(key)
        names[kind] = [str(frame.loc[i, "name"]) for i in graph[f"ids_{kind}"]]
    total = sum(len(v) for v in names.values())
    print(f"{total} names, for example: " + "; ".join(f"{k}: {v[:3]}" for k, v in names.items()))
    if args.dry_run:
        return
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in .env")
    from embedders import embed

    out = {}
    for kind, values in names.items():
        vectors, _ = embed("text-embedding-ada-002", values)
        out[kind] = np.array(vectors, dtype=np.float32)
    np.savez(GRAPH_DIR / "node_names.npz", **out)
    print(f"saved node_names.npz in {GRAPH_DIR}")


if __name__ == "__main__":
    main()
