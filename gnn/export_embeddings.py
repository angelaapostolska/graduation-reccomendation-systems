"""Exports the event text vectors from Postgres in graph order. Usage: python gnn/export_embeddings.py [--column embedding_ada002]"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--column", default="embedding_ada002")
    parser.add_argument("--graph", default=str(ROOT / "data" / "graph" / "graph.npz"))
    parser.add_argument("--out", default=str(ROOT / "data" / "graph" / "event_text.npy"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL must be set in .env")
    event_ids = np.load(args.graph, allow_pickle=False)["ids_event"]
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(f"SELECT event_id, {args.column}::text FROM events WHERE {args.column} IS NOT NULL")
        vectors = {event_id: np.fromstring(text[1:-1], sep=",", dtype=np.float32) for event_id, text in cur.fetchall()}
    missing = [e for e in event_ids if e not in vectors]
    if missing:
        raise SystemExit(f"{len(missing)} events have no {args.column} vector, for example {missing[:3]}")
    matrix = np.stack([vectors[e] for e in event_ids])
    np.save(args.out, matrix)
    print(f"saved {matrix.shape} to {args.out}")


if __name__ == "__main__":
    main()
