"""Runs the test queries against text, graph-smoothed or GNN event vectors and saves the results.
Usage: python gnn/search.py --variant text|smooth|gnn [--top 10]"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import psycopg
from dotenv import load_dotenv
from openai import OpenAI
from psycopg.rows import dict_row

from common import GRAPH_DIR, ROOT

sys.path.insert(0, str(ROOT / "search"))
from testing_query import QUERIES  # noqa: E402

FILES = {"text": "event_text.npy", "smooth": "event_smooth.npy", "gnn": "event_gnn.npy"}
MODEL = "text-embedding-ada-002"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=list(FILES), required=True)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("DATABASE_URL and OPENAI_API_KEY must be set in .env")

    events = np.load(GRAPH_DIR / FILES[args.variant]).astype(np.float32)
    events /= np.linalg.norm(events, axis=1, keepdims=True)
    event_ids = np.load(GRAPH_DIR / "graph.npz", allow_pickle=False)["ids_event"]
    queries = np.array([item.embedding for item in OpenAI().embeddings.create(model=MODEL, input=QUERIES).data], dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    with psycopg.connect(url, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM event_details")
        details = {row["event_id"]: row for row in cur.fetchall()}

    lines: list[str] = []
    for query, vector in zip(QUERIES, queries):
        scores = events @ vector
        lines.append(f'query searched: "{query}" (model: {args.variant})')
        lines.append("events matched:")
        for rank, index in enumerate(np.argsort(-scores)[: args.top], 1):
            row = details[str(event_ids[index])]
            start = row.get("start_time")
            price = row["entrance_mkd"]
            price_text = "free" if price == 0 else f"{price} MKD" if price else "no price"
            lines.append(f"  {rank}. [{scores[index]:.3f}] {row['name']} | {row['event_date']} {start.strftime('%H:%M') if start else ''} | {row['venue']}, {row['city']} | "
                         f"{row['event_types'] or '-'} | {row['music_genres'] or '-'} | {price_text}")
            lines.append(f"     {row['description']}")
        lines.append("")
    out = args.out or str(ROOT / "data" / "search_runs" / f"results_{args.variant}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(f"saved {len(QUERIES)} queries to {out}")


if __name__ == "__main__":
    main()
