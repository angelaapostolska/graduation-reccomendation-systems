"""Computes text embeddings (event name + description). Usage: python search/embed_events.py [--model text-embedding-ada-002]"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "text-embedding-3-small": ("embedding", 0.02),
    "text-embedding-ada-002": ("embedding_ada002", 0.10),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="text-embedding-3-small")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="only embed N events (for a quick test)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("DATABASE_URL and OPENAI_API_KEY must be set in .env")
    client = OpenAI()
    column, price = MODELS[args.model]

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(f"ALTER TABLE events ADD COLUMN IF NOT EXISTS {column} vector(1536)")
        cur.execute(f"SELECT event_id, name, description FROM events WHERE {column} IS NULL ORDER BY event_id"
                    + (f" LIMIT {args.limit}" if args.limit else ""))
        rows = cur.fetchall()
        print(f"{len(rows)} events to embed")
        tokens = 0
        for start in range(0, len(rows), args.batch):
            batch = rows[start:start + args.batch]
            for attempt in range(4):
                try:
                    response = client.embeddings.create(model=args.model, input=[f"{name}. {description}" for _, name, description in batch])
                    break
                except Exception:  # noqa: BLE001 - retry transient API errors
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            tokens += response.usage.total_tokens
            cur.executemany(
                f"UPDATE events SET {column} = %s::vector WHERE event_id = %s",
                [("[" + ",".join(f"{x:.7f}" for x in item.embedding) + "]", batch[i][0]) for i, item in enumerate(response.data)],
            )
            conn.commit()
            print(f"  {min(start + args.batch, len(rows))}/{len(rows)}")
        cur.execute(f"CREATE INDEX IF NOT EXISTS events_{column}_idx ON events USING hnsw ({column} vector_cosine_ops)")
        conn.commit()
        print(f"done: {tokens:,} tokens, about ${tokens * price / 1e6:.3f}")


if __name__ == "__main__":
    main()
