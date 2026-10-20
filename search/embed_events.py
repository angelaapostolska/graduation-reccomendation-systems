"""Computes text embeddings (event name + description). Usage: python search/embed_events.py [--model text-embedding-ada-002|distilbert]"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from embedders import MODELS, check_env, embed, literal

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="text-embedding-3-small")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="only embed N events (for a quick test)")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = check_env(args.model)
    column, dim, price = MODELS[args.model]["column"], MODELS[args.model]["dim"], MODELS[args.model]["price"]

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(f"ALTER TABLE events ADD COLUMN IF NOT EXISTS {column} vector({dim})")
        cur.execute(f"SELECT event_id, name, description FROM events WHERE {column} IS NULL ORDER BY event_id"
                    + (f" LIMIT {args.limit}" if args.limit else ""))
        rows = cur.fetchall()
        print(f"{len(rows)} events to embed with {args.model}")
        tokens = 0
        for start in range(0, len(rows), args.batch):
            batch = rows[start:start + args.batch]
            for attempt in range(4):
                try:
                    vectors, used = embed(args.model, [f"{name}. {description}" for _, name, description in batch])
                    break
                except Exception:  # noqa: BLE001 - retry transient API errors
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
            tokens += used
            cur.executemany(
                f"UPDATE events SET {column} = %s::vector WHERE event_id = %s",
                [(literal(vector), batch[i][0]) for i, vector in enumerate(vectors)],
            )
            conn.commit()
            print(f"  {min(start + args.batch, len(rows))}/{len(rows)}")
        cur.execute(f"CREATE INDEX IF NOT EXISTS events_{column}_idx ON events USING hnsw ({column} vector_cosine_ops)")
        conn.commit()
        print(f"done: {tokens:,} tokens, about ${tokens * price / 1e6:.3f}")


if __name__ == "__main__":
    main()
