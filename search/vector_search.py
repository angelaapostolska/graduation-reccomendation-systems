"""Baseline search: ranks events by text similarity to a query. Usage: python search/vector_search.py "query" [--top 10]"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = {"text-embedding-3-small": "embedding", "text-embedding-ada-002": "embedding_ada002"}
SQL = """
SELECT d.name, d.event_date, d.start_time, d.venue, d.city, d.music_genres, d.event_types, d.performers, d.entrance_mkd, d.description,
       1 - (e.{col} <=> %s::vector) AS similarity
FROM events e JOIN event_details d USING (event_id)
WHERE e.{col} IS NOT NULL
ORDER BY e.{col} <=> %s::vector
LIMIT %s
"""


def search(cur: psycopg.Cursor, client: OpenAI, query: str, top: int, model: str) -> None:
    vector = client.embeddings.create(model=model, input=[query]).data[0].embedding
    literal = "[" + ",".join(f"{x:.7f}" for x in vector) + "]"
    cur.execute(SQL.format(col=COLUMNS[model]), (literal, literal, top))
    print(f"\nQuery: {query}")
    for rank, (name, day, start, venue, city, genres, types, performers, price, description, similarity) in enumerate(cur.fetchall(), 1):
        price_text = "free" if price == 0 else f"{price} MKD" if price else "no price"
        print(f"{rank:2}. [{similarity:.3f}] {name} | {day} {start.strftime('%H:%M') if start else ''} | {venue}, {city} | {types or '-'} | {genres or '-'} | {price_text}")
        print(f"      {description}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="query text; leave empty for an interactive prompt")
    parser.add_argument("--model", choices=list(COLUMNS), default="text-embedding-3-small")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("DATABASE_URL and OPENAI_API_KEY must be set in .env")
    client = OpenAI()
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        if args.query:
            search(cur, client, " ".join(args.query), args.top, args.model)
            return
        while query := input("\nSearch (empty to quit): ").strip():
            search(cur, client, query, args.top, args.model)


if __name__ == "__main__":
    main()
