"""Baseline search: ranks events by text similarity to a query. Usage: python search/vector_search.py "query" [--top 10] [--model distilbert]"""

from __future__ import annotations

import argparse
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from embedders import MODELS, check_env, embed, literal

ROOT = Path(__file__).resolve().parent.parent
SQL = """
SELECT d.name, d.venue, d.city, d.music_genres, d.event_types, d.performers, d.entrance_mkd, d.description,
       1 - (e.{col} <=> %s::vector) AS similarity
FROM events e JOIN event_details d USING (event_id)
WHERE e.{col} IS NOT NULL
ORDER BY e.{col} <=> %s::vector
LIMIT %s
"""


def search(cur: psycopg.Cursor, query: str, top: int, model: str) -> None:
    vector = literal(embed(model, [query], is_query=True)[0][0])
    cur.execute(SQL.format(col=MODELS[model]["column"]), (vector, vector, top))
    print(f"\nQuery: {query}")
    for rank, (name, venue, city, genres, types, performers, price, description, similarity) in enumerate(cur.fetchall(), 1):
        price_text = "free" if price == 0 else f"{price} MKD" if price else "no price"
        print(f"{rank:2}. [{similarity:.3f}] {name} | {venue}, {city} | {types or '-'} | {genres or '-'} | {price_text}")
        print(f"      {description}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="*", help="query text; leave empty for an interactive prompt")
    parser.add_argument("--model", choices=list(MODELS), default="text-embedding-3-small")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = check_env(args.model)
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        if args.query:
            search(cur, " ".join(args.query), args.top, args.model)
            return
        while query := input("\nSearch (empty to quit): ").strip():
            search(cur, query, args.top, args.model)


if __name__ == "__main__":
    main()
