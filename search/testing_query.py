"""Runs a list of test queries and saves what each returns. Usage: python search/testing_query.py [--top 10] [--model text-embedding-ada-002] [--out file]"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
COLUMNS = {"text-embedding-3-small": "embedding", "text-embedding-ada-002": "embedding_ada002"}
QUERIES = [
    "free techno night in Skopje",
    "cheap standup comedy in Skopje",
    "live jazz concert this weekend",
    "workshop for beginners",
    "hip-hop party with a student discount",
    "romantic dinner with live music",
    "open mic night",
    "beach party with DJ",
]
SQL = """
SELECT d.name, d.event_date, d.start_time, d.venue, d.city, d.event_types, d.music_genres, d.entrance_mkd, d.description,
       1 - (e.{col} <=> %s::vector) AS similarity
FROM events e JOIN event_details d USING (event_id)
WHERE e.{col} IS NOT NULL
ORDER BY e.{col} <=> %s::vector
LIMIT %s
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(COLUMNS), default="text-embedding-3-small")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url or not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("DATABASE_URL and OPENAI_API_KEY must be set in .env")
    client = OpenAI()
    out = Path(args.out) if args.out else ROOT / "data" / "search_runs" / f"results_{args.model}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    vectors = client.embeddings.create(model=args.model, input=QUERIES).data
    lines: list[str] = []
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        for query, item in zip(QUERIES, vectors):
            literal = "[" + ",".join(f"{x:.7f}" for x in item.embedding) + "]"
            cur.execute(SQL.format(col=COLUMNS[args.model]), (literal, literal, args.top))
            lines.append(f'query searched: "{query}" (model: {args.model})')
            lines.append("events matched:")
            for rank, (name, day, start, venue, city, types, genres, price, description, score) in enumerate(cur.fetchall(), 1):
                price_text = "free" if price == 0 else f"{price} MKD" if price else "no price"
                lines.append(f"  {rank}. [{score:.3f}] {name} | {day} {start.strftime('%H:%M') if start else ''} | {venue}, {city} | {types or '-'} | {genres or '-'} | {price_text}")
                lines.append(f"     {description}")
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {len(QUERIES)} queries to {out}")


if __name__ == "__main__":
    main()
