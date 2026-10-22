"""Runs a list of test queries and saves what each returns. Usage: python search/testing_query.py [--top 10] [--model distilbert] [--out file]"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from embedders import MODELS, check_env, embed, literal

ROOT = Path(__file__).resolve().parent.parent
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
SELECT d.name, d.event_date, d.venue, d.city, d.event_types, d.music_genres, d.entrance_mkd, d.description,
       1 - (e.{col} <=> %s::vector) AS similarity
FROM events e JOIN event_details d USING (event_id)
WHERE e.{col} IS NOT NULL
ORDER BY e.{col} <=> %s::vector
LIMIT %s
"""


def read_picks(path: Path) -> list[set[str]]:
    """Events (name and date) listed under each query of a saved results file."""
    picks: list[set[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("query searched:"):
            picks.append(set())
        elif re.match(r"\s+\d+\. \[", line) and picks:
            name, day = [part.strip() for part in line.split("] ", 1)[1].split(" | ")[:2]]
            picks[-1].add(f"{name}|{day}")
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), default="text-embedding-3-small")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--out", default="")
    parser.add_argument("--exact", action="store_true", help="compare the query with every event instead of using the approximate index")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = check_env(args.model)
    suffix = "_exact" if args.exact else ""
    out = Path(args.out) if args.out else ROOT / "data" / "search_runs" / f"results_{args.model}{suffix}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    vectors, _ = embed(args.model, QUERIES)
    lines: list[str] = []
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        if args.exact:
            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")
        for query, vector in zip(QUERIES, vectors):
            cur.execute(SQL.format(col=MODELS[args.model]["column"]), (literal(vector), literal(vector), args.top))
            lines.append(f'query searched: "{query}" (model: {args.model})')
            lines.append("events matched:")
            for rank, (name, day, venue, city, types, genres, price, description, score) in enumerate(cur.fetchall(), 1):
                price_text = "free" if price == 0 else f"{price} MKD" if price else "no price"
                lines.append(f"  {rank}. [{score:.3f}] {name} | {day} | {venue}, {city} | {types or '-'} | {genres or '-'} | {price_text}")
                lines.append(f"     {description}")
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {len(QUERIES)} queries to {out}")
    earlier = out.with_name(f"results_{args.model}.txt")
    if args.exact and earlier.exists():
        old, new = read_picks(earlier), read_picks(out)
        print("events in the exact top 10 that the approximate search did not return, per query:")
        for query, before, after in zip(QUERIES, old, new):
            print(f'  "{query}": {len(after - before)} of {len(after)}')


if __name__ == "__main__":
    main()
