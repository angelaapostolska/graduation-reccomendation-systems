"""Runs the 20 test queries with exact search (every event compared, no index) and saves what each returns.
Usage: python search/testing_query.py [--model all|<model>] [--top 10] [--approximate]"""

from __future__ import annotations

import argparse
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
    # twelve queries added for the final comparison, fixed before any result was seen
    "late night DJ set in Ohrid",
    "kids workshop on the weekend",
    "live rock band tonight",
    "wine tasting evening",
    "stand up comedy in English",
    "karaoke night",
    "free entry party for students",
    "outdoor movie night",
    "salsa dancing lessons",
    "brunch with live music",
    "underground electronic music",
    "something cultural in Skopje, theatre or poetry",
]
SQL = """
SELECT d.name, d.event_date, d.venue, d.city, d.event_types, d.music_genres, d.entrance_mkd, d.description,
       1 - (e.{col} <=> %s::vector) AS similarity
FROM events e JOIN event_details d USING (event_id)
WHERE e.{col} IS NOT NULL
ORDER BY e.{col} <=> %s::vector
LIMIT %s
"""


def run_model(model: str, top: int, exact: bool) -> Path:
    url = check_env(model)
    out = ROOT / "data" / "search_runs" / f"results_{model}{'_exact' if exact else ''}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    vectors, _ = embed(model, QUERIES, is_query=True)
    lines: list[str] = []
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        if exact:
            cur.execute("SET enable_indexscan = off")
            cur.execute("SET enable_bitmapscan = off")
        for query, vector in zip(QUERIES, vectors):
            cur.execute(SQL.format(col=MODELS[model]["column"]), (literal(vector), literal(vector), top))
            lines.append(f'query searched: "{query}" (model: {model})')
            lines.append("events matched:")
            for rank, (name, day, venue, city, types, genres, price, description, score) in enumerate(cur.fetchall(), 1):
                price_text = "free" if price == 0 else f"{price} MKD" if price else "no price"
                lines.append(f"  {rank}. [{score:.3f}] {name} | {day} | {venue}, {city} | {types or '-'} | {genres or '-'} | {price_text}")
                lines.append(f"     {description}")
            lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{model}: saved {len(QUERIES)} queries to {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["all", *MODELS], default="all")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--approximate", action="store_true", help="use the index instead of comparing with every event")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    for model in (MODELS if args.model == "all" else [args.model]):
        run_model(model, args.top, exact=not args.approximate)


if __name__ == "__main__":
    main()
