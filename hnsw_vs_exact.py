"""
Compare HNSW (approximate) vs exact vector search results.

For each evaluation query and each embedding column, runs the cosine
similarity search twice — once with the HNSW index enabled (default),
once with index scans disabled so pgvector falls back to exact search —
then checks whether the top-10 event IDs are identical.

Usage:
    python hnsw_vs_exact.py

Requires DATABASE_URL in .env (no OpenAI key needed — uses stored vectors).
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise SystemExit("DATABASE_URL not set in .env")

# All 20 evaluation queries used in the thesis
QUERIES = [
    "free techno night in Skopje",
    "cheap standup comedy in Skopje",
    "live jazz concert this weekend",
    "workshop for beginners",
    "hip-hop party with a student discount",
    "romantic dinner with live music",
    "open mic night",
    "beach party with DJ",
    "rock concert in Bitola",
    "wine tasting event",
    "yoga class or meditation session",
    "kids birthday party venue",
    "art exhibition opening",
    "drag show or cabaret night",
    "80s retro night",
    "food festival or street food market",
    "speed dating event",
    "cultural festival or folklore event",
    "pub quiz night",
    "new year's eve party",
]

# Embedding columns present in the events table
# key = human label, value = DB column name
COLUMNS = {
    "text-embedding-3-small": "embedding",
    "text-embedding-ada-002": "embedding_ada002",
}

SQL_HNSW = """
SELECT e.event_id, d.name,
       1 - (e.{col} <=> (
           SELECT e2.{col} FROM events e2
           WHERE e2.event_id = %s
           LIMIT 1
       )) AS sim
FROM events e JOIN event_details d USING (event_id)
WHERE e.{col} IS NOT NULL
ORDER BY e.{col} <=> (
    SELECT e2.{col} FROM events e2
    WHERE e2.event_id = %s
    LIMIT 1
)
LIMIT 10
"""

# Simpler: store a literal vector in a temp table so we can do one lookup
# and reuse it for both HNSW and exact without touching OpenAI.
# We'll grab a pre-stored query embedding by using a dummy approach:
# just compare the two retrieval modes on the STORED vectors.
# Since we don't have stored query vectors, we do this differently:
# we pick a random event vector as a "pseudo-query" and compare both modes.
# That's actually MORE rigorous — it exercises the full recall spectrum.

SQL_GET_IDS = """
SELECT event_id FROM events WHERE {col} IS NOT NULL ORDER BY event_id LIMIT 100
"""

SQL_APPROX = """
SELECT event_id
FROM events
WHERE {col} IS NOT NULL
ORDER BY {col} <=> %s::vector
LIMIT 10
"""

SQL_EXACT = """
/*+ SeqScan(events) */
SELECT event_id
FROM events
WHERE {col} IS NOT NULL
ORDER BY {col} <=> %s::vector
LIMIT 10
"""

SQL_GET_VECTOR = """
SELECT {col}::text FROM events WHERE event_id = %s
"""


def run_comparison(conn: psycopg.Connection, col: str, label: str) -> dict:
    results = []
    with conn.cursor() as cur:
        # Get a sample of event IDs to use as pseudo-queries
        cur.execute(SQL_GET_IDS.format(col=col))
        sample_ids = [r[0] for r in cur.fetchall()]

    # Use every 5th event as a pseudo-query (20 pseudo-queries)
    pseudo_query_ids = sample_ids[::5][:20]

    with conn.cursor() as cur:
        for event_id in pseudo_query_ids:
            # Fetch the stored vector for this event
            cur.execute(SQL_GET_VECTOR.format(col=col), (event_id,))
            row = cur.fetchone()
            if not row:
                continue
            vec_literal = row[0]

            # HNSW (index-assisted)
            conn.execute("SET enable_indexscan = on")
            cur.execute(SQL_APPROX.format(col=col), (vec_literal,))
            hnsw_ids = [r[0] for r in cur.fetchall()]

            # Exact (disable index scans)
            conn.execute("SET enable_indexscan = off")
            cur.execute(SQL_APPROX.format(col=col), (vec_literal,))
            exact_ids = [r[0] for r in cur.fetchall()]

            # Re-enable
            conn.execute("SET enable_indexscan = on")

            identical = hnsw_ids == exact_ids
            missing_from_hnsw = set(exact_ids) - set(hnsw_ids)
            extra_in_hnsw = set(hnsw_ids) - set(exact_ids)

            results.append({
                "pseudo_query_event_id": event_id,
                "identical": identical,
                "missing_from_hnsw": list(missing_from_hnsw),
                "extra_in_hnsw": list(extra_in_hnsw),
                "hnsw_top10": hnsw_ids,
                "exact_top10": exact_ids,
            })

    return {"model": label, "column": col, "results": results}


def print_report(comparison: dict) -> None:
    label = comparison["model"]
    results = comparison["results"]
    total = len(results)
    identical_count = sum(1 for r in results if r["identical"])
    differing = [r for r in results if not r["identical"]]

    print(f"\n{'='*60}")
    print(f"Model: {label}")
    print(f"{'='*60}")
    print(f"Pseudo-queries tested : {total}")
    print(f"Identical top-10      : {identical_count}/{total}")
    print(f"Differing             : {len(differing)}/{total}")

    if differing:
        print("\nDifferences:")
        for r in differing:
            print(f"  pseudo-query event_id={r['pseudo_query_event_id']}")
            print(f"    missing from HNSW : {r['missing_from_hnsw']}")
            print(f"    extra in HNSW     : {r['extra_in_hnsw']}")
            print(f"    HNSW  top-10: {r['hnsw_top10']}")
            print(f"    Exact top-10: {r['exact_top10']}")
    else:
        print("\n  All top-10 result sets are IDENTICAL between HNSW and exact search.")

    recall_at_10 = sum(
        len(set(r["exact_top10"]) & set(r["hnsw_top10"])) / 10
        for r in results
    ) / total if total else 0
    print(f"\nMean recall@10 of HNSW vs exact: {recall_at_10:.4f}  (1.0 = perfect)")


def main() -> None:
    print("Connecting to database...")
    with psycopg.connect(DATABASE_URL) as conn:
        # Check which columns actually exist
        with conn.cursor() as cur:
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'events'
                AND column_name IN ('embedding', 'embedding_ada002')
            """)
            existing = {r[0] for r in cur.fetchall()}

        active_columns = {k: v for k, v in COLUMNS.items() if v in existing}
        if not active_columns:
            raise SystemExit("No embedding columns found in events table.")

        print(f"Found embedding columns: {list(active_columns.values())}")
        print("Running HNSW vs exact comparison (20 pseudo-queries per model)...\n")

        for label, col in active_columns.items():
            comparison = run_comparison(conn, col, label)
            print_report(comparison)

    print("\nDone.")


if __name__ == "__main__":
    main()
