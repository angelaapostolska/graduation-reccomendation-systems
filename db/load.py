"""Creates the schema and loads data/tables/*.csv into Postgres. Usage: python db/load.py [--reset]"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
LOAD_ORDER = [
    "cities", "business_types", "venues", "venue_types", "genres", "party_types", "performers", "performer_genres",
    "events", "event_genres", "event_party_types", "event_performers",
]


def load_table(cur: psycopg.Cursor, table: str, path: Path) -> None:
    with path.open(encoding="utf-8") as handle:
        columns = handle.readline().strip().split(",")
        handle.seek(0)
        with cur.copy(f"COPY {table} ({', '.join(columns)}) FROM STDIN WITH (FORMAT csv, HEADER true, NULL '')") as copy:
            while chunk := handle.read(1 << 20):
                copy.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--reset", action="store_true", help="drop all existing tables first")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set (see .env.example)")

    with psycopg.connect(url, autocommit=True) as conn, conn.cursor() as cur:
        if args.reset:
            cur.execute("DROP VIEW IF EXISTS event_details")
            for table in reversed(LOAD_ORDER):
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        cur.execute((ROOT / "db" / "schema.sql").read_text(encoding="utf-8"))
        for table in LOAD_ORDER:
            load_table(cur, table, Path(args.tables) / f"{table}.csv")
        print("loaded:")
        for table in LOAD_ORDER:
            cur.execute(f"SELECT count(*) FROM {table}")
            print(f"  {table:20} {cur.fetchone()[0]:6}")


if __name__ == "__main__":
    main()
