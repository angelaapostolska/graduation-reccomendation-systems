"""Adds start and end times to an existing database from data/tables/event_times.csv. Usage: python db/add_times.py"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import psycopg
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--times", default=str(ROOT / "data" / "tables" / "event_times.csv"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set")
    view = re.search(r"CREATE VIEW event_details AS.*?;\n", (ROOT / "db" / "schema.sql").read_text(encoding="utf-8"), re.S).group(0)

    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS start_time time")
        cur.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS end_time time")
        cur.execute("ALTER TABLE events ADD COLUMN IF NOT EXISTS time_source text")
        cur.execute("CREATE TEMP TABLE event_times (event_id text, start_time time, end_time time, time_source text) ON COMMIT DROP")
        with open(args.times, encoding="utf-8") as handle, cur.copy("COPY event_times FROM STDIN WITH (FORMAT csv, HEADER true)") as copy:
            while chunk := handle.read(1 << 20):
                copy.write(chunk)
        cur.execute("""UPDATE events e SET start_time = t.start_time, end_time = t.end_time, time_source = t.time_source
                       FROM event_times t WHERE e.event_id = t.event_id""")
        print(f"updated {cur.rowcount} events")
        cur.execute("DROP VIEW IF EXISTS event_details")
        cur.execute(view)
        cur.execute("SELECT time_source, count(*), min(start_time), max(start_time) FROM events GROUP BY 1 ORDER BY 1")
        for source, count, first, last in cur.fetchall():
            print(f"  {source}: {count} events, starts {first} to {last}")


if __name__ == "__main__":
    main()
