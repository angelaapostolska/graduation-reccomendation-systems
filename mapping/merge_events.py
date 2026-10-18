"""Merges real events and the two synthetic sets into one events file."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mapping"))
from real_events import read_csv, write_csv  # noqa: E402

COLUMNS = [
    "event_id", "source_event_id", "origin", "is_synthetic", "name", "description", "description_source", "date", "city",
    "venue_id", "venue_name", "location_types", "music_genres", "genre_source", "party_types", "performers", "theme",
    "dress_code", "entrance", "entrance_mkd", "entrance_source", "promotion", "age_limit", "language", "includes",
    "synthetic_fields",
]
OPTIONAL = ("theme", "entrance", "promotion", "language", "includes", "age_limit")


def entrance_mkd(text: str) -> str:
    if text == "Free entry":
        return "0"
    match = re.search(r"(\d+)\s*MKD", text)
    return match.group(1) if match else ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data"))
    args = parser.parse_args()
    data = Path(args.data)
    rows = []
    for r in read_csv(data / "real" / "events.csv"):
        r.update(origin="real", is_synthetic="0")
        rows.append(r)
    for path, origin in ((data / "synthetic" / "proposal_events_synthetic.csv", "synthetic_nonmusic"),
                         (data / "synthetic_gap" / "proposal_events_gap.csv", "synthetic_gap")):
        for r in read_csv(path):
            r.update(origin=origin, is_synthetic="1", genre_source="synthetic" if r.get("music_genres") else "none")
            if origin == "synthetic_nonmusic":
                r.update(entrance_mkd=entrance_mkd(r["entrance"]), entrance_source="synthetic" if r["entrance"] else "",
                         synthetic_fields="|".join(k for k in OPTIONAL if r.get(k)))
            rows.append(r)
    rows.sort(key=lambda r: (r["date"], r["origin"], r["source_event_id"]))
    for i, r in enumerate(rows, 1):
        r["event_id"] = f"evt_{i:05d}"
    write_csv(data / "events_final.csv", [{c: r.get(c, "") for c in COLUMNS} for r in rows])
    print(f"{len(rows)} events -> {data / 'events_final.csv'}")
    print("origin:", dict(Counter(r["origin"] for r in rows)))


if __name__ == "__main__":
    main()
