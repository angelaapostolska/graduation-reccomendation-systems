"""Start and end times per event: real ones from the Findzzer dump, generated ones for synthetic events.
Usage: python mapping/event_times.py --dump forge.sql.gz [--events data/events_final.csv] [--out data/tables/event_times.csv]"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mapping"))
from enrichment import _rng  # noqa: E402
from real_events import read_dump  # noqa: E402

# (start hour, duration hours, weight); duration may run past midnight
WORKSHOP = [(18.0, 2.0, 30), (18.5, 2.5, 20), (10.0, 3.0, 25), (11.0, 3.0, 15), (14.0, 3.0, 10)]
EVENING_TALK = [(18.5, 2.0, 35), (19.0, 2.0, 35), (19.0, 2.5, 20), (17.5, 2.0, 10)]
SHOW = [(19.0, 2.0, 15), (19.5, 2.0, 30), (20.0, 2.0, 35), (20.5, 2.0, 20)]
STANDUP = [(20.0, 2.0, 20), (20.5, 2.0, 40), (21.0, 2.0, 40)]
BEACH = [(14.0, 5.0, 25), (16.0, 5.0, 25), (18.0, 5.0, 25), (21.0, 5.0, 25)]
RULES = [
    (r"kids|children|child", [(10.0, 2.0, 50), (11.0, 2.5, 30), (16.0, 2.0, 20)]),
    (r"brunch|breakfast", [(11.0, 3.0, 40), (12.0, 3.0, 40), (13.0, 3.0, 20)]),
    (r"buffet|all you can eat|lunch", [(12.0, 3.0, 50), (16.0, 3.0, 25), (17.0, 3.0, 25)]),
    (r"beach|pool", BEACH),
    (r"hackathon", [(9.0, 10.0, 100)]),
    (r"open day|open doors", [(10.0, 6.0, 60), (11.0, 5.0, 40)]),
    (r"market|fair\b", [(10.0, 7.0, 50), (11.0, 7.0, 50)]),
    (r"coffee", [(9.5, 1.5, 60), (10.0, 1.5, 40)]),
    (r"conference|summit|forum|demo day|thesis|day$", [(9.5, 7.0, 50), (10.0, 6.0, 50)]),
    (r"charity|fundrais", [(19.0, 3.0, 100)]),
    (r"quiz|board game", [(19.5, 3.0, 50), (20.0, 2.5, 50)]),
    (r"stand ?up|comedy", STANDUP),
    (r"poetry|spoken|film|play|theat|monodrama", SHOW),
    (r"meetup|mixer|networking|alumni|language exchange|women in tech|founders|gathering", EVENING_TALK),
    (r"workshop|basics|beginners|101|intro|making|printing|pottery|skills|seo|python|excel|figma", WORKSHOP),
]
BY_PARTY = {
    "Concert": [(20.0, 3.0, 20), (20.5, 3.0, 40), (21.0, 3.0, 40)],
    "Festival": [(14.0, 10.0, 30), (15.0, 9.0, 40), (17.0, 7.0, 30)],
    "Live Music": [(20.0, 3.0, 30), (21.0, 3.0, 50), (21.5, 3.0, 20)],
    "Disco": [(22.0, 5.0, 40), (23.0, 5.0, 60)],
    "Dj Set": [(22.0, 4.0, 40), (23.0, 5.0, 60)],
    "Workshop": WORKSHOP,
    "Convention": [(9.5, 7.0, 50), (10.0, 6.0, 50)],
    "Networking": EVENING_TALK,
    "Open Day": [(10.0, 6.0, 60), (11.0, 5.0, 40)],
    "Standup Comedy": STANDUP,
    "Poetry": SHOW,
    "Theater": SHOW,
    "Handmade Market": [(10.0, 7.0, 100)],
    "University Event": [(10.0, 5.0, 100)],
    "Social Gathering": EVENING_TALK,
    "Charity": [(19.0, 3.0, 100)],
}


def clock(hours: float) -> str:
    minutes = round(hours * 60) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def synthetic_times(event: dict) -> tuple[str, str]:
    rng = _rng(event["event_id"], "time")
    name = event["name"].lower()
    options = next((opts for pattern, opts in RULES if re.search(pattern, name)), None)
    if options is None and "Beach Bar" in event["location_types"]:
        options = BEACH
    if options is None:
        options = next((BY_PARTY[p] for p in event["party_types"].split("|") if p in BY_PARTY), [(18.0, 2.0, 100)])
    start, duration, _ = rng.choices(options, weights=[o[2] for o in options])[0]
    return clock(start), clock(start + duration)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="the Findzzer pg_dump (.sql or .sql.gz)")
    parser.add_argument("--events", default=str(ROOT / "data" / "events_final.csv"))
    parser.add_argument("--out", default=str(ROOT / "data" / "tables" / "event_times.csv"))
    args = parser.parse_args()
    source = {r["id"]: r for r in read_dump(Path(args.dump))["events"]}
    rows, missing = [], 0
    with open(args.events, encoding="utf-8", newline="") as handle:
        for event in csv.DictReader(handle):
            if event["origin"] == "real":
                original = source.get(event["source_event_id"])
                if not original or not original.get("start") or not original.get("end"):
                    missing += 1
                    continue
                rows.append([event["event_id"], original["start"][11:16], original["end"][11:16], "findzzer"])
            else:
                rows.append([event["event_id"], *synthetic_times(event), "synthetic"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "start_time", "end_time", "time_source"])
        writer.writerows(rows)
    print(f"{len(rows)} events with times ({missing} real events without times) -> {args.out}")


if __name__ == "__main__":
    main()
