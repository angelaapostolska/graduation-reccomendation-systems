"""Splits events_final.csv into the normalized tables of the ontology (output: data/tables/)."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mapping"))
sys.path.insert(0, str(ROOT / "generator"))
from real_events import norm, read_csv, write_csv  # noqa: E402
from taxonomy import ALL_LOCATION_TYPES, EXTRA_LOCATION_TYPES, MUSIC_GENRES, PARTY_TYPES  # noqa: E402

EVENT_COLUMNS = [
    "event_id", "source_event_id", "origin", "is_synthetic", "name", "description", "description_source", "event_date",
    "venue_id", "entrance_mkd", "entrance_source", "theme", "dress_code", "promotion", "age_limit", "language",
    "includes", "genre_source", "synthetic_fields",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--venues", default=str(ROOT / "generator" / "data" / "venues.csv"))
    args = parser.parse_args()
    data = Path(args.data)
    out = data / "tables"
    out.mkdir(parents=True, exist_ok=True)

    events = read_csv(data / "events_final.csv")
    venues = read_csv(Path(args.venues))

    cities = {name: i for i, name in enumerate(sorted({v["city"] for v in venues}), 1)}
    types = {name: i for i, name in enumerate(ALL_LOCATION_TYPES, 1)}
    genres = {name: i for i, name in enumerate(MUSIC_GENRES, 1)}
    party_types = {name: i for i, name in enumerate(PARTY_TYPES, 1)}
    venue_ids = {v["venue_id"] for v in venues}

    write_csv(out / "cities.csv", [{"city_id": i, "name": n, "country": "Macedonia"} for n, i in cities.items()])
    write_csv(out / "business_types.csv", [
        {"type_id": i, "name": n, "is_project_addition": str(n in EXTRA_LOCATION_TYPES).lower()} for n, i in types.items()])
    write_csv(out / "genres.csv", [{"genre_id": i, "name": n} for n, i in genres.items()])
    write_csv(out / "party_types.csv", [{"party_type_id": i, "name": n} for n, i in party_types.items()])
    write_csv(out / "venues.csv", [
        {"venue_id": v["venue_id"], "name": v["name"], "city_id": cities[v["city"]], "instagram_url": v["instagram_url"],
         "is_fictional": "true" if v.get("is_fictional") == "1" else "false"} for v in venues])
    write_csv(out / "venue_types.csv", [
        {"venue_id": v["venue_id"], "type_id": types[t]} for v in venues for t in v["location_types"].split("|")])

    real_performers = {norm(r["name"]): r for r in read_csv(data / "real" / "performers.csv")}
    known_kind = {norm(r["name"]): r["kind"] for r in read_csv(data / "synthetic_gap" / "performers.csv")}
    appearances: dict[str, dict] = {}
    for e in events:
        for name in filter(None, e["performers"].split("|")):
            p = appearances.setdefault(norm(name), {"name": name, "events": 0, "standup": 0})
            p["events"] += 1
            p["standup"] += "Standup Comedy" in e["party_types"]
    ordered = sorted(appearances.items(), key=lambda kv: (-kv[1]["events"], kv[1]["name"]))
    performer_id = {key: f"per_{i:04d}" for i, (key, _) in enumerate(ordered, 1)}
    performer_rows = []
    for key, p in ordered:
        kind = real_performers.get(key, {}).get("kind") or known_kind.get(key) or ("comedian" if p["standup"] else "other")
        performer_rows.append({"performer_id": performer_id[key], "name": p["name"], "kind": kind,
                               "source": "findzzer_events" if key in real_performers else "synthetic"})
    write_csv(out / "performers.csv", performer_rows)

    event_rows, ev_genres, ev_parties, ev_performers = [], [], [], []
    performer_genre: dict[tuple[str, int], int] = Counter()
    for e in events:
        if e["venue_id"] not in venue_ids:
            raise ValueError(f"unknown venue {e['venue_id']}")
        row = {c: e.get(c, "") for c in EVENT_COLUMNS if c not in ("event_date", "is_synthetic", "entrance_mkd")}
        row.update(event_date=e["date"], is_synthetic="true" if e["is_synthetic"] == "1" else "false",
                   entrance_mkd=e["entrance_mkd"])
        event_rows.append({c: row[c] for c in EVENT_COLUMNS})
        names = [n for n in e["performers"].split("|") if n]
        for g in filter(None, e["music_genres"].split("|")):
            ev_genres.append({"event_id": e["event_id"], "genre_id": genres[g], "source": e["genre_source"]})
            if e["genre_source"] in ("real", "synthetic"):
                for n in names:
                    performer_genre[(performer_id[norm(n)], genres[g])] += 1
        for p in filter(None, e["party_types"].split("|")):
            ev_parties.append({"event_id": e["event_id"], "party_type_id": party_types[p]})
        for n in names:
            ev_performers.append({"event_id": e["event_id"], "performer_id": performer_id[norm(n)]})
    write_csv(out / "events.csv", event_rows)
    write_csv(out / "event_genres.csv", ev_genres)
    write_csv(out / "event_party_types.csv", ev_parties)
    write_csv(out / "event_performers.csv", sorted({(r["event_id"], r["performer_id"]): r for r in ev_performers}.values(),
                                                   key=lambda r: (r["event_id"], r["performer_id"])))
    write_csv(out / "performer_genres.csv", [
        {"performer_id": p, "genre_id": g, "event_count": c} for (p, g), c in sorted(performer_genre.items())])

    counts = {p.stem: len(read_csv(p)) for p in sorted(out.glob("*.csv"))}
    print("tables written to", out)
    for name, n in counts.items():
        print(f"  {name:20} {n:6} rows")


if __name__ == "__main__":
    main()
