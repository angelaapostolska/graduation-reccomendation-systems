"""Synthetic music events that top up thin categories: concert venues and rare genres."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from argparse import Namespace
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mapping"))
sys.path.insert(0, str(ROOT / "generator"))
import generate_events as G  # noqa: E402
import real_events as R  # noqa: E402
from synthetic_events import FEMALE, FIRST, LAST_F, LAST_M  # noqa: E402

DATE_RANGE = (date(2025, 7, 1), date(2026, 10, 31))
MIN_CONCERT_VENUE_EVENTS = 20
MIN_GENRE_EVENTS = 15
RARE_GENRES = ["Ska", "Dancehall", "Country", "Salsa", "Trance", "Metal", "Reggae", "Folk"]
GENRE_VENUE_TYPES = {
    "Ska": {"Bar", "Pub"}, "Dancehall": {"Nightclub", "Beach Bar"}, "Country": {"Pub", "Bar", "Restaurant"},
    "Salsa": {"Bar", "Restaurant", "Nightclub"}, "Trance": {"Nightclub"}, "Metal": {"Bar", "Pub", "Concert Venue"},
    "Reggae": {"Beach Bar", "Bar"}, "Folk": {"Restaurant", "Bar", "Pub"},
}
GENRE_KINDS = {"Dancehall": ["dj", "solo"], "Trance": ["dj"]}
GENRE_PARTY = {"Dancehall": ["Dj Set", "Disco"], "Trance": ["Dj Set", "Disco"]}
LIVE_PARTY = ["Live Music", "Concert"]
KIND_MAP = {"duo": "band", "trio": "band", "collective": "band"}


def event_date(rng: random.Random) -> str:
    for _ in range(50):
        d = DATE_RANGE[0] + timedelta(days=rng.randint(0, (DATE_RANGE[1] - DATE_RANGE[0]).days))
        if d.weekday() in (3, 4, 5, 6):
            return d.isoformat()
    return d.isoformat()


def new_performer(rng: random.Random, genre: str, kind: str, used: set[str], serial: int) -> G.Performer:
    for _ in range(200):
        if kind == "dj":
            name = f"DJ {rng.choice(G.DJ_HANDLES)} {rng.choice(LAST_M)[0]}."
        elif kind == "band":
            name = f"{rng.choice(G.BAND_WORDS_A)} {rng.choice(G.BAND_WORDS_B)}"
        else:
            first = rng.choice(FIRST)
            name = f"{first} {rng.choice(LAST_F if first in FEMALE else LAST_M)}"
        if name not in used:
            used.add(name)
            return G.Performer(f"per_s{serial:04d}", name, kind, (genre,), "fictional")
    raise RuntimeError("no unique performer name")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data" / "synthetic_gap"))
    parser.add_argument("--real", default=str(ROOT / "data" / "real"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    out, real = Path(args.out), Path(args.real)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    random.seed(args.seed)

    venues = G.load_venues()
    real_events = R.read_csv(real / "events.csv")
    real_performers = R.read_csv(real / "performers.csv")
    pool = [G.Performer(r["performer_id"], r["name"], KIND_MAP.get(r["kind"], r["kind"]),
                        tuple(g for g in r["genres"].split("|") if g), "real_sample")
            for r in real_performers if int(r["events"]) >= 2]
    used_names = {p.name for p in pool}
    new_performers: list[G.Performer] = []
    combos = [c for c in G.load_weighted_combinations() if c.music_genre != "Gig"]

    def performer_for(genre: str, party: list[str]) -> G.Performer | None:
        kinds = GENRE_KINDS.get(genre)
        found = None
        if not kinds:
            found = G.pick_performer(genre, party, pool + new_performers)
        if found and found.source == "real_sample":
            return found
        kind_sets = [G.PARTY_TYPE_PERFORMER_KINDS[p] for p in party if p in G.PARTY_TYPE_PERFORMER_KINDS]
        if not kind_sets:
            return None
        allowed = set.intersection(*kind_sets) or kind_sets[0]
        kind = rng.choice(sorted(set(kinds or allowed) & allowed or allowed))
        existing = [p for p in new_performers if genre in p.genres and p.kind == kind]
        if len(existing) >= 3:
            return rng.choice(existing)
        p = new_performer(rng, genre, kind, used_names, len(new_performers) + 1)
        new_performers.append(p)
        return p

    rows, links = [], []

    def add(venue: G.Venue, genre: str, party_types: list[str]) -> None:
        performer = performer_for(genre, party_types)
        sid = f"syn_g_{len(rows) + 1:05d}"
        rows.append({
            "source_event_id": sid, "name_original": performer.name if performer else f"{genre} Night",
            "date": event_date(rng), "venue_id": venue.venue_id, "venue_name": venue.name, "city": venue.city,
            "location_types": "|".join(venue.location_types), "music_genres": genre, "genre_status": "synthetic",
            "party_types": "|".join(party_types), "details_original": "", "entrance_price_real": "",
        })
        if performer:
            links.append({"source_event_id": sid, "performer_id": performer.performer_id})

    concert_venues = [v for v in venues if "Concert Venue" in v.location_types]
    have = Counter(e["venue_id"] for e in real_events)
    for venue in concert_venues:
        for _ in range(max(0, MIN_CONCERT_VENUE_EVENTS - have[venue.venue_id])):
            party = ["Festival"] if "Fest" in venue.name and rng.random() < 0.6 else [rng.choice(["Concert", "Concert", "Live Music"])]
            options = [c for c in combos if c.party_type in set(party) | {"Live Music"} and c.music_genre not in RARE_GENRES]
            add(venue, G.pick_weighted(options).music_genre, party)

    genre_have = Counter(g for e in real_events for g in e["music_genres"].split("|") if g)
    for genre in RARE_GENRES:
        need = max(0, MIN_GENRE_EVENTS - genre_have[genre] - sum(1 for r in rows if r["music_genres"] == genre))
        candidates = [v for v in venues if set(v.location_types) & GENRE_VENUE_TYPES[genre] and not v.venue_id.startswith("ven_10")]
        for _ in range(need):
            venue = rng.choice(candidates)
            allowed = G.allowed_party_types(venue)
            party = [p for p in GENRE_PARTY.get(genre, LIVE_PARTY) if p in allowed] or LIVE_PARTY
            add(venue, genre, [rng.choice(party)])

    rows.sort(key=lambda r: (r["date"], r["source_event_id"]))
    R.write_csv(out / "events_base.csv", rows)
    perf_rows = [{"performer_id": r["performer_id"], "name": r["name"], "kind": r["kind"], "genres": r["genres"], "events": r["events"]}
                 for r in real_performers]
    perf_rows += [{"performer_id": p.performer_id, "name": p.name, "kind": p.kind, "genres": "|".join(p.genres), "events": 0}
                  for p in new_performers]
    R.write_csv(out / "performers.csv", perf_rows)
    R.write_csv(out / "event_performers.csv", links)
    print(f"{len(rows)} events planned; new fictional performers: {len(new_performers)}")
    print("by venue:", dict(Counter(r["venue_name"] for r in rows)))
    print("by genre:", dict(Counter(r["music_genres"] for r in rows).most_common(20)))

    R.stage_descriptions(Namespace(out=str(out), sample=0, limit=0))
    final = R.read_csv(out / "events.csv")
    for r in final:
        r.update(genre_source="synthetic", is_synthetic="1")
    R.write_csv(out / "proposal_events_gap.csv", final)
    (out / "events.csv").unlink()


if __name__ == "__main__":
    main()
