"""Try one search for one user profile: plain ada-002 next to the hybrid result (date filter, intent guard, profile boost).
Usage: python hybrid/try_search.py "live jazz concert this weekend" --persona "pop and balkan fan" [--today 2026-01-16]
       python hybrid/try_search.py --list-profiles
Broad queries worth trying: "party tonight", "live music this weekend", "night out", "something fun on Friday", "concert", "good bar"."""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_boost import MODEL, PERSONAS, POOL, TOP, boost, load_features, read_users  # noqa: E402

DEFAULT_TODAY = "2026-01-16"  # a Friday inside the data, which has few events after October 2026
ALIASES = {"hip hop": "Hip-Hop", "hiphop": "Hip-Hop", "rnb": "R&B", "r and b": "R&B", "dj": "Dj Set", "stand up": "Standup Comedy",
           "stand-up": "Standup Comedy", "standup": "Standup Comedy", "comedy": "Standup Comedy"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&]+", " ", text.lower())).strip()


def named_in_query(query: str, genres: list[str], types: list[str]) -> tuple[set, set]:
    """Genres and event types the query names itself, from the ontology's own names plus a few spellings."""
    q = " " + norm(query) + " "
    found_genres, found_types = set(), set()
    for name in genres:
        if " " + norm(name) + " " in q:
            found_genres.add(name)
    for name in types:
        if " " + norm(name) + " " in q:
            found_types.add(name)
    for alias, target in ALIASES.items():
        if " " + norm(alias) + " " in q:
            (found_genres if target in genres else found_types).add(target)
    return found_genres, found_types


def date_window(query: str, today: dt.date) -> tuple[dt.date, dt.date, str] | None:
    q = norm(query)
    if re.search(r"\b(tonight|today)\b", q):
        return today, today, "today"
    if "tomorrow" in q:
        d = today + dt.timedelta(days=1)
        return d, d, "tomorrow"
    if re.search(r"\b(this )?weekend\b", q):
        friday = today + dt.timedelta(days=(4 - today.weekday()) % 7) if today.weekday() < 4 else today
        sunday = today + dt.timedelta(days=6 - today.weekday())
        return max(friday, today), sunday, "this weekend"
    return None


def guarded_profile(profile: dict, named_genres: set, named_types: set) -> tuple[dict, list[str]]:
    """The query's own words win: parts of the profile that the query already names are left out of the boost."""
    notes, out = [], dict(profile)
    if named_genres:
        out["music"] = []
        notes.append(f"query names the genre {sorted(named_genres)}, so profile genres are ignored")
    if named_types:
        out["types"] = []
        notes.append(f"query names the event type {sorted(named_types)}, so profile party types are ignored")
    return out, notes


def run(query: str, profile: dict, features: dict, scores: np.ndarray, weight: float, today: dt.date, use_dates: bool, use_guard: bool) -> dict:
    events = features["events"]
    genres = [n for n, _ in sorted(features["genre_index"].items(), key=lambda x: x[1])]
    types = [n for n, _ in sorted(features["type_index"].items(), key=lambda x: x[1])]
    notes = []
    allowed = np.ones(len(events), dtype=bool)
    window = date_window(query, today) if use_dates else None
    if window:
        start, end, label = window
        dates = pd.to_datetime(events["event_date"]).dt.date
        allowed = ((dates >= start) & (dates <= end)).to_numpy()
        notes.append(f"date filter '{label}': {start} to {end}, {int(allowed.sum())} events remain")
    used = profile
    if use_guard:
        used, guard_notes = guarded_profile(profile, *named_in_query(query, genres, types))
        notes += guard_notes
    values = boost(features, used)
    plain_order = np.argsort(-scores)
    plain_rank = {int(e): r + 1 for r, e in enumerate(plain_order)}
    pool = np.array([e for e in plain_order if allowed[e]][:POOL])
    if len(pool) == 0:
        return {"notes": notes + ["no event passes the filters"], "plain": [], "hybrid": []}
    low, high = scores[pool].min(), scores[pool].max()
    scaled = (scores[pool] - low) / (high - low) if high > low else np.zeros(len(pool))
    hybrid = [int(pool[i]) for i in np.argsort(-(scaled + weight * values[pool]), kind="stable")[:TOP]]
    plain = [int(e) for e in plain_order[:TOP]]

    def row(e: int) -> dict:
        r = events.iloc[e]
        return {"name": r["name"], "date": r["event_date"], "city": r.get("city", ""), "cosine": float(scores[e]), "boost": float(values[e]),
                "plain_rank": plain_rank[e], "passes_filters": bool(allowed[e])}

    return {"notes": notes, "plain": [row(e) for e in plain], "hybrid": [row(e) for e in hybrid]}


def show(result: dict, query: str, profile_name: str, weight: float) -> None:
    print(f'\nquery: "{query}"   profile: {profile_name}   boost weight: {weight}')
    for note in result["notes"]:
        print("  note:", note)
    plain_names = {(r["name"], r["date"]) for r in result["plain"]}
    print(f'\n{"PLAIN ada-002":<58}| HYBRID')
    for i in range(max(len(result["plain"]), len(result["hybrid"]))):
        left = result["plain"][i] if i < len(result["plain"]) else None
        right = result["hybrid"][i] if i < len(result["hybrid"]) else None
        l = f'{i + 1:>2}. {left["name"][:34]} {left["date"]}{"" if left["passes_filters"] else " (fails date filter)"}' if left else ""
        r = ""
        if right:
            mark = "" if (right["name"], right["date"]) in plain_names else f'  NEW (was #{right["plain_rank"]}, boost {right["boost"]:.2f})'
            r = f'{i + 1:>2}. {right["name"][:34]} {right["date"]}{mark}'
        print(f"{l:<58}| {r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="the search text; leave out to type queries one by one")
    parser.add_argument("--persona", default="", help="name of a hand-made profile (see --list-profiles)")
    parser.add_argument("--real", type=int, default=0, help="number of a real onboarding profile, 1 to 100, taken from --users")
    parser.add_argument("--users", default="", help="CSV with music_genres and party_types columns (stays on your computer)")
    parser.add_argument("--weight", type=float, default=0.2, help="boost weight, 0.1 to 0.2 keeps the query in charge")
    parser.add_argument("--today", default="", help=f"reference date for words like tonight or this weekend (default {DEFAULT_TODAY})")
    parser.add_argument("--no-dates", action="store_true", help="switch the date filter off")
    parser.add_argument("--no-guard", action="store_true", help="switch the intent guard off")
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--list-profiles", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    profiles = dict(PERSONAS)
    if args.users:
        real, _ = read_users(args.users, 100, 15, 7)
        profiles.update(real)
    if args.list_profiles:
        for name, p in profiles.items():
            print(f"{name}: music {p['music']}, party types {p['types']}")
        return
    chosen = args.persona or (f"real profile {args.real}" if args.real else "pop and balkan fan")
    if chosen not in profiles:
        raise SystemExit(f"unknown profile '{chosen}'; use --list-profiles" + ("" if args.users or not args.real else " (add --users for real profiles)"))
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in .env")
    from openai import OpenAI

    features = load_features(args.tables)
    cities = pd.read_csv(f"{args.tables}/cities.csv").set_index("city_id")["name"]
    venues = pd.read_csv(f"{args.tables}/venues.csv").set_index("venue_id")["city_id"]
    features["events"]["city"] = features["events"]["venue_id"].map(venues).map(cities)
    vectors = np.load(ROOT / "data" / "graph" / "event_text.npy").astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    assert len(vectors) == len(features["events"]), "events.csv and the saved vectors do not match"
    today = dt.date.fromisoformat(args.today or DEFAULT_TODAY)
    client = OpenAI()

    def search(text: str) -> None:
        q = np.array(client.embeddings.create(model=MODEL, input=[text]).data[0].embedding, dtype=np.float32)
        q /= np.linalg.norm(q)
        show(run(text, profiles[chosen], features, vectors @ q, args.weight, today, not args.no_dates, not args.no_guard), text, chosen, args.weight)

    if args.query:
        search(args.query)
    else:
        while text := input("\nsearch (empty to quit): ").strip():
            search(text)


if __name__ == "__main__":
    main()
