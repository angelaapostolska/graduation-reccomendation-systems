"""Synthetic event generator: venue first, then plausible party type, genres, performer and description."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from taxonomy import ALL_LOCATION_TYPES, MUSIC_GENRES

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
DISTRIBUTION_CSVS = [
    HERE / "genre_party_type_distribution.csv",
    HERE / "genre_party_type_distribution_extra.csv",
]
VENUES_CSV = DATA_DIR / "venues.csv"
PERFORMERS_CSV = DATA_DIR / "performers.csv"

EXCLUDED_PARTY_TYPES = {"Private Party"}

DEFAULT_PARTY_TYPES = {"Bar", "Live Music", "Dj Set", "Concert"}
VENUE_TYPE_PARTY_TYPES: dict[str, set[str]] = {
    "Bar": {"Bar", "Caffe Bar", "Live Music", "Karaoke Night", "Dj Set", "Concert", "Disco"},
    "Pub": {"Bar", "Live Music", "Karaoke Night", "Dj Set", "Concert"},
    "Nightclub": {"Disco", "Dj Set", "Concert", "Live Music"},
    "Cafe": {"Caffe Bar", "Live Music", "Dj Set"},
    "Restaurant": {"Restaurant", "Live Music", "Karaoke Night", "Dj Set"},
    "Concert Venue": {"Concert", "Festival", "Live Music"},
    "Theater": {"Theater", "Standup Comedy", "Concert"},
    "Workshop Space": {"Workshop", "Social Gathering"},
    "Beach Bar": {"Beach Party", "Dj Set", "Live Music", "Bar", "Caffe Bar", "Concert"},
}

PARTY_TYPE_PERFORMER_KINDS: dict[str, set[str]] = {
    "Dj Set": {"dj"},
    "Disco": {"dj"},
    "Silent Disco": {"dj"},
    "Concert": {"band", "solo"},
    "Live Music": {"band", "solo"},
    "Festival": {"band", "solo", "dj"},
    "New Year Party": {"band", "solo", "dj"},
}
SECOND_PARTY_TYPE_PROBABILITY = 0.25
SECOND_GENRE_PROBABILITY = 0.3
REAL_PERFORMER_WEIGHT = 3

ELECTRONIC_GENRES = {"Techno", "House", "Trance", "Electronic", "Afro House", "Disco", "Vinyl"}

MK_FIRST_NAMES = [
    "Aleksandar", "Bojan", "Darko", "Filip", "Goran", "Igor", "Kristijan", "Marko",
    "Nikola", "Stefan", "Viktor", "Ana", "Biljana", "Elena", "Jovana", "Katerina",
    "Marija", "Simona", "Tamara", "Teodora",
]
MK_LAST_NAMES = [
    "Stojanovski", "Petrovski", "Trajkovski", "Nikolovski", "Ilievski", "Georgievski",
    "Popovski", "Dimitrovski", "Angelovski", "Ristovski", "Atanasovska", "Jovanovska",
    "Mitrevska", "Velkovska",
]
BAND_WORDS_A = ["Midnight", "Velvet", "Rusty", "Silent", "Golden", "Lake", "Paper", "Blue", "Iron", "Last"]
BAND_WORDS_B = ["Echoes", "Strings", "Collective", "Brothers", "Orchestra", "Parade", "Section", "Project", "Wolves"]
DJ_HANDLES = ["Mark K", "Vlado", "Nexa", "Dino", "Sasha M", "Tijana", "Kiro", "Luna B", "Ognen", "Mila"]

GENERATION_PROMPT = """
You write event listings for Findzzer, a nightlife and culture events app in
North Macedonia. Write in English only.

STYLE (match these real listings; they show tone and sentence shape only, do not
reuse their facts):
  Name: DJ Cena
  Description: Dance into the night with a DJ set by DJ Cena at Beer Garden, featuring late-night club lighting and energetic beats.

  Name: OVERTONE
  Description: Live music performance by Overtone at Octava Lounge Bar featuring a band performance against a brick wall backdrop with musical instruments decor.

  Name: Jam Session Vol. 1
  Description: The first Jam Session at Street Pub, this time with three local musicians. Singers, instrumentalists and musicians of any kind are welcome to join.

RULES:
- Factual and plain. 1 or 2 sentences, under 220 characters.
- Use only the facts below. Do not invent prices, start times, guest names,
  sponsors, food or drink offers, or anything about the venue beyond its name,
  type and city.
- Mention the venue by name and the performer by name when one is given.
- Weave the music genre and event type into the sentence naturally; do not just
  list them. Mention the venue type only if it reads naturally, and never
  contradict it.
- Banned: "Expect", "Join us", "unforgettable", "perfect for", "vibrant",
  "energetic atmosphere", "lively night", exclamation marks, emoji, hashtags.
- Name: if a performer is given, the name is exactly the performer name.
  Otherwise write a short descriptive title of at most 4 words, such as
  "Hip-Hop Night" or "Karaoke Night". The name must not be the venue name.

FACTS (fixed, do not change):
- Performer: {performer_name}
- Music genre(s): {genres}
- Event type(s): {party_types}
- Venue: {venue_name}
- Venue type: {location_type}
- City: {city}

Return ONLY valid JSON:
{{"name": "string", "description": "string"}}
""".strip()


@dataclass(frozen=True)
class GenreWeight:
    music_genre: str
    party_type: str
    weight: float


@dataclass(frozen=True)
class Venue:
    venue_id: str
    name: str
    instagram_url: str
    location_types: tuple[str, ...]
    city: str
    country: str


@dataclass(frozen=True)
class Performer:
    performer_id: str
    name: str
    kind: str
    genres: tuple[str, ...]
    source: str  # "real_sample" or "fictional"


@dataclass
class GeneratedEvent:
    event_id: str
    name: str
    description: str
    city: str
    venue_id: str
    venue_name: str
    location_type: str
    music_genres: list[str]
    party_types: list[str]
    performer_id: Optional[str]
    performer_name: Optional[str]


# data


def load_weighted_combinations() -> list[GenreWeight]:
    combos: list[GenreWeight] = []
    for path in DISTRIBUTION_CSVS:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["party_type"] in EXCLUDED_PARTY_TYPES:
                    continue
                combos.append(GenreWeight(row["music_genre"], row["party_type"], float(row["percentage"])))
    return combos


def load_venues() -> list[Venue]:
    with VENUES_CSV.open(newline="", encoding="utf-8") as handle:
        venues = [
            Venue(r["venue_id"], r["name"], r["instagram_url"], tuple(r["location_types"].split("|")), r["city"], r["country"])
            for r in csv.DictReader(handle)
        ]
    unknown = {t for v in venues for t in v.location_types} - set(ALL_LOCATION_TYPES)
    if unknown:
        raise ValueError(f"venues.csv uses location types missing from taxonomy: {sorted(unknown)}")
    return venues


def load_real_performers() -> list[Performer]:
    with PERFORMERS_CSV.open(newline="", encoding="utf-8") as handle:
        return [
            Performer(r["performer_id"], r["name"], r["kind"], tuple(r["genres"].split("|")), r["source"])
            for r in csv.DictReader(handle)
        ]


def build_fictional_performers(per_genre_and_kind: int = 2) -> list[Performer]:
    performers: list[Performer] = []
    used: set[str] = set()

    def new_name(kind: str) -> str:
        for _ in range(500):
            if kind == "dj":
                name = f"DJ {random.choice(DJ_HANDLES)} {random.choice(MK_LAST_NAMES)[0]}."
            elif kind == "band":
                name = f"{random.choice(BAND_WORDS_A)} {random.choice(BAND_WORDS_B)}"
            else:
                name = f"{random.choice(MK_FIRST_NAMES)} {random.choice(MK_LAST_NAMES)}"
            if name not in used:
                used.add(name)
                return name
        raise RuntimeError("could not build a unique fictional performer name")

    genres = [g for g in MUSIC_GENRES if g not in {"Gospel", "Bluegrass", "Ska", "Opera", "Classical"}]
    for genre in genres:
        kinds = ["dj"] if genre in ELECTRONIC_GENRES else ["dj", "band", "solo"]
        for kind in kinds:
            for _ in range(per_genre_and_kind):
                performers.append(Performer(f"per_f{len(performers) + 1:03d}", new_name(kind), kind, (genre,), "fictional"))
    return performers


# sampling


def allowed_party_types(venue: Venue) -> set[str]:
    allowed: set[str] = set()
    for location_type in venue.location_types:
        allowed |= VENUE_TYPE_PARTY_TYPES.get(location_type, DEFAULT_PARTY_TYPES)
    return allowed


def location_type_for(venue: Venue, party_type: str) -> str:
    for location_type in venue.location_types:
        if party_type in VENUE_TYPE_PARTY_TYPES.get(location_type, DEFAULT_PARTY_TYPES):
            return location_type
    return venue.location_types[0]


def pick_weighted(combos: list[GenreWeight]) -> GenreWeight:
    return random.choices(combos, weights=[c.weight for c in combos], k=1)[0]


def pick_second_genre(primary: str, party_type: str, combos: list[GenreWeight]) -> list[str]:
    if random.random() > SECOND_GENRE_PROBABILITY:
        return [primary]
    others = [c for c in combos if c.party_type == party_type and c.music_genre != primary]
    if not others:
        return [primary]
    return [primary, pick_weighted(others).music_genre]


def pick_performer(genre: str, party_types: list[str], pool: list[Performer]) -> Optional[Performer]:
    kind_sets = [PARTY_TYPE_PERFORMER_KINDS[p] for p in party_types if p in PARTY_TYPE_PERFORMER_KINDS]
    if not kind_sets:
        return None
    kinds = set.intersection(*kind_sets) or kind_sets[0]
    matching = [p for p in pool if genre in p.genres and p.kind in kinds]
    if not matching:
        return None
    weights = [REAL_PERFORMER_WEIGHT if p.source == "real_sample" else 1 for p in matching]
    return random.choices(matching, weights=weights, k=1)[0]


def build_generation_prompt(
    *, performer: Optional[Performer], genres: list[str], party_types: list[str],
    venue: Venue, location_type: str,
) -> str:
    return GENERATION_PROMPT.format(
        performer_name=performer.name if performer else "none (no named performer)",
        genres=", ".join(genres),
        party_types=", ".join(party_types),
        venue_name=venue.name,
        location_type=location_type,
        city=venue.city,
    )


def generate_one_event(
    event_index: int,
    combos: list[GenreWeight],
    venues: list[Venue],
    performers: list[Performer],
    *,
    call_llm: Optional[Callable[[str], str]] = None,
) -> GeneratedEvent:
    venue = random.choice(venues)
    allowed = allowed_party_types(venue)
    venue_combos = [c for c in combos if c.party_type in allowed]

    combo = pick_weighted(venue_combos)
    party_types = [combo.party_type]
    if random.random() < SECOND_PARTY_TYPE_PROBABILITY:
        extras = [c for c in venue_combos if c.music_genre == combo.music_genre and c.party_type != combo.party_type]
        if extras:
            party_types.append(pick_weighted(extras).party_type)
    genres = pick_second_genre(combo.music_genre, combo.party_type, venue_combos)
    location_type = location_type_for(venue, party_types[0])

    performer = pick_performer(combo.music_genre, party_types, performers)

    prompt = build_generation_prompt(
        performer=performer, genres=genres, party_types=party_types,
        venue=venue, location_type=location_type,
    )
    if call_llm is not None:
        payload = _parse_json_response(call_llm(prompt))
        name, description = payload["name"], payload["description"]
    else:
        name, description = _deterministic_fallback_text(performer, genres, party_types, venue)

    return GeneratedEvent(
        event_id=f"evt_{event_index:05d}", name=name, description=description,
        city=venue.city, venue_id=venue.venue_id, venue_name=venue.name,
        location_type=location_type, music_genres=genres, party_types=party_types,
        performer_id=performer.performer_id if performer else None,
        performer_name=performer.name if performer else None,
    )


def _parse_json_response(raw_text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.I | re.M)
    return json.loads(text)


def _deterministic_fallback_text(
    performer: Optional[Performer], genres: list[str], party_types: list[str], venue: Venue
) -> tuple[str, str]:
    name = performer.name if performer else f"{genres[0]} {party_types[0]}"
    description = f"{', '.join(genres)} {party_types[0].lower()} at {venue.name}, {venue.city}"
    description += f", featuring {performer.name}." if performer else "."
    return name, description


def generate_dataset(
    count: int, *, call_llm: Optional[Callable[[str], str]] = None, seed: Optional[int] = None,
) -> list[GeneratedEvent]:
    if seed is not None:
        random.seed(seed)
    combos = load_weighted_combinations()
    venues = load_venues()
    performers = load_real_performers() + build_fictional_performers()
    return [generate_one_event(i, combos, venues, performers, call_llm=call_llm) for i in range(count)]


def validate_events(events: list[GeneratedEvent]) -> list[str]:
    venues = {v.venue_id: v for v in load_venues()}
    problems = []
    for e in events:
        v = venues[e.venue_id]
        if (e.venue_name, e.city) != (v.name, v.city):
            problems.append(f"{e.event_id}: venue/city mismatch")
        if e.location_type not in v.location_types:
            problems.append(f"{e.event_id}: {e.location_type} is not a type of {v.name}")
        if not set(e.party_types) <= allowed_party_types(v):
            problems.append(f"{e.event_id}: party types {e.party_types} not plausible at {v.name}")
        if "Private Party" in e.party_types:
            problems.append(f"{e.event_id}: Private Party")
        if e.performer_name and e.performer_name.lower() == e.venue_name.lower():
            problems.append(f"{e.event_id}: performer equals venue name")
    return problems


COLUMNS = [
    "event_id", "name", "description", "city", "venue_id", "venue_name", "location_type",
    "music_genres", "party_types", "performer_id", "performer_name",
]


def write_events_csv(events: list[GeneratedEvent], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for e in events:
            writer.writerow([
                e.event_id, e.name, e.description, e.city, e.venue_id, e.venue_name,
                e.location_type, "|".join(e.music_genres), "|".join(e.party_types),
                e.performer_id or "", e.performer_name or "",
            ])


def build_openai_call(*, model: str = "gpt-4o-mini", temperature: float = 0.8) -> Callable[[str], str]:
    import os
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. See .env.example.")
    client = OpenAI(api_key=api_key)

    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    return call_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Findzzer events.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=HERE / "proposal_generated_events_sample.csv")
    parser.add_argument("--no-llm", action="store_true", help="use the deterministic fallback text")
    parser.add_argument("--show-prompt", action="store_true", help="print one example prompt and exit")
    args = parser.parse_args()

    if args.show_prompt:
        random.seed(args.seed)
        combos, venues = load_weighted_combinations(), load_venues()
        performers = load_real_performers() + build_fictional_performers()
        captured: list[str] = []

        def capture(prompt: str) -> str:
            captured.append(prompt)
            return '{"name": "x", "description": "x"}'

        generate_one_event(0, combos, venues, performers, call_llm=capture)
        print(captured[0])
        return

    call_llm = None if args.no_llm else build_openai_call()
    dataset = generate_dataset(args.count, seed=args.seed, call_llm=call_llm)
    problems = validate_events(dataset)
    write_events_csv(dataset, args.out)
    mode = "deterministic fallback" if call_llm is None else "OpenAI gpt-4o-mini"
    print(f"Wrote {len(dataset)} events to {args.out} [{mode}]")
    print(f"Consistency check: {'clean' if not problems else f'{len(problems)} problems'}")
    for problem in problems[:10]:
        print("  -", problem)


if __name__ == "__main__":
    main()
