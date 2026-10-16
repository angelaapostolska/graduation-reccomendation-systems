"""
Generator script (pipeline step 3a).

Produces synthetic Findzzer-shaped events: realistic, taxonomy-correct, and
weighted by real preference data -- but entirely fictional. No real venues,
no real users, nothing scraped.

Entities generated per event, matching the confirmed ontology:
    Event, Venue, City, BusinessType (location type), Genre(s), PartyType(s),
    Performer (new -- see below)

Why Performer is included here (and wasn't in the real-data pipeline):
the earlier "no clean extraction" blocker only applies to pulling a performer's
name out of a messy real photo/caption. There's no such extraction problem for
synthetic data -- we're authoring the event, so the performer is just another
field to generate, consistent with its genre. Real sample events we pulled from
Findzzer show this pattern constantly: the event *name* often *is* the
performer's name ("OVERTONE", "DJ Cena", "Nina & 2Exit").

An LLM call is injected as a plain function (call_llm: str -> str) so this
script isn't tied to one provider. A deterministic fallback generator is
included so the script is runnable and testable with no API key.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from taxonomy import LOCATION_TYPES, PERFORMER_ELIGIBLE_PARTY_TYPES


DISTRIBUTION_CSV = Path(__file__).parent / "genre_party_type_distribution.csv"

# Small fixed set of fictional venues per location type, enough to let events
# share a venue (needed for the graph's HOSTED_AT / shared-venue connections)
# without needing a real venue database yet.
FICTIONAL_VENUES = {
    "Nightclub": ["Pulse Nightclub", "Stanica 26", "Vault Skopje"],
    "Bar": ["Street Pub", "Beer Garden", "The Brass Tap"],
    "Lounge": ["Octava Lounge Bar", "Velvet Lounge"],
    "Live Music Venue": ["Rock Bar Krug", "The Cellar Stage"],
    "Cocktail Bar": ["Cuban Cocktail Bar", "Bitters & Co."],
    "Pub": ["Irish Times Pub", "The Local Pub"],
    "Concert Venue": ["Arena 041", "Boris Trajkovski Hall"],
    "Disco": ["Disco Palace", "Neon Floor"],
}

CITIES = ["Skopje", "Ohrid", "Bitola"]

FIRST_NAME_SYLLABLES = ["Dar", "Mi", "El", "Van", "Zo", "Kris", "Ale", "Niko", "Tea", "Mar"]
LAST_NAME_SYLLABLES = ["ko", "ska", "ov", "ic", "ov ski", "eva", "ov a", "in"]
STAGE_NAME_PREFIXES = ["DJ", "MC", ""]


GENERATION_PROMPT = """
You write short, factual Findzzer event listings -- a nightlife and culture
events app. Findzzer's house style: 1-3 factual sentences, no marketing
slogans, no hype words, no emoji, no hashtags. Keep it under 400 characters.

Write ONE event with these fixed facts (do not change them):
- Performer / host: {performer_name}
- Primary music genre(s): {genres}
- Party type(s): {party_types}
- Venue: {venue_name}
- Venue type: {location_type}
- City: {city}

Return ONLY valid JSON with this exact shape:
{{
  "name": "string, usually just the performer/host name, matching real Findzzer listings",
  "description": "1-3 factual sentences about the event, venue atmosphere, and genre"
}}
""".strip()


@dataclass(frozen=True)
class GenreWeight:
    music_genre: str
    party_type: str
    weight: float


@dataclass
class GeneratedEvent:
    event_id: str
    name: str
    description: str
    city: str
    venue_name: str
    location_type: str
    music_genres: list[str]
    party_types: list[str]
    performer_name: Optional[str]


def load_weighted_combinations() -> list[GenreWeight]:
    with DISTRIBUTION_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            GenreWeight(row["music_genre"], row["party_type"], float(row["percentage"]))
            for row in reader
        ]


def pick_weighted_combination(combinations: list[GenreWeight]) -> GenreWeight:
    return random.choices(combinations, weights=[c.weight for c in combinations], k=1)[0]


def maybe_second_genre(primary: str, *, probability: float = 0.3) -> list[str]:
    from taxonomy import MUSIC_GENRES
    if random.random() > probability:
        return [primary]
    candidates = [g for g in MUSIC_GENRES if g != primary]
    return [primary, random.choice(candidates)]


def pick_venue(location_type: str) -> str:
    pool = FICTIONAL_VENUES.get(location_type)
    if not pool:
        location_type = random.choice(list(FICTIONAL_VENUES))
        pool = FICTIONAL_VENUES[location_type]
    return random.choice(pool)


def generate_performer_name(genre: str) -> str:
    """Fictional performer name, loosely styled to the genre (DJ-style prefix
    for electronic genres, plain name otherwise)."""
    first = random.choice(FIRST_NAME_SYLLABLES) + random.choice(FIRST_NAME_SYLLABLES).lower()
    last = random.choice(LAST_NAME_SYLLABLES)
    base_name = f"{first}{last}".replace(" ", "")
    if genre in {"Techno", "House", "Trance", "Electronic", "Afro House", "Disco"}:
        return f"DJ {base_name}"
    return base_name


def build_generation_prompt(
    *,
    performer_name: Optional[str],
    genres: list[str],
    party_types: list[str],
    venue_name: str,
    location_type: str,
    city: str,
) -> str:
    return GENERATION_PROMPT.format(
        performer_name=performer_name or "none (no named performer for this event type)",
        genres=", ".join(genres),
        party_types=", ".join(party_types),
        venue_name=venue_name,
        location_type=location_type,
        city=city,
    )


def generate_one_event(
    event_index: int,
    combinations: list[GenreWeight],
    *,
    call_llm: Optional[Callable[[str], str]] = None,
) -> GeneratedEvent:
    combo = pick_weighted_combination(combinations)
    genres = maybe_second_genre(combo.music_genre)
    party_types = [combo.party_type]

    location_type = random.choice(LOCATION_TYPES)
    venue_name = pick_venue(location_type)
    city = random.choice(CITIES)

    performer_name = (
        generate_performer_name(combo.music_genre)
        if combo.party_type in PERFORMER_ELIGIBLE_PARTY_TYPES
        else None
    )

    prompt = build_generation_prompt(
        performer_name=performer_name,
        genres=genres,
        party_types=party_types,
        venue_name=venue_name,
        location_type=location_type,
        city=city,
    )

    if call_llm is not None:
        raw_response = call_llm(prompt)
        payload = _parse_json_response(raw_response)
        name, description = payload["name"], payload["description"]
    else:
        name, description = _deterministic_fallback_text(
            performer_name=performer_name,
            genres=genres,
            party_types=party_types,
            venue_name=venue_name,
        )

    return GeneratedEvent(
        event_id=f"evt_{event_index:05d}",
        name=name,
        description=description,
        city=city,
        venue_name=venue_name,
        location_type=location_type,
        music_genres=genres,
        party_types=party_types,
        performer_name=performer_name,
    )


def _parse_json_response(raw_text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.I | re.M)
    return json.loads(text)


def _deterministic_fallback_text(
    *, performer_name: Optional[str], genres: list[str], party_types: list[str], venue_name: str
) -> tuple[str, str]:
    name = performer_name or f"{party_types[0]} Night"
    description = (
        f"{', '.join(genres)} {party_types[0].lower()} at {venue_name}"
        + (f", featuring {performer_name}." if performer_name else ".")
    )
    return name, description


def generate_dataset(
    count: int,
    *,
    call_llm: Optional[Callable[[str], str]] = None,
    seed: Optional[int] = None,
) -> list[GeneratedEvent]:
    if seed is not None:
        random.seed(seed)
    combinations = load_weighted_combinations()
    return [generate_one_event(i, combinations, call_llm=call_llm) for i in range(count)]


def write_events_csv(events: list[GeneratedEvent], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "event_id", "name", "description", "city", "venue_name",
            "location_type", "music_genres", "party_types", "performer_name",
        ])
        for event in events:
            writer.writerow([
                event.event_id, event.name, event.description, event.city,
                event.venue_name, event.location_type,
                "|".join(event.music_genres), "|".join(event.party_types),
                event.performer_name or "",
            ])


if __name__ == "__main__":
    dataset = generate_dataset(20, seed=42)  # no call_llm -> deterministic fallback
    output_file = Path(__file__).parent / "generated_events_sample.csv"
    write_events_csv(dataset, output_file)
    print(f"Wrote {len(dataset)} events to {output_file}")
    for event in dataset[:5]:
        print(f"  {event.name!r} -- {event.music_genres} / {event.party_types} @ {event.venue_name}")
