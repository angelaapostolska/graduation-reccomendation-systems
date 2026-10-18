"""Synthetic fill-in events: workshops, meetups, conferences, comedy and similar non-nightlife events."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mapping"))
from real_events import USAGE, get_client, llm_json, transliterate, violations  # noqa: E402

DATE_RANGE = (date(2025, 7, 1), date(2026, 10, 31))

# venue name -> (event count, {party type: weight})
PLANS = {
    "Public Room": (60, {"Standup Comedy": 45, "Theater": 15, "Poetry": 10, "Social Gathering": 20, "Convention": 10}),
    "Netaville": (50, {"Workshop": 30, "Networking": 30, "Convention": 25, "University Event": 5, "Open Day": 10}),
    "Base 45": (50, {"Workshop": 30, "Networking": 30, "Convention": 25, "University Event": 5, "Open Day": 10}),
    "Startupclub Skopje": (50, {"Workshop": 30, "Networking": 30, "Convention": 25, "University Event": 5, "Open Day": 10}),
    "Lakeview Event Hall": (45, {"Convention": 30, "Workshop": 25, "Networking": 15, "Charity": 10, "Poetry": 10, "Social Gathering": 10}),
    "Maker Yard Bitola": (45, {"Workshop": 70, "Open Day": 15, "Handmade Market": 15}),
}

TECH_WORKSHOPS = [
    "Intro to Python for Beginners", "Figma Basics for Non-Designers", "Excel for Small Business",
    "Introduction to Prompt Writing", "Public Speaking Basics", "Personal Finance Basics",
    "Photography for Social Media", "Pitch Deck Workshop", "Product Management 101",
    "Basic SEO for Local Businesses", "Video Editing on Your Phone", "Time Management for Freelancers",
    "UX Research Basics", "Git and GitHub Basics", "Data Visualization with Charts", "Negotiation Skills",
]
CRAFT_WORKSHOPS = [
    "Pottery for Beginners", "Candle Making", "Screen Printing Basics", "Introduction to Woodworking",
    "Leather Wallet Making", "Macrame Wall Hanging", "Linocut Printing", "Watercolor Basics",
    "Ceramic Mug Painting", "3D Printing Basics", "Soap Making", "Embroidery Basics",
]
TOPICS = {
    "Networking": [
        "Founders Meetup", "Product Managers Meetup", "Women in Tech Coffee", "Freelancers Meetup", "Startup Mixer",
        "Designers Meetup", "Developers Meetup", "Marketing Professionals Meetup", "Investor and Founder Mixer",
        "Remote Workers Meetup", "Student and Mentor Mixer", "Open Source Meetup",
    ],
    "Convention": [
        "Startup Summit", "Digital Marketing Conference", "AI in Business Day", "Tech Talks Day", "Product Conference",
        "Design Conference", "Cybersecurity Awareness Day", "Women in Tech Conference", "Freelance Summit",
        "Hackathon Weekend", "Demo Day", "Pitch Night", "Innovation Forum", "E-commerce Conference", "Cloud and DevOps Day",
    ],
    "University Event": ["Career Day", "Student Startup Day", "Internship Fair", "Thesis Presentation Day", "Alumni Meetup"],
    "Open Day": ["Open Day", "Open Doors for Founders", "Coworking Open Day", "Community Open Day"],
    "Standup Comedy": ["Standup Night", "Open Mic Standup", "Standup Special", "Comedy Showcase", "Roast Night", "Improv Comedy Night"],
    "Theater": ["Improv Theatre Night", "Monodrama Evening", "Short Plays Night", "Theatre Sports"],
    "Poetry": ["Poetry Night", "Open Mic Poetry", "Spoken Word Evening", "Poetry and Music"],
    "Social Gathering": ["Board Game Night", "Pub Quiz", "Film Club Evening", "Book Club Meetup", "Language Exchange Evening", "Trivia Night"],
    "Charity": ["Charity Auction", "Charity Fundraiser Evening", "Donation Drive"],
    "Handmade Market": ["Handmade Market", "Craft Fair", "Makers Market"],
}
RECURRING = {"Networking", "Standup Comedy", "Social Gathering", "Poetry"}

# (probability of a paid entrance, price points in MKD); otherwise free entry
ENTRANCE = {
    "Workshop": (0.85, [500, 800, 1000, 1200, 1500, 1800, 2000]),
    "Networking": (0.3, [200, 300, 500]),
    "Convention": (0.7, [500, 800, 1200, 1500, 2000, 2500]),
    "University Event": (0.1, [100, 200]),
    "Open Day": (0.0, []),
    "Standup Comedy": (0.9, [200, 300, 400, 500]),
    "Theater": (1.0, [300, 400, 500]),
    "Poetry": (0.5, [100, 200]),
    "Social Gathering": (0.4, [100, 200]),
    "Charity": (0.0, []),
    "Handmade Market": (0.0, []),
}
PROMOTIONS = {
    "Workshop": ["a student discount", "a discount for groups of three or more", "an early bird price"],
    "Convention": ["early bird tickets", "a student discount", "a discount for teams of four or more"],
    "Standup Comedy": ["a student discount", "a 2-for-1 ticket offer"],
    "Theater": ["a student discount"],
}
INCLUDES = {
    "Workshop": ["materials included", "certificate of attendance", "coffee break included"],
    "Convention": ["coffee breaks included", "lunch included", "recordings shared afterwards", "certificate of attendance"],
    "Networking": ["drinks and snacks included"],
}
WEEKDAYS = {  # Monday = 0
    "Workshop": {1, 2, 3, 5}, "Networking": {1, 2, 3}, "Convention": {3, 4, 5}, "University Event": {1, 2, 3},
    "Open Day": {5}, "Standup Comedy": {3, 4, 5}, "Theater": {3, 4, 5, 6}, "Poetry": {2, 3},
    "Social Gathering": {1, 2, 3, 4}, "Charity": {4, 5, 6}, "Handmade Market": {5, 6},
}
FIRST = ["Aleksandar", "Bojan", "Darko", "Filip", "Goran", "Igor", "Kristijan", "Marko", "Nikola", "Stefan", "Viktor",
         "Ana", "Biljana", "Elena", "Jovana", "Katerina", "Marija", "Simona", "Tamara", "Teodora"]
LAST_M = ["Stojanovski", "Petrovski", "Trajkovski", "Nikolovski", "Ilievski", "Georgievski", "Popovski", "Dimitrovski"]
LAST_F = ["Stojanovska", "Petrovska", "Trajkovska", "Nikolovska", "Ilievska", "Georgievska", "Popovska", "Dimitrovska"]
FEMALE = {"Ana", "Biljana", "Elena", "Jovana", "Katerina", "Marija", "Simona", "Tamara", "Teodora"}

PROMPT = """You write event listings for Findzzer, an events app in North Macedonia. English only.

Each item has: title, venue, city, party_type, topic, performer (a comedian or host, may be empty), and "facts":
entrance, promotion, language, includes, age_limit (each may be empty). Every fact you may use is in the input.

Write "description": two or three plain sentences, third person, under 330 characters.
  Sentence 1: what the event is, who leads it (if performer is given), and where (venue and city).
  Sentence 2: what it covers, only from title and topic. Skip it if there is nothing to add.
  Sentence 3: the filled facts only, as labelled parts, for example "Entry 800 MKD. Early bird price. Language: English. Materials included. 18+."
Rules:
- Use only the given facts. Never invent an agenda, speaker, guest, sponsor, schedule, price or venue detail.
- Do not state dates, weekdays or times. No phone numbers, e-mails, URLs or @handles.
- Banned words: vibrant, lively, energetic, unforgettable, perfect, amazing, great, cozy, exciting, experience, join, expect, enjoy, celebrate, atmosphere, opportunity, chance, you, your; no exclamation marks, emoji or hashtags.

Input is a JSON list. Return ONLY JSON: {{"results":[{{"id":int,"description":str}}]}}"""


def comedians(rng: random.Random) -> list[str]:
    names: set[str] = set()
    while len(names) < 10:
        first = rng.choice(FIRST)
        names.add(f"{first} {rng.choice(LAST_F if first in FEMALE else LAST_M)}")
    return sorted(names)


def pick_date(rng: random.Random, party: str) -> str:
    days = (DATE_RANGE[1] - DATE_RANGE[0]).days
    for _ in range(50):
        d = DATE_RANGE[0] + timedelta(days=rng.randint(0, days))
        if d.weekday() in WEEKDAYS[party]:
            return d.isoformat()
    return d.isoformat()


def build_events(seed: int) -> list[dict]:
    rng = random.Random(seed)
    with open(ROOT / "generator" / "data" / "venues.csv", newline="", encoding="utf-8") as handle:
        venues = {r["name"]: r for r in csv.DictReader(handle)}
    comics = comedians(rng)
    events = []
    for venue_name, (count, weights) in PLANS.items():
        venue = venues[venue_name]
        craft = "Workshop Space" in venue["location_types"] or venue_name == "Lakeview Event Hall"
        for _ in range(count):
            party = rng.choices(list(weights), weights=list(weights.values()))[0]
            if party == "Workshop":
                pool = CRAFT_WORKSHOPS if craft and rng.random() < (1.0 if venue_name == "Maker Yard Bitola" else 0.5) else TECH_WORKSHOPS
                title = rng.choice(pool)
            else:
                options = TOPICS[party]
                title = options[rng.randrange(len(options))]
                if party == "Open Day" and venue_name == "Maker Yard Bitola":
                    title = ["Open Day", "Open Studio Day", "Workshop Open Day", "Community Open Day"][options.index(title)]
                if party in RECURRING and rng.random() < 0.6:
                    title += f" Vol. {rng.randint(2, 14)}"
            performer = ""
            if party == "Standup Comedy":
                performer = rng.choice(comics)
                title = f"{title} with {performer}"
            paid_p, prices = ENTRANCE[party]
            entrance = f"Entry {rng.choice(prices)} MKD" if prices and rng.random() < paid_p else "Free entry"
            promotion = rng.choice(PROMOTIONS[party]) if party in PROMOTIONS and "Free" not in entrance and rng.random() < 0.3 else ""
            includes = rng.choice(INCLUDES[party]) if party in INCLUDES and rng.random() < 0.5 else ""
            tech = party in ("Workshop", "Networking", "Convention") and not (craft and title in CRAFT_WORKSHOPS)
            language = rng.choice(["English", "Macedonian"]) if tech and rng.random() < 0.7 else ("Macedonian" if rng.random() < 0.3 else "")
            age = "18+" if party == "Standup Comedy" and rng.random() < 0.3 else ""
            events.append({
                "party_types": party, "name": title, "venue_id": venue["venue_id"], "venue_name": venue_name,
                "city": venue["city"], "location_types": venue["location_types"], "performers": performer,
                "date": pick_date(rng, party), "theme": title.split(" with ")[0], "entrance": entrance,
                "promotion": promotion, "language": language, "includes": includes, "age_limit": age,
            })
    events.sort(key=lambda e: (e["date"], e["venue_name"]))
    for i, e in enumerate(events, 1):
        e["source_event_id"] = f"syn_{i:05d}"
    return events


def template(e: dict) -> str:
    text = f"{e['name']} at {e['venue_name']} in {e['city']}."
    parts = [e["entrance"], e["promotion"].capitalize() if e["promotion"] else "",
             f"Language: {e['language']}" if e["language"] else "", e["includes"].capitalize() if e["includes"] else "", e["age_limit"]]
    parts = [p for p in parts if p]
    return text + (" " + ". ".join(parts) + "." if parts else "")


def write_descriptions(events: list[dict], out: Path) -> None:
    cache_path = out / "cache_synthetic_descriptions.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = [e for e in events if e["source_event_id"] not in cache]
    client = get_client() if todo else None

    def run(batch):
        payload = [{"id": i, "title": e["name"], "venue": e["venue_name"], "city": e["city"], "party_type": e["party_types"],
                    "topic": e["theme"], "performer": e["performers"],
                    "facts": {k: e[k] for k in ("entrance", "promotion", "language", "includes", "age_limit")}}
                   for i, e in enumerate(batch)]
        res = llm_json(client, PROMPT, payload).get("results", [])
        done = {}
        for r in res:
            if isinstance(r, dict) and isinstance(r.get("id"), int) and 0 <= r["id"] < len(batch) and isinstance(r.get("description"), str):
                done[batch[r["id"]]["source_event_id"]] = r["description"].strip()
        return done

    batches = [todo[i:i + 8] for i in range(0, len(todo), 8)]
    with ThreadPoolExecutor(8) as pool:
        for part in pool.map(run, batches):
            cache.update(part)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    for e in events:
        text = transliterate(cache.get(e["source_event_id"], ""))
        facts = {k: e[k] for k in ("entrance", "promotion", "language", "includes", "age_limit")}
        if text and not violations(text, facts, e["name"]):
            e["description"], e["description_source"] = text, "llm_write"
        else:
            e["description"], e["description_source"] = template(e), "template_fallback"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--final", action="store_true", help="write events_synthetic.csv instead of the proposal file")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events = build_events(args.seed)
    write_descriptions(events, out)
    columns = ["source_event_id", "name", "description", "description_source", "date", "city", "venue_id", "venue_name",
               "location_types", "music_genres", "genre_source", "party_types", "performers", "theme", "dress_code",
               "entrance", "promotion", "age_limit", "language", "includes", "is_synthetic"]
    for e in events:
        e.update(music_genres="", genre_source="none", dress_code="", is_synthetic="1")
    target = out / ("events_synthetic.csv" if args.final else "proposal_events_synthetic.csv")
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    print(f"{len(events)} events -> {target}")
    print("per venue:", dict(Counter(e["venue_name"] for e in events)))
    print("per party type:", dict(Counter(e["party_types"] for e in events)))
    print("description_source:", dict(Counter(e["description_source"] for e in events)))
    print(f"API calls {USAGE['calls']}, tokens in {USAGE['in']:,} out {USAGE['out']:,}")


if __name__ == "__main__":
    main()
