"""
Real-event extraction (pipeline step 2, option C: real events + synthetic fill-ins).

Reads a local pg_dump of the Findzzer database and produces a clean, normalized
corpus of the real events held at the curated venues in generator/data/venues.csv.

Nothing this script reads or writes is committed: the dump, the venue list and all
outputs (default: data/real/) are gitignored. Only the code is public.

Stages (each can be run on its own; LLM stages cache their results so reruns are cheap):

  base          dump + venues.csv  ->  events_base.csv
                Keeps non-deleted events at the curated venues. Maps duplicate
                venue templates to one venue. Treats default genre tags as missing
                and drops "Private Party" (Findzzer's own classifier forbids it).
  performers    events_base.csv    ->  performers.csv, event_performers.csv
                LLM decides which event titles are performer names.
  descriptions  events_base.csv    ->  events.csv
                LLM rewrites each description in Findzzer house style (English,
                only facts already present, no contact details) and fills in
                genres that were only default tags.

Usage:
  python mapping/real_events.py base --dump forge.sql.gz
  python mapping/real_events.py performers
  python mapping/real_events.py descriptions --sample 20     # preview, prints rows
  python mapping/real_events.py descriptions                  # full run
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "generator"))
from taxonomy import MUSIC_GENRES  # noqa: E402

DEFAULT_GENRE_SETS = [frozenset({"Gig"}), frozenset({"Hip-Hop", "Pop", "R&B"})]
EXCLUDED_PARTY_TYPES = {"Private Party"}
TABLES = {
    "events", "venue_templates", "venue_events", "event_music", "event_types",
    "music_genres", "party_types",
}
MODEL = "gpt-4o-mini"
CYRILLIC = re.compile("[Ѐ-ӿ]")
CONTACT = re.compile(r"(https?://|www\.|@\w|\+\d[\d\s/-]{6,}|\b\d{3}[\s/-]\d{3}[\s/-]\d{3,4}\b)")


# ------------------------------------------------------------------ dump parsing


def read_dump(path: Path) -> dict[str, list[dict]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    tables: dict[str, list[dict]] = {}
    for match in re.finditer(r"COPY public\.(\w+) \(([^)]*)\) FROM stdin;\n(.*?)\n\\\.\n", text, re.S):
        name = match.group(1)
        if name not in TABLES:
            continue
        cols = [c.strip().strip('"') for c in match.group(2).split(",")]
        rows = []
        for line in match.group(3).split("\n"):
            fields = [
                None if f == "\\N" else f.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
                for f in line.split("\t")
            ]
            if len(fields) == len(cols):
                rows.append(dict(zip(cols, fields)))
        tables[name] = rows
    return tables


def handle_of(url: str | None) -> str | None:
    m = re.search(r"instagram\.com/([^/?]+)", url or "")
    return m.group(1).lower() if m else None


def norm(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower())


MK_TO_LATIN = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ѓ": "gj", "е": "e", "ж": "zh", "з": "z", "ѕ": "dz",
    "и": "i", "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p",
    "р": "r", "с": "s", "т": "t", "ќ": "kj", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "ch", "џ": "dz", "ш": "sh",
}


def latin_tokens(text: str) -> list[str]:
    lowered = "".join(MK_TO_LATIN.get(ch, ch) for ch in text.lower())
    ascii_text = unicodedata.normalize("NFKD", lowered).encode("ascii", "ignore").decode()
    return [t for t in re.split(r"[^a-z0-9]+", ascii_text) if t]


def name_supported(name: str, title: str, details: str) -> bool:
    """Guard against the LLM inventing a performer: at least one meaningful token of
    the name must (fuzzily) appear in the title or caption, after transliteration."""
    from difflib import SequenceMatcher
    source = latin_tokens(title) + latin_tokens(details)
    name_tokens = [t for t in latin_tokens(name) if len(t) >= 3 and t not in {"dj", "the", "and", "band"}] or latin_tokens(name)
    return any(SequenceMatcher(None, t, s).ratio() >= 0.75 for t in name_tokens for s in source)


# --------------------------------------------------------------------- stage: base


def stage_base(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    db = read_dump(Path(args.dump))
    with open(args.venues, newline="", encoding="utf-8") as handle:
        venues = list(csv.DictReader(handle))

    # template id -> curated venue id (Instagram handle first, exact normalized name second)
    by_handle: dict[str, list[dict]] = defaultdict(list)
    by_name: dict[str, list[dict]] = defaultdict(list)
    for t in db["venue_templates"]:
        if handle_of(t["instagram_link"]):
            by_handle[handle_of(t["instagram_link"])].append(t)
        by_name[norm(t["business_name"])].append(t)
    template_to_venue: dict[str, str] = {}
    unmatched = []
    for v in venues:
        found = by_handle.get(handle_of(v["instagram_url"]) or "", [])
        if not found:
            found = by_name.get(norm(v["name"]), [])
        if not found:
            unmatched.append(v["name"])
        for t in found:
            template_to_venue.setdefault(t["id"], v["venue_id"])
    venue_by_id = {v["venue_id"]: v for v in venues}

    genre_name = {r["id"]: r["name"] for r in db["music_genres"]}
    party_name = {r["id"]: r["name"] for r in db["party_types"]}
    genres: dict[str, list[str]] = defaultdict(list)
    for r in db["event_music"]:
        genres[r["event_id"]].append(genre_name[r["music_genre_id"]])
    parties: dict[str, list[str]] = defaultdict(list)
    for r in db["event_types"]:
        parties[r["event_id"]].append(party_name[r["party_type"]])
    event_template = {r["event_id"]: r["template_id"] for r in db["venue_events"]}

    rows = []
    for e in db["events"]:
        if e["deleted_at"] or event_template.get(e["id"]) not in template_to_venue:
            continue
        venue = venue_by_id[template_to_venue[event_template[e["id"]]]]
        raw_genres = genres.get(e["id"], [])
        default_tag = frozenset(raw_genres) in DEFAULT_GENRE_SETS
        clean_genres = [] if default_tag else [g for g in raw_genres if g in MUSIC_GENRES]
        clean_parties = [p for p in parties.get(e["id"], []) if p not in EXCLUDED_PARTY_TYPES]
        rows.append({
            "source_event_id": e["id"],
            "name_original": e["name"].strip(),
            "date": e["date"],
            "venue_id": venue["venue_id"],
            "venue_name": venue["name"],
            "city": venue["city"],
            "location_types": venue["location_types"],
            "music_genres": "|".join(clean_genres),
            "genre_status": "default_tag" if default_tag else ("real" if clean_genres else "none"),
            "party_types": "|".join(clean_parties),
            "details_original": (e["details"] or "").strip(),
        })
    rows.sort(key=lambda r: (r["date"], int(r["source_event_id"])))
    write_csv(out / "events_base.csv", rows)
    status = Counter(r["genre_status"] for r in rows)
    print(f"{len(rows)} events at {len({r['venue_id'] for r in rows})} venues -> {out / 'events_base.csv'}")
    print(f"genre status: {dict(status)}; venues without a match in the dump: {unmatched or 'none'}")


# ------------------------------------------------------------------- LLM helpers


def llm_json(client, system: str, payload) -> dict:
    last = None
    for _ in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            )
            return json.loads(r.choices[0].message.content)
        except Exception as err:  # noqa: BLE001 - retry any transport / parse failure
            last = err
    print("LLM batch failed:", last, file=sys.stderr)
    return {}


def get_client():
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    return OpenAI()


def load_cache(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ----------------------------------------------------------------- stage: performers

PERFORMER_PROMPT = """You classify event titles from a Macedonian nightlife and culture events app.
For each title decide whether it is (or contains) the name of a PERFORMER: a DJ, band, singer, musician, comedian, collective or other named act.

Return is_performer=false for generic event titles (e.g. "Quiz Night", "Birthday Party", "Ladies Night", "New Year's Eve", "Live Music", "Karaoke", "Weekly Agenda"), venue names, and holidays.
If is_performer=true:
- "performers": a list of the act names, cleaned: remove venue names, dates, "Live", "Vol. 1", "at", "w/", emoji, prefixes like "Night with". If the title lists separate acts joined by "+", "x", "w/", "feat." or commas, list each act separately. Keep names like "Nina & 2Exit" or "Rafinerija & Adrijana" as ONE act when it is a duo or band name.
- Transliterate Cyrillic names to Latin (Macedonian romanisation).
- "kinds": one of dj, band, solo, collective, comedian, other (per act, same order as performers).
Use the genres and venue context only as hints. If unsure, is_performer=false.

Input is a JSON list of {id, title, genres, party_types, venue, detail}. Return ONLY JSON:
{"results":[{"id":int,"is_performer":bool,"performers":[str],"kinds":[str]}]}"""


def stage_performers(args) -> None:
    out = Path(args.out)
    events = read_csv(out / "events_base.csv")
    titles = sorted({e["name_original"] for e in events})
    cache_path = out / "cache_performers.json"
    cache = load_cache(cache_path)
    todo = [t for t in titles if t not in cache]
    if todo:
        client = get_client()
        context = {}
        for e in events:
            context.setdefault(e["name_original"], e)
        batches = [todo[i:i + 40] for i in range(0, len(todo), 40)]

        def run(batch):
            payload = [
                {"id": i, "title": t, "genres": context[t]["music_genres"].split("|") if context[t]["music_genres"] else [],
                 "party_types": context[t]["party_types"].split("|"), "venue": context[t]["venue_name"],
                 "detail": context[t]["details_original"][:100]}
                for i, t in enumerate(batch)
            ]
            res = llm_json(client, PERFORMER_PROMPT, payload).get("results", [])
            return {batch[r["id"]]: r for r in res if isinstance(r.get("id"), int) and 0 <= r["id"] < len(batch)}

        with ThreadPoolExecutor(8) as pool:
            for part in pool.map(run, batches):
                cache.update(part)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    performers: dict[str, dict] = {}
    links = []
    rejected = 0
    for e in events:
        r = cache.get(e["name_original"])
        if not r or not r.get("is_performer"):
            continue
        for j, name in enumerate(r["performers"]):
            name = name.strip()
            if len(name) < 2:
                continue
            if not name_supported(name, e["name_original"], e["details_original"]):
                rejected += 1
                continue
            key = norm(name)
            kinds = r.get("kinds") or []
            p = performers.setdefault(key, {"name": name, "kinds": Counter(), "genres": Counter(), "events": 0})
            p["events"] += 1
            p["kinds"][kinds[j] if j < len(kinds) else (kinds[0] if kinds else "other")] += 1
            for g in e["music_genres"].split("|"):
                if g:
                    p["genres"][g] += 1
            links.append({"source_event_id": e["source_event_id"], "performer_id": key})
    ids = {k: f"per_{i + 1:04d}" for i, k in enumerate(sorted(performers, key=lambda k: -performers[k]["events"]))}
    write_csv(out / "performers.csv", [
        {"performer_id": ids[k], "name": p["name"], "kind": p["kinds"].most_common(1)[0][0],
         "genres": "|".join(g for g, _ in p["genres"].most_common(3)), "events": p["events"]}
        for k, p in sorted(performers.items(), key=lambda kv: -kv[1]["events"])
    ])
    write_csv(out / "event_performers.csv", [
        {"source_event_id": l["source_event_id"], "performer_id": ids[l["performer_id"]]} for l in links
    ])
    print(f"{len(performers)} performers, {len(links)} event-performer links -> {out} "
          f"({rejected} links dropped: name not found in title or caption)")


# ---------------------------------------------------------------- stage: descriptions

DESCRIPTION_PROMPT = """You rewrite event listings for Findzzer, a nightlife and culture events app in North Macedonia. English only.

For each event you get: title, performers, venue, venue type(s), city, genres (may be empty), party types, and "caption" (raw text scraped from the venue's Instagram, may be empty, may be in Macedonian or English).

Write for each event:
- "name": the title in English / Latin script. Transliterate Cyrillic names; translate generic words (e.g. "Караоке" -> "Karaoke"). Keep performer names exactly as given in "performers" when the title is just the act.
- "description": 1 or 2 plain factual sentences, under 220 characters, third person.
- "genres": ONLY when the input genres are empty: up to 2 music genres chosen from the allowed list, if the title, performers or caption clearly support them; otherwise []. When genres are given, return them unchanged.

Description rules:
- Use only facts present in the input. Never invent prices, times, guests, offers, sponsors or venue details.
- Mention the venue by name and the performers by name when given.
- Keep concrete facts from the caption that are not contact details or times (an offer, a theme, a named guest, a special occasion).
- NEVER include phone numbers, email addresses, URLs, @handles or reservation instructions. Do not state dates or start times.
- Banned: "Experience", "Join us", "Expect", "vibrant", "lively", "energetic", "perfect for", "unforgettable", exclamation marks, emoji, hashtags, second person ("you").
- Style reference: "Live music performance by Overtone at Octava Lounge Bar." / "Dance into the night with a DJ set by DJ Cena at Beer Garden."
- If the input has too little to say more, a single short sentence is fine (e.g. "Karaoke night at Bajka 2D Bar in Skopje.").

Allowed genres: {genres}

Input is a JSON list. Return ONLY JSON: {{"results":[{{"id":int,"name":str,"description":str,"genres":[str]}}]}}"""


def keep_original(details: str) -> bool:
    """A real, clean English description can stay as it is."""
    return (
        bool(details) and not details.startswith("📌") and not CYRILLIC.search(details)
        and not CONTACT.search(details) and 30 <= len(details) <= 300
    )


def stage_descriptions(args) -> None:
    out = Path(args.out)
    events = read_csv(out / "events_base.csv")
    perf_name = {r["performer_id"]: r["name"] for r in read_csv(out / "performers.csv")}
    perf_by_event: dict[str, list[str]] = defaultdict(list)
    for r in read_csv(out / "event_performers.csv"):
        perf_by_event[r["source_event_id"]].append(perf_name[r["performer_id"]])

    cache_path = out / "cache_descriptions.json"
    cache = load_cache(cache_path)
    pool = [e for e in events if not keep_original(e["details_original"])]
    if args.sample:
        import random
        random.seed(11)
        pool = random.sample(pool, args.sample)
    todo = [e for e in pool if e["source_event_id"] not in cache]
    client = get_client() if todo else None
    system = DESCRIPTION_PROMPT.format(genres=", ".join(MUSIC_GENRES))

    def run(batch):
        payload = []
        for i, e in enumerate(batch):
            caption = re.sub(r"^📌[^\n]*\n?", "", e["details_original"]).strip()
            payload.append({
                "id": i, "title": e["name_original"], "performers": perf_by_event.get(e["source_event_id"], []),
                "venue": e["venue_name"], "venue_types": e["location_types"].split("|"), "city": e["city"],
                "genres": e["music_genres"].split("|") if e["music_genres"] else [],
                "party_types": e["party_types"].split("|") if e["party_types"] else [],
                "caption": caption[:600],
            })
        res = llm_json(client, system, payload).get("results", [])
        return {batch[r["id"]]["source_event_id"]: r for r in res if isinstance(r.get("id"), int) and 0 <= r["id"] < len(batch)}

    batches = [todo[i:i + 8] for i in range(0, len(todo), 8)]
    if batches:
        with ThreadPoolExecutor(8) as ex:
            for part in ex.map(run, batches):
                cache.update(part)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    result = []
    scope = {e["source_event_id"] for e in pool} if args.sample else None
    for e in events:
        sid = e["source_event_id"]
        if scope is not None and sid not in scope:
            continue
        if keep_original(e["details_original"]):
            name, desc, genres, source = e["name_original"], e["details_original"], e["music_genres"], "kept_original"
        else:
            r = cache.get(sid)
            if not r:
                continue
            name, desc, source = r["name"], r["description"], "llm_rewrite"
            inferred = [g for g in (r.get("genres") or []) if g in MUSIC_GENRES]
            genres = e["music_genres"] or "|".join(inferred[:2])
        result.append({
            "source_event_id": sid, "name": name, "description": desc, "description_source": source,
            "date": e["date"], "city": e["city"], "venue_id": e["venue_id"], "venue_name": e["venue_name"],
            "location_types": e["location_types"], "music_genres": genres,
            "genre_source": "real" if e["music_genres"] else ("inferred" if genres else "none"),
            "party_types": e["party_types"], "performers": "|".join(perf_by_event.get(sid, [])),
            "name_original": e["name_original"], "details_original": e["details_original"],
        })
    target = out / ("events_sample.csv" if args.sample else "events.csv")
    if result:
        write_csv(target, result)
    print(f"{len(result)} events -> {target}  (kept original: {sum(r['description_source'] == 'kept_original' for r in result)})")
    if args.sample:
        for r in result:
            print(f"\n[{r['venue_name']}, {r['city']}] {r['name']}  <- {r['name_original']!r}"
                  f"\n   genres: {r['music_genres'] or '-'} ({r['genre_source']}) | party: {r['party_types'] or '-'}"
                  f" | performers: {r['performers'] or '-'}"
                  f"\n   NEW: {r['description']}"
                  f"\n   OLD: {r['details_original'][:160].replace(chr(10), ' / ')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["base", "performers", "descriptions"])
    parser.add_argument("--dump", help="path to the pg_dump (.sql or .sql.gz); base stage only")
    parser.add_argument("--venues", default=str(ROOT / "generator" / "data" / "venues.csv"))
    parser.add_argument("--out", default=str(ROOT / "data" / "real"))
    parser.add_argument("--sample", type=int, default=0, help="descriptions stage: preview N events only")
    args = parser.parse_args()
    if args.stage == "base":
        if not args.dump:
            parser.error("base stage needs --dump")
        stage_base(args)
    elif args.stage == "performers":
        stage_performers(args)
    else:
        stage_descriptions(args)


if __name__ == "__main__":
    main()
