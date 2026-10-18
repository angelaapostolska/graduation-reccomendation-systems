"""Adds theme, entrance price, promotion, dress code and age limit to events."""

from __future__ import annotations

import hashlib
import random

GENRE_NOTES = {
    "Balkan Party": "regional folk-pop and turbo-folk dance hits",
    "Ex-Yu Hits": "popular songs from the former Yugoslavia",
    "Macedonian": "Macedonian pop and evergreens",
    "Hip-Hop": "hip-hop",
    "R&B": "R&B and soul-influenced pop",
    "House": "four-on-the-floor electronic dance music",
    "Techno": "driving electronic dance music",
    "Disco": "danceable disco and funk grooves",
    "Retro": "hits from past decades",
    "Pop": "current and classic pop",
    "Rock": "guitar-driven rock",
    "Punk": "fast, raw punk rock",
    "Jazz": "improvised jazz",
    "Blues": "blues and blues rock",
    "Latin": "Latin dance music",
    "Funk": "groove-based funk",
    "Folk": "traditional folk songs",
    "Electronic": "electronic and dance music",
    "Indie": "independent guitar pop and rock",
    "Alternative": "alternative rock",
    "Reggae": "reggae and dancehall rhythms",
    "Classical": "classical repertoire",
}

THEMES = {
    "Hip-Hop": ["2000s hip-hop", "old school hip-hop", "90s hip-hop and R&B"],
    "R&B": ["2000s R&B", "90s R&B"],
    "Rap": ["old school rap"],
    "Disco": ["90s disco", "70s and 80s disco"],
    "Retro": ["80s hits", "90s hits", "retro hits from the 70s to the 90s"],
    "Pop": ["2000s pop hits", "90s pop hits"],
    "Rock": ["classic rock", "90s rock", "rock covers"],
    "Punk": ["post-punk", "punk classics"],
    "House": ["deep house", "house classics"],
    "Techno": ["melodic techno"],
    "Balkan Party": ["turbo-folk and Balkan hits"],
    "Ex-Yu Hits": ["ex-Yu classics from the 80s and 90s"],
    "Macedonian": ["Macedonian evergreens"],
    "Latin": ["Latin hits"],
    "Jazz": ["jazz standards", "jazz and soul"],
    "Blues": ["blues covers"],
    "Funk": ["funk and soul"],
    "Folk": ["folk songs"],
    "Alternative": ["indie and alternative hits"],
    "Indie": ["indie and alternative hits"],
    "Electronic": ["electronic and dance"],
    "Metal": ["metal classics"],
    "Reggae": ["reggae and dancehall"],
}

P_PAID = {
    "Nightclub": 0.45, "Concert Venue": 0.7, "Theater": 0.9, "Workshop Space": 0.95,
    "Beach Bar": 0.25, "Bar": 0.12, "Pub": 0.10, "Restaurant": 0.05, "Cafe": 0.05,
}
P_FREE_STATED = 0.2  # of the events that are not paid
# MKD
PRICES = {
    "Nightclub": [200, 300, 400, 500, 600, 800],
    "Concert Venue": [300, 500, 600, 800, 1000, 1200],
    "Theater": [400, 500, 600, 800, 1000],
    "Workshop Space": [800, 1000, 1200, 1500, 1800],
    "Beach Bar": [100, 200, 300, 400, 500],
    "Bar": [100, 150, 200, 300, 400],
    "Pub": [100, 150, 200, 300],
    "Restaurant": [200, 300, 500],
    "Cafe": [100, 200],
}
PROMOTIONS = {
    "Bar": ["2-for-1 cocktails", "discounted draft beer", "a free welcome shot", "a student discount on drinks"],
    "Pub": ["discounted draft beer", "2-for-1 beer", "a free welcome shot"],
    "Beach Bar": ["2-for-1 cocktails", "a free welcome shot", "discounted draft beer"],
    "Nightclub": ["a free welcome shot with entry", "discounted entry for groups", "a student discount on drinks"],
    "Restaurant": ["a set menu offer", "a free dessert with a main course", "10% off for groups of six or more"],
    "Cafe": ["discounted coffee"],
    "Theater": ["a student ticket discount", "an early bird ticket price"],
    "Concert Venue": ["a student ticket discount", "an early bird ticket price"],
    "Workshop Space": ["a discount for groups of four or more"],
}
DRESS_CODES = {
    "Nightclub": ["smart casual", "all black"],
    "Beach Bar": ["beach-casual"],
    "Theater": ["smart casual"],
}
P_THEME, P_PROMO, P_DRESS = 0.35, 0.22, 0.14
P_AGE = {"Nightclub": 0.35, "Beach Bar": 0.15}


def _rng(event_id: str, salt: str) -> random.Random:
    digest = hashlib.sha256(f"{salt}:{event_id}".encode()).hexdigest()
    return random.Random(int(digest[:12], 16))


def enrich(
    event_id: str,
    genres: list[str],
    location_types: list[str],
    real_entrance: float | None,
    thin_caption: bool,
) -> dict:
    vtype = location_types[0]
    out = {"theme": "", "dress_code": "", "entrance": "", "entrance_mkd": "", "entrance_source": "",
           "promotion": "", "age_limit": "", "synthetic_fields": []}

    if real_entrance is not None:
        out["entrance_source"] = "real"
        out["entrance_mkd"] = int(real_entrance)
        out["entrance"] = "Free entry" if real_entrance == 0 else f"Entry {int(real_entrance)} MKD"

    if not thin_caption:
        return out

    if real_entrance is None:
        r = _rng(event_id, "entrance")
        paid = r.random() < P_PAID.get(vtype, 0.1)
        if paid:
            price = r.choice(PRICES.get(vtype, [200, 300]))
            out.update(entrance=f"Entry {price} MKD", entrance_mkd=price, entrance_source="synthetic")
            out["synthetic_fields"].append("entrance")
        elif r.random() < P_FREE_STATED:
            out.update(entrance="Free entry", entrance_mkd=0, entrance_source="synthetic")
            out["synthetic_fields"].append("entrance")

    r = _rng(event_id, "theme")
    options = next((THEMES[g] for g in genres if g in THEMES), None)
    if options and r.random() < P_THEME:
        out["theme"] = r.choice(options)
        out["synthetic_fields"].append("theme")

    r = _rng(event_id, "promo")
    if PROMOTIONS.get(vtype) and r.random() < P_PROMO:
        out["promotion"] = r.choice(PROMOTIONS[vtype])
        out["synthetic_fields"].append("promotion")

    r = _rng(event_id, "dress")
    if DRESS_CODES.get(vtype) and r.random() < P_DRESS:
        out["dress_code"] = r.choice(DRESS_CODES[vtype])
        out["synthetic_fields"].append("dress_code")

    r = _rng(event_id, "age")
    if r.random() < P_AGE.get(vtype, 0.03):
        out["age_limit"] = "18+"
        out["synthetic_fields"].append("age_limit")
    return out
