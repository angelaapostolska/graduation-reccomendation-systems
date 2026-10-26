"""Stripped-text test: removes the genre and event-type words of the ontology from the event texts, embeds the stripped texts with ada-002
and saves them as a separate graph folder, so the graph search scripts can run on it (GRAPH_DIR=data/graph_stripped).
The tags stay in the graph. Only the text loses them, which tests whether the graph can supply what the text no longer says.
Usage: python gnn/stripped_text.py --dry-run     (statistics and examples, no API call)
       python gnn/stripped_text.py               (embeds about 5,900 short texts with ada-002, about $0.03)"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "search"))

# words that are also ordinary words, venue types or languages: stripping them would damage the text without removing a tag
KEEP = {"Other", "Bar", "Restaurant", "Caffe Bar", "Macedonian", "Gig", "Vinyl", "Retro", "Country", "Alternative"}
# spellings that the ontology names do not cover
ALIASES = {
    "Hip-Hop": ["hip hop", "hiphop"],
    "R&B": ["rnb", "r and b", "r'n'b"],
    "Standup Comedy": ["stand up comedy", "stand-up comedy", "standup", "stand up", "stand-up"],
    "Dj Set": ["dj set", "dj sets"],
    "Ex-Yu Hits": ["ex yu", "ex-yu"],
    "Balkan Party": ["balkan"],
    "Karaoke Night": ["karaoke"],
    "Live Music": ["live music"],
    "Electronic": ["electronic dance music"],
    "Outdoor Cinema": ["open air cinema", "outdoor movie"],
    "Folk": ["folk-pop", "turbo-folk", "turbo folk"],
    "Pop": ["pop-rock", "pop rock"],
}


def vocabulary(tables: str) -> list[str]:
    names = list(pd.read_csv(f"{tables}/music_genres.csv")["name"]) + list(pd.read_csv(f"{tables}/event_types.csv")["name"])
    terms = set()
    for name in names:
        if name in KEEP:
            continue
        low = name.lower()
        terms |= {low, low.replace("-", " "), low.replace("-", ""), low.replace(" ", "-")}
        terms |= {a.lower() for a in ALIASES.get(name, [])}
    return sorted(terms, key=len, reverse=True)


def pattern(terms: list[str]) -> re.Pattern:
    parts = [re.escape(t).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]?") for t in terms]
    return re.compile(r"(?<![\w])(?:" + "|".join(parts) + r")(?![\w])(?!-\w)", re.IGNORECASE)


def strip(text: str, regex: re.Pattern) -> str:
    out = text
    for _ in range(3):  # removing one word can expose another, as in "Live electronic music"
        new = regex.sub(" ", out)
        if new == out:
            break
        out = new
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    out = re.sub(r"([,;:])\s*([,.;:!?])", r"\2", out)
    out = re.sub(r"\s+", " ", out).strip(" ,;:-")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--graph", default=str(ROOT / "data" / "graph" / "graph.npz"))
    parser.add_argument("--out-dir", default=str(ROOT / "data" / "graph_stripped"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()

    events = pd.read_csv(f"{args.tables}/events.csv").set_index("event_id")
    order = np.load(args.graph, allow_pickle=False)["ids_event"]
    regex = pattern(vocabulary(args.tables))
    originals, stripped = [], []
    for event_id in order:
        row = events.loc[event_id]
        name = str(row["name"]) if pd.notna(row["name"]) else ""
        description = str(row["description"]) if pd.notna(row["description"]) else ""
        originals.append(f"{name}. {description}")
        new_name, new_description = strip(name, regex) or "Event", strip(description, regex)
        stripped.append(f"{new_name}. {new_description}")
    changed = [i for i, (a, b) in enumerate(zip(originals, stripped)) if a != b]
    left = sum(1 for t in stripped if regex.search(t))
    print(f"{len(order)} events, text changed for {len(changed)} ({len(changed) / len(order):.0%}), ontology words left in {left} stripped texts")
    step = max(1, len(changed) // args.examples)
    for i in changed[::step][: args.examples]:
        print(f"\n  before: {originals[i]}\n  after:  {stripped[i]}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stripped_texts.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["event_id", "original", "stripped"])
        writer.writerows(zip(order, originals, stripped))
    if args.dry_run:
        print(f"\ndry run: texts saved to {out / 'stripped_texts.csv'}, nothing embedded")
        return

    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY must be set in .env")
    from embedders import embed

    vectors, tokens = [], 0
    for start in range(0, len(stripped), 200):
        for attempt in range(4):
            try:
                batch, used = embed("text-embedding-ada-002", stripped[start:start + 200])
                break
            except Exception:  # noqa: BLE001 - retry transient API errors
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)
        vectors += batch
        tokens += used
        print(f"  {min(start + 200, len(stripped))}/{len(stripped)}")
    np.save(out / "event_text.npy", np.array(vectors, dtype=np.float32))
    shutil.copy(args.graph, out / "graph.npz")
    print(f"saved stripped vectors and a copy of the graph to {out}; {tokens:,} tokens, about ${tokens * 0.10 / 1e6:.3f}")


if __name__ == "__main__":
    main()
