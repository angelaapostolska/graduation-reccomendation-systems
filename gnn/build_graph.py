"""Builds the heterogeneous event graph from data/tables/*.csv. Usage: python gnn/build_graph.py [--tables dir] [--out data/graph]"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NODE_TABLES = {
    "event": ("events", "event_id"),
    "performer": ("performers", "performer_id"),
    "venue": ("venues", "venue_id"),
    "city": ("cities", "city_id"),
    "business_type": ("business_types", "type_id"),
    "music_genre": ("music_genres", "music_genre_id"),
    "event_type": ("event_types", "event_type_id"),
}
EDGES = {
    "event__hosted_at__venue": ("events", "event_id", "venue_id", "event", "venue"),
    "venue__located_in__city": ("venues", "venue_id", "city_id", "venue", "city"),
    "venue__has_type__business_type": ("venue_types", "venue_id", "type_id", "venue", "business_type"),
    "event__has_genre__music_genre": ("event_music_genres", "event_id", "music_genre_id", "event", "music_genre"),
    "event__has_event_type__event_type": ("event_event_types", "event_id", "event_type_id", "event", "event_type"),
    "event__features__performer": ("event_performers", "event_id", "performer_id", "event", "performer"),
    "performer__associated_with__music_genre": ("performer_music_genres", "performer_id", "music_genre_id", "performer", "music_genre"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--out", default=str(ROOT / "data" / "graph"))
    args = parser.parse_args()
    tables = Path(args.tables)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    nodes, index = {}, {}
    for kind, (table, key) in NODE_TABLES.items():
        frame = pd.read_csv(tables / f"{table}.csv", dtype={key: str})
        nodes[kind] = frame
        index[kind] = {value: i for i, value in enumerate(frame[key])}

    arrays: dict[str, np.ndarray] = {}
    summary = {"nodes": {kind: len(frame) for kind, frame in nodes.items()}, "edges": {}}
    for kind, frame in nodes.items():
        arrays[f"ids_{kind}"] = np.asarray(frame.iloc[:, 0].astype(str).tolist(), dtype=str)

    for name, (table, source_col, target_col, source_kind, target_kind) in EDGES.items():
        frame = pd.read_csv(tables / f"{table}.csv", dtype=str).dropna(subset=[source_col, target_col])
        source = frame[source_col].map(index[source_kind])
        target = frame[target_col].map(index[target_kind])
        keep = source.notna() & target.notna()
        arrays[f"edge_{name}"] = np.vstack([source[keep].astype(np.int64), target[keep].astype(np.int64)])
        summary["edges"][name] = int(keep.sum())

    events = nodes["event"]
    arrays["event_is_synthetic"] = events["is_synthetic"].astype(str).str.lower().eq("true").to_numpy()
    arrays["event_origin"] = np.asarray(events["origin"].astype(str).tolist(), dtype=str)
    genre_edges = pd.read_csv(tables / "event_music_genres.csv", dtype=str)
    genre_edges["event"] = genre_edges["event_id"].map(index["event"])
    genre_source = genre_edges.groupby("event")["source"].first()
    source_per_event = np.array(["none"] * len(events), dtype=object)
    source_per_event[genre_source.index.to_numpy(dtype=int)] = genre_source.to_numpy()
    arrays["event_genre_source"] = np.asarray(source_per_event.tolist(), dtype=str)
    summary["events_without_genre"] = int((source_per_event == "none").sum())
    summary["events_without_genre_real"] = int(((source_per_event == "none") & ~arrays["event_is_synthetic"]).sum())

    np.savez_compressed(out / "graph.npz", **arrays)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved {out / 'graph.npz'}")


if __name__ == "__main__":
    main()
