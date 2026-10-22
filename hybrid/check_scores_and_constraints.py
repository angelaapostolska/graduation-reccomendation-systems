"""Preparation for the hybrid search idea: how spread out the ada-002 top scores are, and how well date, city and price constraints can be checked.
Usage: python hybrid/check_scores_and_constraints.py [--data-only] [--out data/search_runs/hybrid_check.json]"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "search"))
from testing_query import QUERIES  # noqa: E402

MODEL = "text-embedding-ada-002"
BETAS = (0.05, 0.1)
# constraints read out of the words of a query; "cheap" and "student discount" are too vague for a database field
CONSTRAINTS = {
    "free techno night in Skopje": {"city": "Skopje", "free": True},
    "cheap standup comedy in Skopje": {"city": "Skopje"},
}


def load_events(tables: str) -> pd.DataFrame:
    events = pd.read_csv(f"{tables}/events.csv")
    venues = pd.read_csv(f"{tables}/venues.csv")[["venue_id", "city_id"]]
    cities = pd.read_csv(f"{tables}/cities.csv").rename(columns={"name": "city"})[["city_id", "city"]]
    return events.merge(venues, on="venue_id", how="left").merge(cities, on="city_id", how="left")


def data_report(events: pd.DataFrame) -> dict:
    price_known = events["entrance_mkd"].notna()
    return {
        "events": int(len(events)),
        "date_range": [str(events["event_date"].min()), str(events["event_date"].max())],
        "share_with_known_price": float(price_known.mean()),
        "share_free": float((events["entrance_mkd"] == 0).mean()),
        "share_free_among_known_price": float((events.loc[price_known, "entrance_mkd"] == 0).mean()),
        "known_price_by_source": events.loc[price_known, "entrance_source"].value_counts().to_dict(),
        "events_per_city": {k: int(v) for k, v in events["city"].value_counts().head(8).items()},
    }


def score_spread(scores_per_query: list[np.ndarray]) -> list[dict]:
    rows = []
    for scores in scores_per_query:
        top = np.sort(scores)[::-1][:50]
        spread50 = float(top[0] - top[49])
        rows.append({"top1": float(top[0]), "rank10": float(top[9]), "rank50": float(top[49]), "spread_top1_to_top10": float(top[0] - top[9]),
                     "spread_top1_to_top50": spread50, "std_top50": float(top.std()),
                     **{f"beta_{b}_as_share_of_top50_spread": float(b / spread50) for b in BETAS}})
    return rows


def constraint_check(scores_per_query: list[np.ndarray], events: pd.DataFrame) -> dict:
    out = {}
    for query, scores in zip(QUERIES, scores_per_query):
        rules = CONSTRAINTS.get(query)
        if not rules:
            continue
        order = np.argsort(-scores)
        entry = {"constraints": rules}
        for depth in (10, 50):
            top = events.iloc[order[:depth]]
            ok_city = top["city"] == rules["city"] if "city" in rules else pd.Series(True, index=top.index)
            known = top["entrance_mkd"].notna()
            free = top["entrance_mkd"] == 0
            result = {"in_city": float(ok_city.mean())}
            if rules.get("free"):
                result.update({"price_known": float(known.mean()), "free": float(free.mean()), "known_and_not_free": float((known & ~free).mean()),
                               "kept_if_unknown_price_is_dropped": int((ok_city & free).sum()), "kept_if_unknown_price_is_kept": int((ok_city & (free | ~known)).sum())})
            else:
                result["kept_after_city_filter"] = int(ok_city.sum())
            entry[f"top{depth}"] = result
        out[query] = entry
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--out", default=str(ROOT / "data" / "search_runs" / "hybrid_check.json"))
    parser.add_argument("--data-only", action="store_true", help="skip the ada-002 part, which needs the OpenAI key and the saved vectors")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    events = load_events(args.tables)
    report = {"data": data_report(events)}
    data = report["data"]
    print(f"events: {data['events']}, dates {data['date_range'][0]} to {data['date_range'][1]}")
    print(f"price known for {data['share_with_known_price']:.0%} of events, free for {data['share_free']:.0%} of all events ({data['share_free_among_known_price']:.0%} of those with a price)")
    print("events per city:", data["events_per_city"])

    if not args.data_only:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY must be set in .env (or use --data-only)")
        from openai import OpenAI

        vectors = np.load(ROOT / "data" / "graph" / "event_text.npy").astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        assert len(vectors) == len(events), "events.csv and the saved vectors do not match"
        queries = np.array([d.embedding for d in OpenAI().embeddings.create(model=MODEL, input=QUERIES).data], dtype=np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)
        scores = [vectors @ q for q in queries]
        report["spread"] = dict(zip(QUERIES, score_spread(scores)))
        report["constraints"] = constraint_check(scores, events)
        print("\nada-002 cosine scores, top 1 / rank 10 / rank 50, and the spread from rank 1 to rank 50:")
        for query, row in report["spread"].items():
            print(f'  "{query}": {row["top1"]:.3f} / {row["rank10"]:.3f} / {row["rank50"]:.3f}, spread {row["spread_top1_to_top50"]:.3f}; '
                  f'beta 0.05 = {row["beta_0.05_as_share_of_top50_spread"]:.0%} of it, beta 0.1 = {row["beta_0.1_as_share_of_top50_spread"]:.0%}')
        print("\nconstraint check on the ada-002 results:")
        for query, entry in report["constraints"].items():
            for depth in ("top10", "top50"):
                print(f'  "{query}" {depth}: {entry[depth]}')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
