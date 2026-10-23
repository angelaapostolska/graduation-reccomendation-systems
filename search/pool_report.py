"""Pilot report, no grading: for each query, how many different events the models return in their top 10, how much they overlap,
and how many of the pooled events still need a grade. Usage: python search/pool_report.py [--runs data/search_runs] [--grades evaluation/grades_first8.csv]"""

from __future__ import annotations

import argparse
import csv
import itertools
import re
from pathlib import Path

from embedders import MODELS
from testing_query import QUERIES

ROOT = Path(__file__).resolve().parent.parent


def read_picks(path: Path) -> list[list[str]]:
    """Events (name|date) listed under each query of a saved results file, in rank order."""
    picks: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("query searched:"):
            picks.append([])
        elif re.match(r"\s+\d+\. \[", line) and picks:
            name, day = [part.strip() for part in line.split("] ", 1)[1].split(" | ")[:2]]
            picks[-1].append(f"{name}|{day}")
    return picks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default=str(ROOT / "data" / "search_runs"))
    parser.add_argument("--grades", default=str(ROOT / "evaluation" / "grades_first8.csv"))
    args = parser.parse_args()
    runs = {m: read_picks(Path(args.runs) / f"results_{m}_exact.txt") for m in MODELS if (Path(args.runs) / f"results_{m}_exact.txt").exists()}
    missing = [m for m in MODELS if m not in runs]
    if missing:
        print(f"no exact results yet for: {', '.join(missing)}")
    graded: dict[str, set[str]] = {}
    with open(args.grades, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            graded.setdefault(row["query"], set()).add(f"{row['event']}|{row['date']}")

    total_pool = total_new = 0
    print(f"{'query':58} {'pool':>5} {'to grade':>9}")
    for q, query in enumerate(QUERIES):
        pool = set(itertools.chain.from_iterable(picks[q] for picks in runs.values()))
        new = pool - graded.get(query, set())
        total_pool, total_new = total_pool + len(pool), total_new + len(new)
        print(f"{query[:58]:58} {len(pool):>5} {len(new):>9}")
    print(f"{'total':58} {total_pool:>5} {total_new:>9}\n")

    print("mean overlap between two models (shared events of the top 10, averaged over queries):")
    names = list(runs)
    print(" " * 12 + "".join(f"{n[:10]:>11}" for n in names))
    for a in names:
        row = [sum(len(set(runs[a][q]) & set(runs[b][q])) for q in range(len(QUERIES))) / len(QUERIES) for b in names]
        print(f"{a[:11]:12}" + "".join(f"{v:11.1f}" for v in row))
    print("\nevents each model adds that no other model returned (summed over queries):")
    for a in names:
        print(f"  {a}: {sum(len(set(runs[a][q]) - set().union(*[set(runs[b][q]) for b in names if b != a])) for q in range(len(QUERIES)))}")


if __name__ == "__main__":
    main()
