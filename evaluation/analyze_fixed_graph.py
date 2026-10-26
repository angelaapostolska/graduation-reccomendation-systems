"""Final analysis of the graph methods against plain ada-002, from human grades: normal texts (20 searches) and stripped texts (10 searches).
Usage: python evaluation/analyze_fixed_graph.py --runs data/search_runs --grades evaluation/grades_owner_final.csv [--first8 evaluation/grades_first8.csv] [--out evaluation/results_fixed_graph.md]
Strict P@10 counts grade 2, lenient P@10 grade >= 1, nDCG@10 uses the grade as gain (ideal ranking from all graded events of the search that any listed method returned).
Events in a top 10 that have no grade count as 0 and are reported; the blank-card stripped lists were not graded completely, so their range is shown."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

HEAD = re.compile(r"\s+(\d+)\. \[([\d.\-]+)\] (.*)")
DATE = re.compile(r"^\d{4}-\d\d-\d\d")
STRIPPED_SEARCHES = ["free techno night in Skopje", "cheap standup comedy in Skopje", "live jazz concert this weekend", "hip-hop party with a student discount",
                     "live rock band tonight", "stand up comedy in English", "karaoke night", "salsa dancing lessons", "underground electronic music", "beach party with DJ"]
NORMAL = {"ada-002": "text-embedding-ada-002_exact", "3-small": "text-embedding-3-small_exact", "3-large": "text-embedding-3-large_exact",
          "bge-small": "bge-small_exact", "MiniLM": "minilm_exact", "DistilBERT": "distilbert_exact",
          "GNN, blank cards": "gnn", "GNN, fixed graph": "fixed_gnn", "Smoothing, fixed graph": "fixed_smooth"}
STRIPPED = {"Stripped text (ada-002)": "stripped_text", "Stripped smoothing, blank cards": "stripped_smooth", "Stripped GNN, blank cards": "stripped_gnn",
            "Stripped smoothing, fixed graph": "stripped_fixed_smooth", "Stripped GNN, fixed graph": "stripped_fixed_gnn"}
# rows where the page export split an event name at " | " and lost the rest: (query, name as exported, date as exported) -> (full name, date, venue)
# grades you changed after re-reading the nine serious splits; the grading page database still holds the old value
CORRECTIONS = {("kids workshop on the weekend", "Detska Kreativna Rabotilnica", "2025-10-19", "Public Room"): 2,
               ("brunch with live music", "Sunday Brunch", "2026-08-30", "Amigos Zeleznicka"): 2}
REPAIRS = {("beach party with DJ", "Pena Party", "DJ Toshevv"): ("Pena Party | DJ Toshevv", "2025-08-17", "Pool Bar & Restaurant Oasis"),
           ("free entry party for students", "Student Party", "3VOX"): ("Student Party | 3VOX", "2025-12-10", "Kadmo"),
           ("salsa dancing lessons", "Pate Conga", "Igor"): ("Pate Conga | Igor Sax + DJ Tose", "2026-02-20", "Cuban Cocktail Bar")}


def parse(path: Path) -> dict[str, list[tuple]]:
    runs, cur = {}, None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("query searched:"):
            cur = re.match(r'query searched: "(.*)" \(model', line).group(1)
            runs[cur] = []
        elif HEAD.match(line) and cur:
            parts = [p.strip() for p in HEAD.match(line).group(3).split(" | ")]
            i = next(k for k, p in enumerate(parts) if DATE.match(p))
            place = parts[i + 1]
            runs[cur].append((" | ".join(parts[:i]), parts[i].split()[0], place.rpartition(", ")[0] or place))
    return runs


def read_grades(paths: list[Path]) -> dict[tuple, int]:
    grades = {}
    for path in paths:
        for row in csv.DictReader(open(path, encoding="utf-8-sig", newline="")):
            if row["grade"].strip() not in {"0", "1", "2"}:
                continue
            key = (row["query"], row["event"], row["date"], row["venue"])
            fixed = REPAIRS.get((row["query"], row["event"], row["date"]))
            if fixed:
                key = (row["query"], *fixed)
            grades[key] = int(row["grade"])
    return grades | CORRECTIONS


def sign_test(diffs) -> tuple[int, int, float]:
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    return wins, losses, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap(diffs, n=10000):
    rng = np.random.default_rng(2026)
    d = np.asarray(diffs, float)
    return np.percentile([rng.choice(d, len(d)).mean() for _ in range(n)], [2.5, 97.5])


def evaluate(runs: dict, searches: list[str], grades: dict, fill: int = 0):
    pool = {q: sorted({(q, *it): grades[(q, *it)] for P in runs.values() for it in P[q][:10] if (q, *it) in grades}.values(), reverse=True) for q in searches}
    out = {}
    for name, P in runs.items():
        rows, missing = [], 0
        for q in searches:
            got = [grades.get((q, *it)) for it in P[q][:10]]
            missing += sum(g is None for g in got)
            g = [fill if x is None else x for x in got]
            dcg = sum(x / math.log2(r + 2) for r, x in enumerate(g))
            idcg = sum(x / math.log2(r + 2) for r, x in enumerate(pool[q][:10]))
            rows.append((sum(x == 2 for x in g) / 10, sum(x >= 1 for x in g) / 10, dcg / idcg if idcg else 0.0))
        out[name] = (np.array(rows), missing)
    return out


def table(rows: list[list], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows])


def compare(res: dict, a: str, b: str) -> list[str]:
    d = res[a][0][:, 0] - res[b][0][:, 0]
    wins, losses, p = sign_test(d)
    lo, hi = bootstrap(d)
    return [f"{a} minus {b}", f"{d.mean():+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"{wins} / {losses}", f"{p:.3f}"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="data/search_runs")
    parser.add_argument("--grades", required=True, help="CSV exported from the grading page")
    parser.add_argument("--first8", default="evaluation/grades_first8.csv")
    parser.add_argument("--out", default="evaluation/results_fixed_graph.md")
    args = parser.parse_args()
    grades = read_grades([Path(args.first8), Path(args.grades)])  # the grading page wins where both have a grade
    runs = Path(args.runs)
    report = []
    header = ["Method", "Strict P@10", "Lenient P@10", "nDCG@10"]
    cmp_header = ["Comparison (strict P@10)", "Mean difference", "95% bootstrap CI", "Searches wins / losses", "Sign test p"]

    normal_runs = {name: parse(runs / f"results_{f}.txt") for name, f in NORMAL.items()}
    searches = list(normal_runs["ada-002"])
    res = evaluate(normal_runs, searches, grades)
    report += [f"## Normal texts, all {len(searches)} searches", table([[n, *(f"{a[0][:, k].mean():.3f}" for k in range(3))] for n, a in res.items()], header)]
    report += ["", table([compare(res, a, b) for a, b in [("GNN, fixed graph", "GNN, blank cards"), ("GNN, fixed graph", "ada-002"), ("GNN, blank cards", "ada-002"),
                                                        ("Smoothing, fixed graph", "ada-002"), ("Smoothing, fixed graph", "GNN, fixed graph")]], cmp_header)]
    report.append(f"\nUngraded slots: {', '.join(f'{n} {m}' for n, (_, m) in res.items() if m) or 'none'}.")

    stripped_runs = {name: parse(runs / f"results_{f}.txt") for name, f in STRIPPED.items()}
    stripped_runs["Normal ada-002 (reference)"] = normal_runs["ada-002"]
    low = evaluate(stripped_runs, STRIPPED_SEARCHES, grades, fill=0)
    high = evaluate(stripped_runs, STRIPPED_SEARCHES, grades, fill=2)
    rows = []
    for n, (a, m) in low.items():
        extra = f"{m} ungraded; strict between {a[:, 0].mean():.3f} and {high[n][0][:, 0].mean():.3f}" if m else "all graded"
        rows.append([n, *(f"{a[:, k].mean():.3f}" for k in range(3)), extra])
    report += ["", f"## Stripped texts, the {len(STRIPPED_SEARCHES)} searches that name a genre or type", table(rows, header + ["Coverage"])]
    report += ["", table([compare(low, a, b) for a, b in [("Stripped smoothing, fixed graph", "Stripped text (ada-002)"), ("Stripped GNN, fixed graph", "Stripped text (ada-002)"),
                                                        ("Stripped GNN, fixed graph", "Stripped smoothing, fixed graph"), ("Stripped text (ada-002)", "Normal ada-002 (reference)")]], cmp_header)]
    report += ["", "Strict P@10 per search (stripped text / stripped smoothing, fixed / stripped GNN, fixed / normal ada-002):"]
    for i, q in enumerate(STRIPPED_SEARCHES):
        report.append(f"  {q}: " + " / ".join(f"{low[n][0][i, 0]:.1f}" for n in ["Stripped text (ada-002)", "Stripped smoothing, fixed graph", "Stripped GNN, fixed graph", "Normal ada-002 (reference)"]))
    text = "\n".join(report)
    Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
