"""Final analysis of the six-model comparison from human grades.
Usage: python evaluation/analyze.py --runs data/search_runs --grades grades_owner.csv [--second grades_second.csv|second_sheet.csv|second_tabs.xlsx] [--out evaluation/results]
Grades are 0, 1 or 2. Strict P@10 counts grade 2, lenient P@10 counts grade >= 1, nDCG@10 uses the grade as gain."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np

MODELS = ["distilbert", "minilm", "bge-small", "text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
SHORT = {"distilbert": "DistilBERT", "minilm": "MiniLM", "bge-small": "bge-small", "text-embedding-ada-002": "ada-002",
         "text-embedding-3-small": "3-small", "text-embedding-3-large": "3-large"}
SMALL = ["distilbert", "minilm", "bge-small"]
LARGE = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
PAIRS = [("text-embedding-ada-002", "text-embedding-3-small"), ("text-embedding-3-large", "text-embedding-ada-002"),
         ("text-embedding-3-large", "text-embedding-3-small"), ("bge-small", "minilm"), ("bge-small", "distilbert"),
         ("minilm", "distilbert"), ("text-embedding-3-small", "bge-small"), ("text-embedding-ada-002", "bge-small"),
         ("text-embedding-3-large", "bge-small")]
ORIGINAL_8 = 8
HEAD = re.compile(r"\s+(\d+)\. \[([\d.\-]+)\] (.*)")


def read_run(path: Path) -> tuple[list[str], list[list[tuple[str, str, str]]]]:
    queries, ranked = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("query searched:"):
            queries.append(re.match(r'query searched: "(.*)" \(model', line).group(1))
            ranked.append([])
        elif HEAD.match(line) and ranked:
            parts = [p.strip() for p in HEAD.match(line).group(3).split(" | ")]
            place = parts[2]
            venue = place.rpartition(", ")[0] or place
            ranked[-1].append((parts[0], parts[1], venue))
    return queries, ranked


def load_runs(folder: Path):
    runs, queries = {}, None
    for model in MODELS:
        queries, runs[model] = read_run(folder / f"results_{model}_exact.txt")
    return queries, runs


def read_grade_csv(path: Path) -> dict[tuple[str, str, str], int]:
    out = {}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    header = [h.strip().lower() for h in rows[0]]
    if "query" in header:  # export of the grading page
        i = {k: header.index(k) for k in ("query", "event", "date", "venue", "grade")}
        for r in rows[1:]:
            if len(r) > i["grade"] and r[i["grade"]].strip() in {"0", "1", "2"}:
                out[(r[i["query"]], r[i["event"]], r[i["date"]])] = int(r[i["grade"]])
        return out
    # export of the single-tab Google Sheet: Search, Event, Date, ..., grade in the column headed YOUR GRADE
    i = {"query": header.index("search"), "event": header.index("event"), "date": header.index("date")}
    g = next(k for k, h in enumerate(header) if h.startswith("your grade"))
    for r in rows[1:]:
        if len(r) > g and r[g].strip() in {"0", "1", "2"}:
            out[(r[i["query"]], r[i["event"]], r[i["date"]])] = int(r[g].strip())
    return out


def read_grade_tabs(path: Path, queries: list[str]) -> dict[tuple[str, str, str], int]:
    """Export of the sheet with one tab per search (tabs named Q01_..., Q02_...; columns Event, Details, Description, GRADE)."""
    from openpyxl import load_workbook

    wb = load_workbook(path, data_only=True)
    out = {}
    for ws in wb.worksheets:
        m = re.match(r"Q(\d+)_", ws.title)
        if not m:
            continue
        query = queries[int(m.group(1)) - 1]
        for row in ws.iter_rows(min_row=4, values_only=True):
            if not row or row[0] is None or row[3] is None or str(row[3]).strip() not in {"0", "1", "2", "0.0", "1.0", "2.0"}:
                continue
            date = str(row[1]).strip()[:10]
            out[(query, str(row[0]).strip(), date)] = int(float(row[3]))
    return out


def read_grades(path: Path, queries: list[str]):
    return read_grade_tabs(path, queries) if path.suffix.lower() == ".xlsx" else read_grade_csv(path)


def grade_of(grades, query, item):
    return grades.get((query, item[0], item[1]))


def metrics_for(query, ranked, pool_grades, grades):
    top = ranked[:10]
    got = [grade_of(grades, query, it) for it in top]
    missing = sum(g is None for g in got)
    g = [x if x is not None else 0 for x in got]
    strict = sum(x == 2 for x in g) / 10
    lenient = sum(x >= 1 for x in g) / 10
    dcg = sum(x / math.log2(r + 2) for r, x in enumerate(g))
    ideal = sorted(pool_grades, reverse=True)[:10]
    idcg = sum(x / math.log2(r + 2) for r, x in enumerate(ideal))
    return strict, lenient, (dcg / idcg if idcg else 0.0), missing


def sign_test(diffs):
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return wins, losses, p


def bootstrap_ci(diffs, rng, n=10000):
    d = np.asarray(diffs, dtype=float)
    means = np.array([rng.choice(d, size=len(d), replace=True).mean() for _ in range(n)])
    return np.percentile(means, [2.5, 97.5])


def kappa(a, b, labels, weights=None):
    n = len(a)
    idx = {l: i for i, l in enumerate(labels)}
    obs = np.zeros((len(labels), len(labels)))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= n
    exp = np.outer(obs.sum(1), obs.sum(0))
    w = np.zeros_like(obs)
    for i in range(len(labels)):
        for j in range(len(labels)):
            w[i, j] = (0 if i == j else 1) if weights is None else abs(i - j) / (len(labels) - 1)
    den = (w * exp).sum()
    return 1.0 if den == 0 else 1 - (w * obs).sum() / den


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |"
    return "\n".join([line(header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"] + [line(r) for r in rows])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="data/search_runs")
    parser.add_argument("--grades", required=True, help="your grades (CSV exported from the grading page)")
    parser.add_argument("--second", default="", help="second grader: CSV from the grading page, CSV of the one-tab sheet, or XLSX of the tabbed sheet")
    parser.add_argument("--out", default="evaluation/results")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    queries, runs = load_runs(Path(args.runs))
    grades = read_grades(Path(args.grades), queries)
    rng = np.random.default_rng(2026)
    report = []

    # pooled graded grades per query, for the ideal ranking of nDCG
    pool = {}
    for q, query in enumerate(queries):
        seen = {}
        for model in MODELS:
            for it in runs[model][q][:10]:
                seen[(it[0], it[1])] = grades.get((query, it[0], it[1]))
        pool[q] = seen
    missing_total = sum(v is None for q in pool for v in pool[q].values())
    pooled_total = sum(len(pool[q]) for q in pool)
    report.append(f"Pooled distinct events: {pooled_total}. Graded: {pooled_total - missing_total}. Still without a grade: {missing_total}.")
    if missing_total:
        for q in pool:
            n = sum(v is None for v in pool[q].values())
            if n:
                report.append(f"  - search {q + 1} ({queries[q]}): {n} ungraded (counted as 0 until graded)")

    per = {m: [] for m in MODELS}
    for q, query in enumerate(queries):
        pg = [v for v in pool[q].values() if v is not None]
        for m in MODELS:
            per[m].append(metrics_for(query, runs[m][q], pg, grades)[:3])
    arr = {m: np.array(per[m]) for m in MODELS}

    def mean_rows(sl):
        rows = []
        for m in MODELS:
            a = arr[m][sl]
            rows.append([SHORT[m], "small" if m in SMALL else "large", *(f"{a[:, k].mean():.3f}" for k in range(3))])
        return rows

    hdr = ["Model", "Group", "Strict P@10", "Lenient P@10", "nDCG@10"]
    report += ["", f"## All {len(queries)} searches", table(mean_rows(slice(None)), hdr),
               "", f"## Original {ORIGINAL_8} searches (comparable with the earlier three-model results)", table(mean_rows(slice(0, ORIGINAL_8)), hdr)]
    with open(out / "model_means.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["search"] + [f"{SHORT[m]}_{k}" for m in MODELS for k in ("strict", "lenient", "ndcg")])
        for q in range(len(queries)):
            w.writerow([queries[q]] + [f"{arr[m][q][k]:.3f}" for m in MODELS for k in range(3)])

    names = ["strict P@10", "lenient P@10", "nDCG@10"]
    grp = [["Group", *names]]
    for label, ms in (("small", SMALL), ("large", LARGE)):
        grp.append([label, *(f"{np.mean([arr[m][:, k].mean() for m in ms]):.3f}" for k in range(3))])
    report += ["", "## H1: small against large, paired over searches (difference = large minus small)"]
    rows = []
    for k, name in enumerate(names):
        d = np.mean([arr[m][:, k] for m in LARGE], axis=0) - np.mean([arr[m][:, k] for m in SMALL], axis=0)
        w, l, p = sign_test(d)
        lo, hi = bootstrap_ci(d, rng)
        rows.append([name, f"{d.mean():+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"{w} / {l}", f"{p:.4f}"])
    report.append(table(rows, ["Measure", "Mean difference", "95% bootstrap CI", "Searches large wins / loses", "Sign test p"]))

    report += ["", "## Model pairs on strict P@10 (first minus second; sign test over searches, not corrected for multiple tests)"]
    rows = []
    for a, b in PAIRS:
        d = arr[a][:, 0] - arr[b][:, 0]
        w, l, p = sign_test(d)
        lo, hi = bootstrap_ci(d, rng)
        rows.append([f"{SHORT[a]} vs {SHORT[b]}", f"{d.mean():+.3f}", f"[{lo:+.3f}, {hi:+.3f}]", f"{w} / {l}", f"{p:.4f}"])
    report.append(table(rows, ["Pair", "Mean difference", "95% bootstrap CI", "Wins / loses", "Sign test p"]))

    report += ["", "## Control: bge-small (small, trained for retrieval) against the other small models and against the large models",
               table([[SHORT[m], f"{arr[m][:, 0].mean():.3f}"] for m in ["bge-small", "minilm", "distilbert", "text-embedding-3-small"]], ["Model", "Strict P@10"])]

    if args.second:
        second = read_grades(Path(args.second), queries)
        shared = [k for k in second if k in grades]
        report += ["", "## Agreement between the two graders"]
        if shared:
            a = [grades[k] for k in shared]
            b = [second[k] for k in shared]
            agree = np.mean([x == y for x, y in zip(a, b)])
            report.append(f"Events graded by both: {len(shared)} (second grader graded {len(second)}).")
            report.append(f"Exact agreement: {agree:.1%}. Cohen's kappa: {kappa(a, b, [0, 1, 2]):.3f}. Linear weighted kappa: {kappa(a, b, [0, 1, 2], weights=True):.3f}.")
            report.append(f"Kappa for relevant against the rest (grade 2): {kappa([x == 2 for x in a], [y == 2 for y in b], [False, True]):.3f}. "
                          f"Kappa for any relevance (grade >= 1): {kappa([x >= 1 for x in a], [y >= 1 for y in b], [False, True]):.3f}.")
            with open(out / "disagreements.csv", "w", newline="", encoding="utf-8") as h:
                w = csv.writer(h)
                w.writerow(["query", "event", "date", "you", "second"])
                for k in shared:
                    if grades[k] != second[k]:
                        w.writerow([*k, grades[k], second[k]])
            report.append(f"Disagreements are listed in {out / 'disagreements.csv'}.")
        else:
            report.append("No events graded by both yet.")

    text = "\n".join(report)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
