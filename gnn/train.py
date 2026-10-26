"""Trains the GNN to predict hidden genre and event-type links, then saves event vectors for search.
Usage: python gnn/train.py [--epochs 300] [--tables data/tables]"""

from __future__ import annotations

import argparse
import copy
import csv
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import GRAPH_DIR, ROOT, TARGETS, Net, initial_features, load_graph, load_names, message_edges, smooth


def series_split(tables: str, count: int, fraction: float, seed: int) -> torch.Tensor:
    """Validation events chosen by series (same title at the same venue), so a series never spans both sides."""
    with open(f"{tables}/events.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == count, "events.csv and graph.npz do not match; rerun gnn/build_graph.py"
    keys = sorted({(r["name"], r["venue_id"]) for r in rows})
    rng = np.random.default_rng(seed)
    chosen = {k for k, pick in zip(keys, rng.random(len(keys)) < fraction) if pick}
    return torch.tensor([(r["name"], r["venue_id"]) in chosen for r in rows])


def hit_at(scores: np.ndarray, truth: list[set], k: int) -> float:
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([bool(set(row) & t) for row, t in zip(top, truth)]))


def truth_sets(graph: dict, relation: str, events: torch.Tensor, only_real: bool) -> dict[int, set]:
    source, target = graph["edges"][relation]
    out: dict[int, set] = {}
    event_set = set(events.tolist())
    for e, c in zip(source.tolist(), target.tolist()):
        if e in event_set and (not only_real or graph["genre_source"][e] == "real"):
            out.setdefault(e, set()).add(c)
    return out


def evaluate(net, graph, features, val: torch.Tensor, x: torch.Tensor, train_mask: torch.Tensor) -> dict:
    net.eval()
    with torch.no_grad():
        h = net.encode(features, message_edges(graph, val))
        report = {}
        val_events = val.nonzero().squeeze(1)
        for target, (relation, kind) in TARGETS.items():
            truth = truth_sets(graph, relation, val_events, only_real=(target == "genre"))
            events = torch.tensor(sorted(truth))
            wanted = [truth[e] for e in events.tolist()]
            model = net.all_scores(target, h, events).numpy()
            source, target_ids = graph["edges"][relation]
            train_edges = train_mask[source]
            counts = np.bincount(target_ids[train_edges].numpy(), minlength=graph["counts"][kind])
            prior = np.tile(counts, (len(events), 1)).astype(float)
            sims = (x[events] @ x[train_mask.nonzero().squeeze(1)].T).numpy()
            neighbours = np.argsort(-sims, axis=1)[:, :15]
            train_index = {e: i for i, e in enumerate(train_mask.nonzero().squeeze(1).tolist())}
            knn = np.zeros((len(events), graph["counts"][kind]))
            by_event: dict[int, list[int]] = {}
            for e, c in zip(source[train_edges].tolist(), target_ids[train_edges].tolist()):
                by_event.setdefault(train_index[e], []).append(c)
            for row, nbrs in enumerate(neighbours):
                for n in nbrs:
                    for c in by_event.get(int(n), []):
                        knn[row, c] += sims[row, n]
            report[target] = {
                "events": len(events),
                "gnn_hit@3": hit_at(model, wanted, 3), "text_knn_hit@3": hit_at(knn, wanted, 3), "prior_hit@3": hit_at(prior, wanted, 3),
                "gnn_hit@1": hit_at(model, wanted, 1), "text_knn_hit@1": hit_at(knn, wanted, 1), "prior_hit@1": hit_at(prior, wanted, 1),
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=str(GRAPH_DIR / "event_text.npy"))
    parser.add_argument("--tables", default=str(ROOT / "data" / "tables"))
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hide", type=float, default=0.2, help="share of training events whose links are hidden each epoch")
    parser.add_argument("--negatives", type=int, default=5)
    parser.add_argument("--drift", type=float, default=1.0, help="weight of the penalty that keeps event vectors close to their text vector")
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--named", action="store_true", help="label nodes start as the vector of their own name (run gnn/node_names.py first)")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    graph = load_graph()
    x = F.normalize(torch.from_numpy(np.load(args.text)).float(), dim=1)
    n_events = graph["counts"]["event"]
    val = series_split(args.tables, n_events, 0.15, args.seed)
    train_mask = ~val
    print(f"events: {n_events}, training {int(train_mask.sum())}, held out {int(val.sum())} (split by series)")

    names = load_names() if args.named else None
    features = initial_features(graph, x, ignore_events=val, names=names)
    net = Net(graph, x.shape[1], rank=args.rank)
    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, waited = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        net.train()
        hidden_train = train_mask & (torch.rand(n_events) < args.hide)
        h = net.encode(features, message_edges(graph, val | hidden_train))
        loss = torch.zeros(())
        for target, (relation, kind) in TARGETS.items():
            source, class_ids = graph["edges"][relation]
            pick = hidden_train[source]
            if pick.sum() == 0:
                continue
            events, positives = source[pick], class_ids[pick]
            negatives = torch.randint(graph["counts"][kind], (len(events) * args.negatives,))
            pos = net.scores(target, h, events, positives)
            neg = net.scores(target, h, events.repeat_interleave(args.negatives), negatives)
            loss = loss + F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos)) + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        loss = loss + args.drift * (1 - F.cosine_similarity(h["event"], x, dim=1)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            report = evaluate(net, graph, features, val, x, train_mask)
            score = np.mean([report[t]["gnn_hit@3"] for t in TARGETS])
            print(f"epoch {epoch:3}  loss {loss.item():.4f}  held-out hit@3: genre {report['genre']['gnn_hit@3']:.3f}, event type {report['event_type']['gnn_hit@3']:.3f}")
            if score > best:
                best, best_state, waited = score, copy.deepcopy(net.state_dict()), 0
            else:
                waited += 10
                if waited >= args.patience:
                    print("stopping early")
                    break

    net.load_state_dict(best_state)
    report = evaluate(net, graph, features, val, x, train_mask)
    print("\nheld-out events, hit@3 / hit@1 (higher is better):")
    for target, values in report.items():
        print(f"  {target} ({values['events']} events): GNN {values['gnn_hit@3']:.3f} / {values['gnn_hit@1']:.3f}, "
              f"text neighbours {values['text_knn_hit@3']:.3f} / {values['text_knn_hit@1']:.3f}, most common {values['prior_hit@3']:.3f} / {values['prior_hit@1']:.3f}")

    final_features = initial_features(graph, x, ignore_events=None, names=names)
    final_edges = message_edges(graph, None)
    net.eval()
    with torch.no_grad():
        gnn_vectors = F.normalize(net.encode(final_features, final_edges)["event"], dim=1)
        smooth_vectors = F.normalize(smooth(final_features, final_edges)["event"], dim=1)
    suffix = "_named" if args.named else ""
    np.save(GRAPH_DIR / f"event_gnn{suffix}.npy", gnn_vectors.numpy())
    np.save(GRAPH_DIR / f"event_smooth{suffix}.npy", smooth_vectors.numpy())
    torch.save(best_state, GRAPH_DIR / f"net{suffix}.pt")
    (GRAPH_DIR / f"link_prediction{suffix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    drift = float((gnn_vectors * x).sum(1).mean())
    print(f"\nmean cosine between GNN and text vectors: {drift:.3f}")
    print(f"saved event_gnn{suffix}.npy, event_smooth{suffix}.npy in {GRAPH_DIR}")


if __name__ == "__main__":
    main()
