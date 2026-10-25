"""Training-free graph vectors: each event takes in the average of its neighbours. Usage: python gnn/smooth.py [--alpha 0.5]"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from common import GRAPH_DIR, initial_features, load_graph, message_edges, smooth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=str(GRAPH_DIR / "event_text.npy"))
    parser.add_argument("--alpha", type=float, default=0.5, help="weight of the neighbour average")
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    graph = load_graph()
    x = F.normalize(torch.from_numpy(np.load(args.text)).float(), dim=1)
    features = initial_features(graph, x, ignore_events=None)
    vectors = smooth(features, message_edges(graph, None), alpha=args.alpha, iterations=args.iterations)["event"]
    np.save(GRAPH_DIR / "event_smooth.npy", F.normalize(vectors, dim=1).numpy())
    print(f"saved event_smooth.npy ({tuple(vectors.shape)}), mean cosine to the text vectors {float((vectors * x).sum(1).mean()):.3f}")


if __name__ == "__main__":
    main()
