"""Shared graph loading, message passing and model code for the GNN path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
GRAPH_DIR = ROOT / "data" / "graph"
EXCLUDED_RELATIONS = {"performer__associated_with__music_genre"}
TARGETS = {
    "genre": ("event__has_genre__music_genre", "music_genre"),
    "event_type": ("event__has_event_type__event_type", "event_type"),
}
TARGET_RELATIONS = {relation for relation, _ in TARGETS.values()}


def load_graph(path: Path = GRAPH_DIR / "graph.npz") -> dict:
    data = np.load(path, allow_pickle=False)
    kinds = [key[4:] for key in data.files if key.startswith("ids_")]
    return {
        "counts": {kind: len(data[f"ids_{kind}"]) for kind in kinds},
        "ids": {kind: data[f"ids_{kind}"] for kind in kinds},
        "edges": {key[5:]: torch.from_numpy(data[key]) for key in data.files if key.startswith("edge_")},
        "genre_source": data["event_genre_source"],
        "is_synthetic": data["event_is_synthetic"],
    }


def relation_ends(name: str) -> tuple[str, str]:
    parts = name.split("__")
    return parts[0], parts[2]


def message_edges(graph: dict, hidden_events: torch.Tensor | None) -> dict:
    """Edges used for message passing, in both directions. Genre and event-type links of hidden events are left out."""
    out = {}
    for name, edge in graph["edges"].items():
        if name in EXCLUDED_RELATIONS:
            continue
        source, target = edge
        if hidden_events is not None and name in TARGET_RELATIONS:
            keep = ~hidden_events[source]
            source, target = source[keep], target[keep]
        source_kind, target_kind = relation_ends(name)
        out[name] = (source_kind, target_kind, source, target)
        out[name + "_rev"] = (target_kind, source_kind, target, source)
    return out


def initial_features(graph: dict, event_vectors: torch.Tensor, ignore_events: torch.Tensor | None) -> dict:
    """Event nodes start as their text vector; every other node starts as the mean of the nodes connected to it."""
    features = {"event": event_vectors}
    dim = event_vectors.shape[1]
    sums = {kind: torch.zeros(count, dim) for kind, count in graph["counts"].items() if kind != "event"}
    counts = {kind: torch.zeros(count) for kind, count in graph["counts"].items() if kind != "event"}
    for name, edge in graph["edges"].items():
        if name in EXCLUDED_RELATIONS:
            continue
        source_kind, target_kind = relation_ends(name)
        if source_kind != "event":
            continue
        source, target = edge
        if ignore_events is not None and name in TARGET_RELATIONS:
            keep = ~ignore_events[source]
            source, target = source[keep], target[keep]
        sums[target_kind].index_add_(0, target, event_vectors[source])
        counts[target_kind].index_add_(0, target, torch.ones(len(target)))
    for name, edge in graph["edges"].items():
        source_kind, target_kind = relation_ends(name)
        if name in EXCLUDED_RELATIONS or source_kind == "event" or source_kind not in sums:
            continue
        source, target = edge
        mean_source = sums[source_kind] / counts[source_kind].clamp(min=1).unsqueeze(1)
        if target_kind in sums:
            sums[target_kind].index_add_(0, target, mean_source[source])
            counts[target_kind].index_add_(0, target, torch.ones(len(target)))
    for kind in sums:
        features[kind] = F.normalize(sums[kind] / counts[kind].clamp(min=1).unsqueeze(1), dim=1)
    return features


class Layer(nn.Module):
    def __init__(self, relations: list[str], dim: int, rank: int):
        super().__init__()
        self.down = nn.ModuleDict({r: nn.Linear(dim, rank, bias=False) for r in relations})
        self.up = nn.ModuleDict({r: nn.Linear(rank, dim, bias=False) for r in relations})
        for module in self.up.values():
            nn.init.zeros_(module.weight)

    def forward(self, h: dict, edges: dict) -> dict:
        new = {kind: value.clone() for kind, value in h.items()}
        for name, (source_kind, target_kind, source, target) in edges.items():
            message = self.down[name](h[source_kind])[source]
            total = torch.zeros(h[target_kind].shape[0], message.shape[1]).index_add_(0, target, message)
            degree = torch.zeros(h[target_kind].shape[0]).index_add_(0, target, torch.ones(len(target)))
            new[target_kind] = new[target_kind] + self.up[name](F.relu(total / degree.clamp(min=1).unsqueeze(1)))
        return new


class Net(nn.Module):
    def __init__(self, graph: dict, dim: int, rank: int = 64, layers: int = 2, hidden: int = 128):
        super().__init__()
        relations = [n for n in message_edges(graph, None)]
        self.layers = nn.ModuleList([Layer(relations, dim, rank) for _ in range(layers)])
        self.event_side = nn.ModuleDict({t: nn.Linear(dim, hidden) for t in TARGETS})
        self.class_side = nn.ModuleDict({t: nn.Linear(dim, hidden) for t in TARGETS})

    def encode(self, features: dict, edges: dict) -> dict:
        h = features
        for layer in self.layers:
            h = layer(h, edges)
        return h

    def scores(self, target: str, h: dict, events: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        class_kind = TARGETS[target][1]
        return (self.event_side[target](h["event"][events]) * self.class_side[target](h[class_kind][classes])).sum(-1)

    def all_scores(self, target: str, h: dict, events: torch.Tensor) -> torch.Tensor:
        class_kind = TARGETS[target][1]
        return self.event_side[target](h["event"][events]) @ self.class_side[target](h[class_kind]).T


def smooth(features: dict, edges: dict, alpha: float = 0.5, iterations: int = 2) -> dict:
    """Training-free baseline: every node takes in the mean of its neighbours."""
    h = features
    for _ in range(iterations):
        new = {kind: value.clone() for kind, value in h.items()}
        for name, (source_kind, target_kind, source, target) in edges.items():
            total = torch.zeros_like(h[target_kind]).index_add_(0, target, h[source_kind][source])
            degree = torch.zeros(h[target_kind].shape[0]).index_add_(0, target, torch.ones(len(target)))
            new[target_kind] = new[target_kind] + alpha * total / degree.clamp(min=1).unsqueeze(1)
        h = {kind: F.normalize(value, dim=1) for kind, value in new.items()}
    return h
