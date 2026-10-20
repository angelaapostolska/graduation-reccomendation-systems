"""Embedding models used by the search scripts: OpenAI API models and a local DistilBERT."""

from __future__ import annotations

import os

MODELS = {
    "text-embedding-3-small": {"column": "embedding", "dim": 1536, "price": 0.02, "backend": "openai"},
    "text-embedding-ada-002": {"column": "embedding_ada002", "dim": 1536, "price": 0.10, "backend": "openai"},
    "distilbert": {"column": "embedding_distilbert", "dim": 768, "price": 0.0, "backend": "local"},
}
DISTILBERT_CHECKPOINT = os.environ.get("DISTILBERT_CHECKPOINT", "sentence-transformers/distilbert-base-nli-stsb-mean-tokens")

_local_model = None
_openai_client = None


def embed(model: str, texts: list[str]) -> tuple[list[list[float]], int]:
    """Returns one vector per text and the number of billed tokens (0 for local models)."""
    if MODELS[model]["backend"] == "local":
        global _local_model
        if _local_model is None:
            from sentence_transformers import SentenceTransformer

            _local_model = SentenceTransformer(DISTILBERT_CHECKPOINT)
        return [v.tolist() for v in _local_model.encode(texts, batch_size=32, show_progress_bar=False)], 0
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        _openai_client = OpenAI()
    response = _openai_client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data], response.usage.total_tokens


def literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7f}" for x in vector) + "]"


def check_env(model: str) -> str:
    url = os.environ.get("DATABASE_URL")
    if not url or (MODELS[model]["backend"] == "openai" and not os.environ.get("OPENAI_API_KEY")):
        raise SystemExit("DATABASE_URL (and OPENAI_API_KEY for OpenAI models) must be set in .env")
    return url
