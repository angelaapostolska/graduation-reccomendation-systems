"""Embedding models used by the search scripts: OpenAI API models and local sentence-transformers models."""

from __future__ import annotations

import os

MODELS = {
    "text-embedding-3-small": {"column": "embedding", "dim": 1536, "price": 0.02, "backend": "openai"},
    "text-embedding-ada-002": {"column": "embedding_ada002", "dim": 1536, "price": 0.10, "backend": "openai"},
    "text-embedding-3-large": {"column": "embedding_3large", "dim": 3072, "price": 0.13, "backend": "openai"},
    "distilbert": {"column": "embedding_distilbert", "dim": 768, "price": 0.0, "backend": "local"},
    "minilm": {"column": "embedding_minilm", "dim": 384, "price": 0.0, "backend": "local"},
    "bge-small": {"column": "embedding_bge_small", "dim": 384, "price": 0.0, "backend": "local"},
}
DISTILBERT_CHECKPOINT = os.environ.get("DISTILBERT_CHECKPOINT", "sentence-transformers/distilbert-base-nli-stsb-mean-tokens")
CHECKPOINTS = {
    "distilbert": DISTILBERT_CHECKPOINT,
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
}
# bge-small is trained with this prefix on the query side only (events are embedded without it)
QUERY_PREFIX = {"bge-small": "Represent this sentence for searching relevant passages: "}
# pgvector can build HNSW/IVFFlat indexes only up to this many dimensions; a bigger column is stored fine and searched exactly
MAX_INDEXED_DIM = 2000

_local_models: dict = {}
_openai_client = None


def embed(model: str, texts: list[str], is_query: bool = False) -> tuple[list[list[float]], int]:
    """Returns one vector per text and the number of billed tokens (0 for local models). is_query adds the model's query prefix, if it has one."""
    if MODELS[model]["backend"] == "local":
        if model not in _local_models:
            from sentence_transformers import SentenceTransformer

            _local_models[model] = SentenceTransformer(CHECKPOINTS[model])
        if is_query:
            texts = [QUERY_PREFIX.get(model, "") + text for text in texts]
        # normalised vectors; cosine similarity in the database is unaffected
        return [v.tolist() for v in _local_models[model].encode(texts, batch_size=32, show_progress_bar=False)], 0
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
