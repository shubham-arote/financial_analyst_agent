"""
cohere_client.py — dense embeddings + reranking via Cohere's REST API (httpx, no SDK).

Phase-1 of the hybrid-retrieval merge. Key-optional, mirroring cloud.py: set
``COHERE_API_KEY`` (free at https://dashboard.cohere.com/api-keys) and the dense + rerank
signals turn on; with no key, ``available()`` is False and HybridRetriever falls back to the
existing BM25 path. ``embed-v4.0`` for vectors, ``rerank-v3.5`` for the cross-encoder stage.

We call the REST endpoints directly with httpx (already a dependency) rather than pulling in
the ``cohere`` SDK — same reasoning as cloud.py keeping the footprint light.
"""

from __future__ import annotations

import httpx

from ..config import settings

_BASE = "https://api.cohere.com/v2"
EMBED_MODEL = settings.cohere_embed_model
RERANK_MODEL = settings.cohere_rerank_model
_EMBED_BATCH = 96


def _key() -> str | None:
    return settings.cohere_api_key


def available() -> bool:
    return bool(_key())


def _post(path: str, payload: dict, timeout: float = 60.0) -> dict | None:
    key = _key()
    if not key:
        return None
    try:
        r = httpx.post(f"{_BASE}/{path}", json=payload, timeout=timeout,
                       headers={"Authorization": f"Bearer {key}"})
        r.raise_for_status()
        return r.json()
    except Exception:
        return None  # callers degrade gracefully (fall back to BM25 / fused order)


def embed(texts: list[str], input_type: str = "search_document") -> list[list[float]] | None:
    """Embed texts with embed-v4.0. ``input_type``: 'search_document' | 'search_query'.
    Returns one float vector per text (order-preserving), or None on failure/no-key."""
    if not texts:
        return []
    out: list[list[float]] = []
    for i in range(0, len(texts), _EMBED_BATCH):
        data = _post("embed", {"model": EMBED_MODEL, "input_type": input_type,
                               "embedding_types": ["float"], "texts": texts[i:i + _EMBED_BATCH]})
        if not data:
            return None
        out.extend(data["embeddings"]["float"])
    return out


def embed_query(text: str) -> list[float] | None:
    v = embed([text], input_type="search_query")
    return v[0] if v else None


def rerank(query: str, documents: list[str], top_n: int) -> list[tuple[int, float]] | None:
    """Cross-encoder rerank. Returns [(orig_index, relevance_score), …] best-first,
    or None on failure/no-key (caller keeps the fused order)."""
    if not documents:
        return []
    data = _post("rerank", {"model": RERANK_MODEL, "query": query,
                            "documents": documents, "top_n": min(top_n, len(documents))})
    if not data:
        return None
    return [(r["index"], r["relevance_score"]) for r in data["results"]]
