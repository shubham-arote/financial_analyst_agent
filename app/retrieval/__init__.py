"""Retrieval layer — the Retriever seam + concrete backends.

The agent depends only on `Retriever` / `Evidence` (base.py). The server builds a
backend with `make_retriever`. Concrete backends — `DocIndex` (BM25, parent-child),
`HybridRetriever` (dense + RRF + Cohere rerank), and the Qdrant store (lazy) — all
satisfy the `Retriever` protocol structurally.
"""

from .base import Evidence, Retriever
from .index import DocIndex
from .hybrid import (HybridRetriever, build_retriever, make_retriever,
                     classify, lookup_terms)
from . import cohere_client

__all__ = ["Evidence", "Retriever", "DocIndex", "HybridRetriever",
           "build_retriever", "make_retriever", "classify", "lookup_terms",
           "cohere_client"]
