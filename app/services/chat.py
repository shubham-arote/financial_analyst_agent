"""
chat.py — ask orchestration.

Binds a document's current index to the configured retriever (hybrid Cohere dense + RRF +
rerank, or Qdrant, or plain BM25 — chosen by `make_retriever`) and returns an agent over it.
The retriever is cached on the doc per *finalized index*, so we embed the chunks once rather
than on every question. The server's `/ws/ask` route is a thin loop over this.
"""

from __future__ import annotations

from ..agent.graph import AgentEngine
from ..retrieval import make_retriever


def get_engine(doc: dict, doc_id: str) -> AgentEngine:
    """Return an AgentEngine bound to the doc's retriever, (re)building the retriever only when
    the underlying index changes (e.g. a PDF finished parsing)."""
    if doc.get("_retr_for") is not doc["index"]:
        doc["_retriever"] = make_retriever(doc_id, doc["index"])
        doc["_retr_for"] = doc["index"]
    return AgentEngine(doc["_retriever"])
