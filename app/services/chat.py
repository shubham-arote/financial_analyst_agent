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


def _doc_retriever(doc: dict, doc_id: str):
    """The doc's retriever, (re)built only when its index changes (embed once, not per question)."""
    if doc.get("_retr_for") is not doc["index"]:
        doc["_retriever"] = make_retriever(doc_id, doc["index"])
        doc["_retr_for"] = doc["index"]
    return doc["_retriever"]


def get_engine(doc: dict, doc_id: str) -> AgentEngine:
    """Agent over a single document."""
    return AgentEngine(_doc_retriever(doc, doc_id))


def get_compare_engine(docs: dict[str, dict]) -> AgentEngine:
    """Agent over several documents (cross-document compare). `docs` maps doc_id -> doc state;
    each is labelled 'Document N' so the agent attributes figures to the right one."""
    from ..retrieval import MultiDocRetriever
    retrievers = {f"Document {i}": _doc_retriever(doc, doc_id)
                  for i, (doc_id, doc) in enumerate(docs.items(), 1)}
    return AgentEngine(MultiDocRetriever(retrievers), multi_doc=True)
