"""
hybrid.py — HybridRetriever: dense + sparse + RRF + Cohere rerank, behind the Retriever seam.

Phase-1 of the flagship merge (ports manual-rag-api's retrieval design). It *composes* an
existing DocIndex — which already owns chunking, the parent-child sections, the bbox-carrying
child chunks, and the BM25 model — and layers on:

  • a **dense** signal: Cohere embed-v4 over the same child chunks (in-memory cosine ANN;
    pgvector swaps in at deploy time);
  • **Reciprocal Rank Fusion** of dense + BM25 with per-query-type weights (ported);
  • a **Cohere rerank-v3.5** cross-encoder stage over the fused candidate window.

It returns the same `Evidence` shape DocIndex does (carrying page + bbox), so the agent,
grounding, and abstain logic are untouched. Key-optional: with no Cohere key, ``retrieve``
delegates to the DocIndex BM25 path — byte-for-byte today's behaviour.
"""

from __future__ import annotations

import re

import numpy as np

from . import cohere_client
from ..config import settings
from .graph import DocIndex, _tok
from .retriever import Evidence, Retriever

# query-type -> (dense_weight, bm25_weight)   [ported from manual-rag-api]
_TYPE_WEIGHTS = {
    "lookup":     (0.4, 0.6),   # exact spec / number / line-item  -> favour keyword
    "procedure":  (0.5, 0.5),
    "diagnostic": (0.7, 0.3),   # symptom / why / cause            -> favour semantic
    "comparison": (0.6, 0.4),
    "general":    (0.5, 0.5),
}
_RRF_K = 60

_LOOKUP = re.compile(r"\b(capacity|torque|pressure|spec|specification|value|amount|rate|"
                     r"ratio|margin|revenue|profit|eps|figure|total|how much|how many)\b", re.I)
_PROC = re.compile(r"\b(how to|how do|steps?|procedure|process|install|replace|remove|"
                   r"calculate|compute)\b", re.I)
_DIAG = re.compile(r"\b(why|cause|reason|fault|error|fail|decline|drop|fell|impact|risk)\b", re.I)
_COMP = re.compile(r"\b(vs|versus|compare|comparison|difference|between|year-?over-?year|"
                   r"yoy|prior year)\b", re.I)


def classify(query: str) -> str:
    """Zero-LLM query routing -> retrieval strategy (drives RRF weights)."""
    if _COMP.search(query):
        return "comparison"
    if _DIAG.search(query):
        return "diagnostic"
    if _PROC.search(query):
        return "procedure"
    if _LOOKUP.search(query):
        return "lookup"
    return "general"


# --------------------------------------------------------------------------- #
# Deterministic table/number lookup (Phase 1b) — exact match, not vector similarity
# --------------------------------------------------------------------------- #
_VALUE = re.compile(r"\d[\d.,]{2,}")                            # 6,303 · 12.1 · 168 · 2025 · 23.8
_PERIOD = re.compile(r"\b(?:FY\d{2,4}|Q[1-4]|H[12])\b", re.I)   # fiscal periods


def lookup_terms(query: str) -> list[str]:
    """Exact tokens a deterministic lookup should match against table cells: explicit
    values + fiscal periods. Empty -> the query has no exact anchor, lookup is skipped."""
    terms = set(_VALUE.findall(query))
    terms |= {m.group(0).upper() for m in _PERIOD.finditer(query)}
    return sorted(terms, key=len, reverse=True)


def table_lookup(chunks: list[dict], terms: list[str], limit: int = 3) -> list[int]:
    """Deterministic exact-match: indices of chunks whose content contains the most query
    terms (exact substring), preferring TABLE chunks. Mirrors manual-rag-api's TableQuerier —
    float exact cell hits to the top, bypassing fuzzy ranking. High precision (exact value/
    period substrings), so safe to *guarantee* into the result set.

    Qdrant deploy equivalent (Phase 3): a filter-only query over per-row `table_row` points,
    no vector — exact payload match, same Evidence (page+bbox) out:
        client.query_points(COLL, query_filter=models.Filter(must=[
            models.FieldCondition(key="row_text", match=models.MatchText(t)) for t in terms]),
            limit=limit)
    (with a TEXT payload index on `row_text`; KEYWORD index on normalized `metric`/`periods`)."""
    if not terms:
        return []
    scored = []
    for i, c in enumerate(chunks):
        text = (c.get("content") or c.get("text") or "").lower()
        n = sum(1 for t in terms if t.lower() in text)
        if n:
            scored.append((n, c.get("type") == "table", i))    # more terms, then table chunks
    scored.sort(reverse=True)
    return [i for _, _, i in scored[:limit]]


class HybridRetriever:
    """Wraps a DocIndex; adds a dense signal + RRF fusion + cross-encoder rerank.
    Satisfies the `Retriever` protocol structurally."""

    def __init__(self, index: DocIndex):
        self.index = index
        self._mat: np.ndarray | None = None   # (n_chunks, dim), L2-normalized
        if cohere_client.available() and index.chunks:
            vecs = cohere_client.embed([c["text"] for c in index.chunks], "search_document")
            if vecs:
                m = np.asarray(vecs, dtype="float32")
                self._mat = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-8, None)

    @property
    def dense_on(self) -> bool:
        return self._mat is not None

    def retrieve(self, query: str, k: int = 6, candidates: int = 30) -> list[Evidence]:
        idx = self.index
        if not idx._bm25:
            return []
        if not self.dense_on:                          # no embeddings -> today's BM25 path
            return idx.retrieve(query, k)

        n = len(idx.chunks)
        # --- sparse ranks (BM25) ---
        bm = idx._bm25.get_scores(_tok(query))
        bm_rank = {int(ci): r for r, ci in enumerate(np.argsort(-bm))}
        # --- dense ranks (cosine) ---
        qv = cohere_client.embed_query(query)
        if not qv:                                     # query embed failed -> fall back
            return idx.retrieve(query, k)
        qv = np.asarray(qv, dtype="float32")
        qv /= max(float(np.linalg.norm(qv)), 1e-8)
        d_rank = {int(ci): r for r, ci in enumerate(np.argsort(-(self._mat @ qv)))}
        # --- Reciprocal Rank Fusion with query-type weights ---
        w_dense, w_bm25 = _TYPE_WEIGHTS[classify(query)]
        fused = sorted(
            ((w_dense / (_RRF_K + d_rank.get(ci, n)) + w_bm25 / (_RRF_K + bm_rank.get(ci, n)), ci)
             for ci in range(n)),
            reverse=True,
        )
        cand_idx = [ci for _, ci in fused[:max(candidates, k)]]
        # --- cross-encoder rerank over the fused candidate window ---
        docs = [(idx.chunks[ci].get("content") or idx.chunks[ci]["text"]) for ci in cand_idx]
        rr = cohere_client.rerank(query, docs, top_n=k)
        if rr:
            top = [cand_idx[i] for i, _ in rr]
            scores = {cand_idx[i]: sc for i, sc in rr}
        else:                                          # rerank unavailable -> keep fused order
            top = cand_idx[:k]
            scores = dict(((ci, s) for s, ci in fused[:k]))
        # Deterministic table/number lookup: float exact value/period matches above the fuzzy
        # ranking (guaranteed inclusion — the finance-critical exact path). See table_lookup().
        det = table_lookup(idx.chunks, lookup_terms(query))
        order = det + [ci for ci in top if ci not in set(det)]
        hits = [{**idx.chunks[ci],
                 "score": float(scores.get(ci, 1.0 if ci in det else 0.0)),
                 "exact": ci in det}
                for ci in order[:max(k, len(det))]]
        return idx._attach_parents(hits)               # shared small-to-big context


def build_retriever(index: DocIndex) -> Retriever:
    """Hybrid retriever when Cohere is configured, else the plain BM25 DocIndex (unchanged)."""
    return HybridRetriever(index) if cohere_client.available() else index


# --------------------------------------------------------------------------- #
# Vector-store selection: in-memory hybrid (default) | external Qdrant (deploy)
# --------------------------------------------------------------------------- #
_QDRANT = None


def _qdrant_client():
    global _QDRANT
    if _QDRANT is None:
        from .qdrant_store import get_client
        _QDRANT = get_client()                 # shared across docs (in-memory locally / Qdrant Cloud)
    return _QDRANT


def make_retriever(doc_id: str, index: DocIndex) -> Retriever:
    """Pick the retrieval backend by `SRR_VECTOR_STORE`: `qdrant` -> external Qdrant hybrid store
    (week2 pattern, stateless/deploy); otherwise the in-memory hybrid/BM25. `qdrant_store` is
    imported lazily so qdrant-client/fastembed stay optional for the lean image."""
    if settings.vector_store == "qdrant" and cohere_client.available():
        from .qdrant_store import build_qdrant_retriever
        return build_qdrant_retriever(doc_id, index, client=_qdrant_client())
    return build_retriever(index)
