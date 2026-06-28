r"""
qdrant_store.py — QdrantRetriever: the deploy-grade (Gold) retrieval backend.

Same hybrid strategy as the in-memory `HybridRetriever`, but **stateless and external** — the
week2 production pattern (validated against `D:\…\week2`): named **dense + sparse** vectors in
Qdrant with **RRF fused inside the database**, then a **Cohere rerank** pass, plus our
**deterministic table/number lookup** (a payload filter, not a vector search) and **page+bbox
grounding** carried in the payload. Returns the same `Evidence` shape, so the agent / grounding /
abstain logic are unchanged when this swaps in for the in-memory store at deploy.

    ingest  blocks → parent-child chunks (reuse DocIndex) → Cohere embed-v4 (dense) +
            fastembed BM25 (sparse) → upsert points {dense, sparse, payload(text, page, bbox,
            parent_text, …, doc_id, lookup_keys)}
    query   ① deterministic: payload filter on exact value/period keys (no vector) → exact hits
            ② hybrid: dense+sparse → query_points(prefetch=[…], FusionQuery RRF, limit=30)
            ③ rerank: Cohere rerank-v3.5 → top-k
            ④ merge ① above ③ → Evidence

Store selection mirrors week2: `QDRANT_MODE=cloud` (uses `QDRANT_URL`+`QDRANT_API_KEY`) else local
/ in-memory. Fully verifiable with `QdrantClient(":memory:")` — no server required.
"""

from __future__ import annotations

import uuid

from qdrant_client import QdrantClient, models

from . import cohere_client
from ..config import settings
from .graph import DocIndex
from .hybrid import lookup_terms
from .retriever import Evidence

_DENSE_DIM = 1536                       # Cohere embed-v4.0
_PAYLOAD_KEYS = ("text", "content", "page", "bbox", "block_id", "section_id",
                 "type", "heading", "parent_text", "section_heading")
_sparse_model = None                    # lazy fastembed BM25 (sparse, IDF — not a neural model)


def _bm25():
    global _sparse_model
    if _sparse_model is None:
        from fastembed import SparseTextEmbedding
        _sparse_model = SparseTextEmbedding("Qdrant/bm25")
    return _sparse_model


def _sparse(text: str, query: bool = False) -> models.SparseVector:
    emb = _bm25().query_embed([text]) if query else _bm25().embed([text])
    e = next(iter(emb))
    return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())


def get_client() -> QdrantClient:
    """Cloud (`QDRANT_MODE=cloud`) or local/in-memory — same selection as week2."""
    if settings.qdrant_mode == "cloud":
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    loc = settings.qdrant_location
    return QdrantClient(loc) if loc == ":memory:" else QdrantClient(url=loc)


def ensure_collection(client: QdrantClient, collection: str) -> None:
    if client.collection_exists(collection):
        return
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=_DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)})
    # doc scoping + the deterministic exact-match lookup both run on indexed payload fields.
    client.create_payload_index(collection, "doc_id", models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(collection, "lookup_keys", models.PayloadSchemaType.KEYWORD)


def index_doc(client: QdrantClient, collection: str, doc_id: str, index: DocIndex) -> int:
    """Upsert one parsed doc's parent-child chunks (dense + sparse + grounded payload)."""
    ensure_collection(client, collection)
    # idempotent re-index: drop any existing points for this doc before re-upserting
    client.delete(collection, points_selector=models.FilterSelector(
        filter=models.Filter(must=[models.FieldCondition(
            key="doc_id", match=models.MatchValue(value=doc_id))])))
    chunks = index.chunks
    if not chunks:
        return 0
    index._attach_parents(chunks)                         # denormalize parent_text/section_heading
    dense = cohere_client.embed([c["text"] for c in chunks], "search_document")
    if not dense:
        raise RuntimeError("Cohere embedding failed — set COHERE_API_KEY")
    points = []
    for i, c in enumerate(chunks):
        body = c.get("content") or c["text"]
        payload = {k: c.get(k) for k in _PAYLOAD_KEYS}
        payload["doc_id"] = doc_id
        payload["lookup_keys"] = lookup_terms(body)        # exact values/periods in this chunk
        points.append(models.PointStruct(
            id=uuid.uuid4().hex,
            vector={"dense": dense[i], "sparse": _sparse(c["text"])},
            payload=payload))
    client.upsert(collection, points)
    return len(points)


class QdrantRetriever:
    """Retriever bound to one `doc_id` over a shared Qdrant collection. Satisfies the
    `Retriever` protocol structurally (`.retrieve(query, k) -> list[Evidence]`)."""

    def __init__(self, client: QdrantClient, collection: str, doc_id: str):
        self.client = client
        self.collection = collection
        self.doc_id = doc_id

    def _doc_filter(self, extra: list | None = None) -> models.Filter:
        must = [models.FieldCondition(key="doc_id", match=models.MatchValue(value=self.doc_id))]
        return models.Filter(must=must + (extra or []))

    def retrieve(self, query: str, k: int = 6) -> list[Evidence]:
        # ① Deterministic table/number lookup — exact value/period match via payload filter (no vector).
        det: list[dict] = []
        terms = lookup_terms(query)
        if terms:
            flt = self._doc_filter([models.FieldCondition(
                key="lookup_keys", match=models.MatchAny(any=terms))])
            pts, _ = self.client.scroll(self.collection, scroll_filter=flt, limit=3, with_payload=True)
            det = [p.payload for p in pts]

        # ② Hybrid dense + sparse → RRF fused inside Qdrant (doc-scoped).
        prefetch = [models.Prefetch(query=_sparse(query, query=True), using="sparse",
                                    limit=30, filter=self._doc_filter())]
        qv = cohere_client.embed_query(query)
        if qv:
            prefetch.insert(0, models.Prefetch(query=qv, using="dense",
                                               limit=30, filter=self._doc_filter()))
        cand = self.client.query_points(self.collection, prefetch=prefetch,
                                        query=models.FusionQuery(fusion=models.Fusion.RRF),
                                        limit=30, with_payload=True).points

        # ③ Cross-encoder rerank the fused candidates → top-k.
        docs = [(p.payload.get("content") or p.payload.get("text", "")) for p in cand]
        rr = cohere_client.rerank(query, docs, top_n=k) if docs else None
        reranked = ([(cand[i].payload, sc) for i, sc in rr] if rr
                    else [(p.payload, p.score) for p in cand[:k]])

        # ④ Merge: deterministic exact hits floated ABOVE the reranked, dedup by block_id.
        out, seen = [], set()
        for pl in det:
            key = pl.get("block_id") or id(pl)
            if key not in seen:
                seen.add(key)
                out.append({**pl, "score": 1.0, "exact": True})
        for pl, sc in reranked:
            key = pl.get("block_id") or id(pl)
            if key not in seen:
                seen.add(key)
                out.append({**pl, "score": float(sc), "exact": False})
        return out[:max(k, len(det))]


def build_qdrant_retriever(doc_id: str, index: DocIndex,
                           client: QdrantClient | None = None,
                           collection: str | None = None) -> QdrantRetriever:
    """Index a doc into Qdrant and return a retriever bound to it. (Deploy: pass a shared
    cloud client; the server would index once at parse time and reuse the retriever.)"""
    client = client or get_client()
    collection = collection or settings.qdrant_collection
    index_doc(client, collection, doc_id, index)
    return QdrantRetriever(client, collection, doc_id)
