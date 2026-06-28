"""Retrieval-layer unit tests (salvaged from the former phase1_check.py).

Covers the two guarantees the hybrid layer must keep:
  * key-optional degradation — with no Cohere key, build_retriever returns the
    plain BM25 DocIndex unchanged (the app must never hard-depend on a key)
  * the fusion path — dense + RRF runs end-to-end and yields the Evidence
    contract (exercised with a deterministic mock embedder, so no network)
  * the zero-LLM query classifier routes query types
"""

from app.srr.core import Block, BBox, BlockType
from app.engine.graph import DocIndex
from app.engine.retriever import Retriever
from app.engine import cohere_client, hybrid

CONTRACT_KEYS = {"page", "bbox", "block_id", "type", "text", "content",
                 "score", "parent_text", "section_heading"}


def _make_index() -> DocIndex:
    def blk(t, content, page, order, bid):
        b = Block(type=t, bbox=BBox(0, order * 30, 100, order * 30 + 20), page=page)
        b.content = content
        b.order = order
        b.id = bid
        return b

    return DocIndex.from_blocks([
        blk(BlockType.TITLE, "Revenue", 1, 0, "b1"),
        blk(BlockType.TEXT, "Total revenue was $48.2 million in Q2 2026.", 1, 1, "b2"),
        blk(BlockType.TITLE, "Operating margin", 2, 0, "b3"),
        blk(BlockType.TEXT, "Operating margin improved to 12% year over year.", 2, 1, "b4"),
    ])


def test_no_key_returns_plain_docindex():
    if cohere_client.available():
        return  # a real key is set in this env; the fallback branch isn't exercised
    idx = _make_index()
    r = hybrid.build_retriever(idx)
    assert r is idx, "without a Cohere key, build_retriever must return the unchanged DocIndex"
    assert r.retrieve("revenue", k=2)


def test_fusion_path_with_mock_embedder(monkeypatch):
    idx = _make_index()
    dim = 16

    def fake_vec(seed: str):
        import random
        rnd = random.Random(seed)
        return [rnd.uniform(-1, 1) for _ in range(dim)]

    monkeypatch.setattr(cohere_client, "available", lambda: True)
    monkeypatch.setattr(cohere_client, "embed",
                        lambda texts, input_type="search_document": [fake_vec(t) for t in texts])
    monkeypatch.setattr(cohere_client, "embed_query", lambda t: fake_vec("Q:" + t))
    monkeypatch.setattr(cohere_client, "rerank", lambda q, docs, top_n: None)  # fused-order branch

    hr = hybrid.HybridRetriever(idx)
    assert isinstance(hr, Retriever)
    hits = hr.retrieve("what was the operating margin", k=3)
    assert hits and CONTRACT_KEYS.issubset(hits[0].keys())


def test_query_classifier():
    assert hybrid.classify("compare 2025 vs 2024") == "comparison"
    assert hybrid.classify("why did profit fall") == "diagnostic"
