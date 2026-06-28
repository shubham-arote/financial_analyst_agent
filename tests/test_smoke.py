"""Fast, offline end-to-end smoke gate — run after every refactor phase.

Exercises the three seams most likely to break during the restructure:
  * the HTTP app boots and serves (server wiring + imports)
  * an index builds and retrieval returns the Evidence contract (retrieval layer)
  * the agent constructs and produces an answer + sources (agent layer)

No network and no API keys required: retrieval is BM25, generation is forced
down the extractive path, the docstore/checkpointer are isolated in conftest.
"""

from fastapi.testclient import TestClient

from app.server import app
from app.srr.core import Block, BBox, BlockType
from app.engine.graph import AgentEngine, DocIndex
from app.engine.hybrid import build_retriever

# Evidence fields the agent + UI depend on (the retrieval contract).
EVIDENCE_KEYS = {"page", "bbox", "block_id", "type", "text", "content", "score"}


def _make_index() -> DocIndex:
    """A tiny 2-page doc with a known fact to retrieve against."""
    def blk(t, content, page, order, bid):
        b = Block(type=t, bbox=BBox(0, order * 30, 100, order * 30 + 20), page=page)
        b.content = content
        b.order = order
        b.id = bid
        return b

    blocks = [
        blk(BlockType.TITLE, "Revenue", 1, 0, "b1"),
        blk(BlockType.TEXT, "Total revenue was $48.2 million in Q2 2026.", 1, 1, "b2"),
        blk(BlockType.TITLE, "Operating margin", 2, 0, "b3"),
        blk(BlockType.TEXT, "Operating margin improved to 12% year over year.", 2, 1, "b4"),
    ]
    return DocIndex.from_blocks(blocks)


def test_http_healthz_and_status():
    with TestClient(app) as client:
        h = client.get("/healthz")
        assert h.status_code == 200
        assert h.json().get("status") == "ok"

        s = client.get("/api/status")
        assert s.status_code == 200
        assert "cloud" in s.json()


def test_retrieval_contract():
    retriever = build_retriever(_make_index())
    hits = retriever.retrieve("operating margin", k=3)
    assert hits, "retrieval returned no hits for an in-document query"
    assert EVIDENCE_KEYS.issubset(hits[0].keys()), "Evidence contract changed"
    assert any("margin" in (h.get("content") or h.get("text", "")).lower() for h in hits)


def test_engine_offline_end_to_end():
    engine = AgentEngine(build_retriever(_make_index()))
    engine.use_cloud = False  # deterministic extractive path, no network
    out = engine.run("What was the operating margin?", thread_id="smoke")
    assert isinstance(out.get("answer"), str) and out["answer"].strip()
    assert isinstance(out.get("sources"), list)
