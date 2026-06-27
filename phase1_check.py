"""Phase-1 verification: HybridRetriever fallback (no key) + fusion path (mock embedder)."""
from app.srr.core import Block, BBox, BlockType
from app.engine.graph import DocIndex
from app.engine.retriever import Retriever
from app.engine import cohere_client
from app.engine import hybrid


def make_index():
    def blk(t, content, page, order, bid):
        b = Block(type=t, bbox=BBox(0, order * 30, 100, order * 30 + 20), page=page)
        b.content = content; b.order = order; b.id = bid
        return b
    blocks = [
        blk(BlockType.TITLE, "Revenue", 1, 0, "b1"),
        blk(BlockType.TEXT, "Total revenue was $48.2 million in Q2 2026.", 1, 1, "b2"),
        blk(BlockType.TITLE, "Operating margin", 2, 0, "b3"),
        blk(BlockType.TEXT, "Operating margin improved to 12% year over year.", 2, 1, "b4"),
    ]
    return DocIndex.from_blocks(blocks)


# 1) No-key fallback: build_retriever must return the plain DocIndex, app unchanged.
idx = make_index()
r = hybrid.build_retriever(idx)
print("no-key: cohere.available =", cohere_client.available())
print("no-key: build_retriever ->", type(r).__name__, "(expect DocIndex)")
assert r is idx, "without a key, must be the unchanged DocIndex"
print("no-key: retrieve ok, hits =", len(r.retrieve("revenue", k=2)))

# 2) Fusion path with a MOCK embedder (proves dense + RRF + Evidence shape run end-to-end).
dim = 16
def _fake_vec(seed: str):
    import random
    rnd = random.Random(seed)
    return [rnd.uniform(-1, 1) for _ in range(dim)]

cohere_client.available = lambda: True
cohere_client.embed = lambda texts, input_type="search_document": [_fake_vec(t) for t in texts]
cohere_client.embed_query = lambda t: _fake_vec("Q:" + t)
cohere_client.rerank = lambda q, docs, top_n: None     # exercise the fused-order branch

hr = hybrid.HybridRetriever(idx)
print("\nmock: dense_on =", hr.dense_on, "| is Retriever =", isinstance(hr, Retriever))
hits = hr.retrieve("what was the operating margin", k=3)
print("mock: hits =", len(hits))
h = hits[0]
need = {"page", "bbox", "block_id", "type", "text", "content", "score",
        "parent_text", "section_heading"}
print("mock: contract keys present =", need.issubset(h.keys()))
print("mock: top hit -> page", h["page"], "bbox", h["bbox"], "score", round(h["score"], 4))
print("classify: 'operating margin' =", hybrid.classify("what is the operating margin"),
      "| 'compare 2025 vs 2024' =", hybrid.classify("compare 2025 vs 2024"),
      "| 'why did profit fall' =", hybrid.classify("why did profit fall"))
print("\nPHASE 1 OK")
