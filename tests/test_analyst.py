"""Analyst layer — the supervisor router + the deterministic calculate node.

Retrieval is the plain BM25 DocIndex (offline); the calculate node's LLM call is
monkeypatched, so the whole file runs with no network and no keys.
"""

import math

from app.srr.core import Block, BBox, BlockType
from app.retrieval import DocIndex
from app.agent.graph import AgentEngine
from app.agent.verify import verify_numbers
from app.srr import cloud


def _make_index() -> DocIndex:
    def blk(t, content, page, order, bid):
        b = Block(type=t, bbox=BBox(0, order * 30, 100, order * 30 + 20), page=page)
        b.content = content
        b.order = order
        b.id = bid
        return b

    return DocIndex.from_blocks([
        blk(BlockType.TITLE, "Income statement", 1, 0, "b1"),
        blk(BlockType.TEXT, "Operating profit was 1,052 in FY26 and 985 in FY25.", 1, 1, "b2"),
        blk(BlockType.TITLE, "Registered office", 2, 0, "b3"),
        blk(BlockType.TEXT, "The registered office is in London.", 2, 1, "b4"),
    ])


def _engine() -> AgentEngine:
    return AgentEngine(_make_index())   # DocIndex is a Retriever — BM25, offline


def test_supervisor_routes_math_to_calc():
    eng = _engine()
    out = eng._supervise({"original_question": "what was the YoY growth in operating profit?",
                          "question": "x"})
    assert out["task"] == "calc"


def test_supervisor_routes_lookup_to_qa():
    eng = _engine()
    out = eng._supervise({"original_question": "where is the registered office?",
                          "question": "x"})
    assert out["task"] == "qa"


def test_calculate_node_is_exact(monkeypatch):
    eng = _engine()
    eng.use_cloud = True
    # the calculate node asks the LLM for an arithmetic expression; return the growth formula
    monkeypatch.setattr(cloud, "chat_text", lambda *a, **k: "(1052-985)/985*100")
    retrieved = eng.index.retrieve("operating profit FY26 FY25", 6)
    out = eng._calculate({"original_question": "YoY growth of operating profit",
                          "question": "operating profit", "retrieved": retrieved})
    assert "computation" in out
    assert math.isclose(out["computation"]["result"], (1052 - 985) / 985 * 100, rel_tol=1e-9)


def test_calculate_offline_skips():
    eng = _engine()
    eng.use_cloud = False   # no LLM to form an expression -> skip cleanly
    out = eng._calculate({"original_question": "growth?", "question": "growth",
                          "retrieved": eng.index.retrieve("operating profit", 6)})
    assert out == {}


# ── verifier ──────────────────────────────────────────────────────────────────
def test_verifier_flags_fabricated_number():
    retrieved = [{"content": "Operating profit was 1,052 in FY26.", "page": 1}]
    bad = verify_numbers("Profit was 1052, and revenue was 999.", retrieved, None)
    assert "999" in bad and "1052" not in bad


def test_verifier_accepts_computed_value():
    retrieved = [{"content": "Operating profit 1052 and 985.", "page": 1}]
    comp = {"expr": "(1052-985)/985*100", "result": 6.802030456852792}
    # 6.80 ~ the verified computation; 1052/985 aren't restated in the answer
    assert verify_numbers("The growth was 6.80%.", retrieved, comp) == []


def test_verifier_ignores_page_citations():
    retrieved = [{"content": "Revenue 500.", "page": 7}]
    assert verify_numbers("Revenue was 500 [page 7].", retrieved, None) == []
