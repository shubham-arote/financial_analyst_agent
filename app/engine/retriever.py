"""
retriever.py — the retrieval seam the agent reasons against.

The LangGraph agent (`graph.py`) does not care *how* a document is searched; it only
needs a `.retrieve(query, k)` that returns a ranked list of `Evidence` hits. Today the
sole implementation is `DocIndex` (lexical BM25 + parent-child sections + heuristic
rerank). The flagship merge adds a `HybridRetriever` — dense + sparse + RRF + cross-encoder
rerank + deterministic table lookup, ported from `manual-rag-api` — *behind this same
interface*, so the agent, grounding (page+block), and abstain logic stay untouched when the
retrieval engine is upgraded.

`Evidence` is the contract both sides agree on. It is a plain dict (`TypedDict`), so the
existing node code that reads `hit["page"]`, `hit["bbox"]`, `hit["parent_text"]`, … keeps
working with zero changes. A new retriever is "correct" iff it returns hits of this shape.
"""

from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable


class Evidence(TypedDict, total=False):
    """One ranked retrieval hit. Required: ``page``, ``text``, ``score``.
    Strongly recommended (grounding + small-to-big context depend on them):
    ``bbox``, ``block_id``, ``parent_text``, ``section_heading``."""

    # --- identity / grounding (drives the page+block citation and click-to-highlight) ---
    page: int                 # 1-based page number                              (required)
    bbox: list[int]           # [x0, y0, x1, y1] of the source block, for highlight
    block_id: str | None      # the SRR/Docling block this hit came from
    section_id: int | None    # parent-section index (small-to-big retrieval)
    type: str                 # block type: text | title | table | figure | ...

    # --- text ---
    text: str                 # searchable child text (heading + content)        (required)
    content: str              # raw block content (no heading prefix)
    heading: str              # nearest heading for the block
    parent_text: str          # parent-section text handed to the LLM (context)
    section_heading: str      # heading of the parent section

    # --- scoring ---
    score: float              # retriever score: BM25 | fused RRF | rerank        (required)
    exact: bool               # True if floated by the deterministic table/number lookup


@runtime_checkable
class Retriever(Protocol):
    """Anything the agent can retrieve from. `DocIndex` and the future
    `HybridRetriever` both satisfy this *structurally* — no subclassing needed.

    Implementations may accept extra keyword args (e.g. ``candidates=``); only
    ``query`` and ``k`` are part of the contract the agent relies on."""

    def retrieve(self, query: str, k: int = 6) -> list[Evidence]:
        """Return up to ``k`` ranked `Evidence` hits for ``query``, best first."""
        ...
