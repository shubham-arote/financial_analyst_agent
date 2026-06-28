"""
index.py — DocIndex: parent-child chunking + BM25 + reranking (the in-process retrieval core).

Extracted from the former engine/graph.py so retrieval lives in the retrieval layer, not
inside the agent. Small-to-big retrieval: BM25 matches precise child blocks (citations point to
the exact spot), a reranker reorders candidates, and the LLM gets each match's parent *section*
for coherent context. `HybridRetriever` (hybrid.py) composes this same DocIndex.

`_tok` / `_STOP` are the shared text helpers (tokenizer + stopwords) used by BM25, the reranker,
and the agent's offline fallbacks; they live here as the retrieval/text primitives.
"""

from __future__ import annotations

import re

from ..config import settings
from ..srr import cloud
from ..srr.core import FURNITURE, BlockType
from .base import Evidence

_STOP = {"the", "a", "an", "of", "to", "in", "for", "and", "or", "is", "was", "were",
         "are", "what", "which", "how", "much", "many", "did", "do", "does", "tell",
         "me", "about", "could", "you", "please", "on", "at", "by", "with", "this",
         "that", "it", "its", "their", "from", "be", "as", "we", "i"}


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9$%.]+", s.lower())


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def chunk_markdown(markdown: str, page: int = 0) -> list[dict]:
    """Split assembled Markdown into heading-tagged paragraph/table chunks."""
    chunks: list[dict] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            tagged = f"{heading}\n{text}" if heading else text
            chunks.append({"text": tagged, "page": page, "heading": heading or "(top)"})
        buf = []

    for line in markdown.splitlines():
        if line.startswith("#"):
            flush()
            heading = line.lstrip("# ").strip()
        elif not line.strip():
            flush()
        else:
            buf.append(line)
    flush()
    return chunks


def _child(text, content, page, heading, b, sec_id) -> dict:
    return {"text": text, "content": content, "page": page, "heading": heading or "(top)",
            "type": b.type.value, "block_id": b.id, "section_id": sec_id,
            "bbox": [int(b.bbox.x0), int(b.bbox.y0), int(b.bbox.x1), int(b.bbox.y1)]}


# --------------------------------------------------------------------------- #
# Reranking — reorder BM25 candidates before truncating to k
#   SRR_RERANK = heuristic (default) | llm | off
# --------------------------------------------------------------------------- #
def _rerank(query: str, cands: list[dict]) -> list[dict]:
    mode = settings.rerank
    if mode == "off" or not cands:
        return cands
    if mode == "llm" and cloud.has_cloud():
        return _rerank_llm(query, cands)
    return _rerank_heuristic(query, cands)


def _rerank_heuristic(query: str, cands: list[dict]) -> list[dict]:
    """Lexical rerank: BM25 + exact-term / numeric / heading / table boosts."""
    qterms = {w for w in _tok(query) if w not in _STOP and len(w) > 2}
    qnums = set(re.findall(r"\d[\d.,]{2,}", query))      # real values (6,303 / 5.9); skip "FY26" -> "26"

    def score(c: dict) -> float:
        s = c.get("score", 0.0)
        text = (c.get("content") or c["text"]).lower()
        s += 0.4 * len(qterms & set(_tok(text)))
        s += 1.2 * sum(1 for n in qnums if n in text)
        if any(w in (c.get("heading") or "").lower() for w in qterms):
            s += 0.6
        if qnums and c.get("type") == "table":
            s += 0.4
        return s

    return sorted(cands, key=score, reverse=True)


def _rerank_llm(query: str, cands: list[dict]) -> list[dict]:
    """Ask the cloud LLM to order the candidates (cheap, cloud-first; no local model)."""
    lst = "\n".join(f"[{i}] (p{c['page']}) {(c.get('content') or c['text'])[:160]}"
                    for i, c in enumerate(cands[:24]))
    resp = cloud.chat_text(
        f"Rank the candidates by relevance to the query. Return ONLY candidate numbers, "
        f"best first, comma-separated.\n\nQuery: {query}\n\nCandidates:\n{lst}", max_tokens=60)
    seen, ranked = set(), []
    for i in (int(x) for x in re.findall(r"\d+", resp)):
        if i < len(cands) and i not in seen:
            seen.add(i); ranked.append(cands[i])
    ranked += [c for i, c in enumerate(cands) if i not in seen]   # keep any the LLM dropped
    return ranked


# --------------------------------------------------------------------------- #
# DocIndex — BM25 over child chunks, parent sections for context
# --------------------------------------------------------------------------- #
class DocIndex:
    """BM25 over small (block-level) child chunks, with parent *sections* for context.

    Small-to-big retrieval: BM25 matches precise child blocks (so citations point to the
    exact spot), a reranker reorders the candidates, and we hand the LLM each match's parent
    *section* (heading + its blocks) for coherent context. Swap BM25 for embeddings via the
    same .retrieve() shape; the section/parent layer stays the same.
    """

    def __init__(self, chunks: list[dict], sections: list[dict] | None = None):
        self.chunks = chunks
        self.sections = sections or []
        from rank_bm25 import BM25Okapi
        self._bm25 = BM25Okapi([_tok(c["text"]) for c in chunks]) if chunks else None

    @classmethod
    def from_pages(cls, pages: list[tuple[int, str]]) -> "DocIndex":
        chunks: list[dict] = []
        for page_no, md in pages:
            for c in chunk_markdown(md, page_no):
                c.setdefault("content", c["text"]); c["section_id"] = None
                chunks.append(c)
        return cls(chunks)

    @classmethod
    def from_blocks(cls, blocks, default_page: int = 1) -> "DocIndex":
        """Children = blocks (precise, with bbox); parents = sections (a heading + the blocks
        under it, per page) for small-to-big context. Works for one page or a whole doc."""
        chunks: list[dict] = []
        sections: list[dict] = []
        heading = ""; cur_page = None; cur_sec = None

        def open_section(page, head) -> int:
            sections.append({"page": page, "heading": head or "(top)", "text": "", "block_ids": []})
            return len(sections) - 1

        # iterate in reading order so a heading precedes its body/table (correct sections)
        blocks = sorted(blocks, key=lambda b: (getattr(b, "page", default_page) or default_page,
                                               b.order if b.order is not None else 1e9))
        for b in blocks:
            page = getattr(b, "page", default_page) or default_page
            if page != cur_page:
                heading, cur_page, cur_sec = "", page, None
            if b.type in FURNITURE:
                continue
            content = (b.content or "").strip()
            if b.type == BlockType.TITLE:
                heading = content or heading
                cur_sec = open_section(page, heading)
                sections[cur_sec]["text"] = content
                sections[cur_sec]["block_ids"].append(b.id)
                chunks.append(_child(content, content, page, heading, b, cur_sec))
                continue
            if not content:
                continue
            if cur_sec is None:
                cur_sec = open_section(page, heading)
            sec = sections[cur_sec]
            sec["text"] = (sec["text"] + "\n" + content).strip()
            sec["block_ids"].append(b.id)
            text = f"{heading}\n{content}" if heading else content
            chunks.append(_child(text, content, page, heading, b, cur_sec))
        return cls(chunks, sections)

    def _attach_parents(self, ranked: list[dict]) -> list[dict]:
        """Attach each hit's parent *section* text + heading (small-to-big context).
        Shared with HybridRetriever so the context handed to the LLM is identical
        regardless of which retriever ranked the hits."""
        for c in ranked:
            sid = c.get("section_id")
            sec = self.sections[sid] if (sid is not None and sid < len(self.sections)) else None
            c["parent_text"] = (sec["text"] if sec else (c.get("content") or c["text"]))[:1500]
            c["section_heading"] = sec["heading"] if sec else c.get("heading", "")
        return ranked

    def retrieve(self, query: str, k: int = 6, candidates: int = 24) -> list[Evidence]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_tok(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        cands = [{**self.chunks[i], "score": float(scores[i])} for i in order[:max(candidates, k)]]
        ranked = _rerank(query, cands)[:k]
        return self._attach_parents(ranked)
