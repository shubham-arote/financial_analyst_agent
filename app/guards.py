"""
guards.py — minimal LLM safety rails (2026 standard layers, scoped to this app).

  * Input validation  — reject empty/oversized questions and obvious prompt-injection in the query.
  * Retrieval rail     — scan retrieved DOCUMENT text for injection patterns. Document content is
                         untrusted input flowing into the prompt; we flag it (and the generate
                         prompt is hardened to treat context as data, never instructions).

This is a pragmatic subset, not a full moderation stack — but it covers the highest-risk path
for a doc-QA agent: a malicious instruction hidden inside an uploaded PDF.
"""

from __future__ import annotations

import re

_INJECT = [re.compile(p, re.I) for p in (
    r"ignore\s+(all\s+|the\s+|any\s+)?(previous|above|prior|earlier)\s+(instruction|prompt|context)",
    r"disregard\s+(the\s+|all\s+)?(previous|above|prior)?\s*instruction",
    r"forget\s+(everything|all|the|previous)",
    r"system\s+prompt",
    r"you\s+are\s+now\b",
    r"new\s+instructions?\s*:",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"act\s+as\s+(a|an|the)\b",
)]
MAX_Q = 600


def scan_text(text: str) -> list[str]:
    """Return the matched injection snippets found in the text (readable labels)."""
    hits = []
    for rx in _INJECT:
        m = rx.search(text or "")
        if m:
            hits.append(" ".join(m.group(0).split())[:40])
    return hits


def check_question(q: str) -> tuple[bool, str]:
    q = (q or "").strip()
    if not q:
        return False, "Empty question."
    if len(q) > MAX_Q:
        return False, "Question is too long — please shorten it."
    if scan_text(q):
        return False, "That looks like a prompt-injection attempt and was blocked."
    return True, ""


def scan_context(chunks) -> list[str]:
    """Flag injection patterns hidden in retrieved document text (the retrieval rail)."""
    flags: list[str] = []
    for c in chunks:
        for h in scan_text(c.get("content") or c.get("text") or ""):
            flags.append(f"p{c.get('page')}:{h}")
    return flags
