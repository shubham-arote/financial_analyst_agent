"""
verify.py — numeric faithfulness check for the analyst agent.

After the answer is generated, every figure it asserts must be traceable to either a
retrieved (cited) chunk or the verified computation. Numbers that match neither are
*hallucination candidates* — we surface them as a transparent caveat rather than silently
trusting them. This complements `guards.py` (which handles prompt-injection, not numbers).
"""

from __future__ import annotations

import math
import re

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_PAGE_CITE = re.compile(r"\[page\s+\d+\]", re.I)


def _norm(s: str) -> str:
    return (s or "").replace(",", "")


def verify_numbers(answer: str, retrieved: list[dict], computation: dict | None = None) -> list[str]:
    """Return figures asserted in `answer` that aren't supported by the cited context or the
    verified computation. Page-citation numbers (``[page N]``) are ignored — they're structural,
    not claims."""
    ans = _PAGE_CITE.sub("", answer or "")
    ctx = " ".join(_norm(c.get("content") or c.get("text") or "") for c in (retrieved or []))
    comp = computation.get("result") if computation else None

    seen: set[str] = set()
    unsupported: list[str] = []
    for tok in (_norm(m.group()) for m in _NUM.finditer(ans)):
        if tok in seen:
            continue
        seen.add(tok)
        if tok in ctx:                                  # appears verbatim in cited context
            continue
        try:
            val = float(tok)
        except ValueError:
            continue
        if comp is not None and math.isclose(val, comp, rel_tol=0.01, abs_tol=0.01):
            continue                                     # equals the verified computation
        unsupported.append(tok)
    return unsupported
