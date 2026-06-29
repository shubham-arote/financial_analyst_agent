"""
multidoc.py — retrieve across several documents for cross-document comparison.

Wraps one `Retriever` per document (each already behind the Retriever protocol) and tags every
hit with its source-document label, so the agent can attribute each figure to the right document
(e.g. FY26 vs FY25). Satisfies the `Retriever` protocol itself, so the agent treats it like any
other retriever — it just sees hits that carry a `doc` label.
"""

from __future__ import annotations

from .base import Evidence, Retriever


class MultiDocRetriever:
    """Retrieve from each wrapped document and tag hits with their document label."""

    def __init__(self, retrievers: dict[str, Retriever]):
        self._retrievers = retrievers          # {label: retriever}

    def retrieve(self, query: str, k: int = 6) -> list[Evidence]:
        n = max(1, len(self._retrievers))
        per = max(2, k // n)                    # keep every document represented in the context
        hits: list[Evidence] = []
        for label, r in self._retrievers.items():
            for h in r.retrieve(query, per):
                tagged = dict(h)
                tagged["doc"] = label
                hits.append(tagged)
        hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        return hits
