"""Retrieval-only A/B: BM25 (DocIndex) vs Hybrid (Cohere dense + RRF + rerank).

Isolates *retrieval* quality (no LLM-judge, no answer generation) so the delta is
attributable to the retriever alone. Reports hit@5 on (a) the golden in-scope questions
and (b) paraphrased queries with little keyword overlap (where dense should win).
"""
from __future__ import annotations
import time

from evals.run_eval import build_index, GOLDEN
from app.engine.hybrid import HybridRetriever

K = 5

# Paraphrased / semantic variants — deliberately avoid the document's exact words,
# so BM25 (lexical) is stressed and the dense signal has to carry recall.
HARD = [
    {"q": "how profitable were the company's core operations?", "pages": [3, 4]},   # operating profit/margin
    {"q": "what did the firm earn before paying tax?", "pages": [3, 4]},            # profit before tax
    {"q": "money spent on new equipment and property", "pages": [6]},              # capital expenditure
    {"q": "what share of sales is left after direct production costs?", "pages": [4]},  # gross margin
    {"q": "what slice of profit went to the government?", "pages": [6]},            # effective tax rate
    {"q": "how big is the company's pile of resources it owns?", "pages": [5]},     # total assets
]


def hit_rate(retriever, items):
    hits = 0
    for it in items:
        pages = {c.get("page") for c in retriever.retrieve(it["q"], k=K)}
        if set(it["pages"]) & pages:
            hits += 1
        time.sleep(0.25)                       # be gentle on the Cohere free tier
    return hits / len(items), hits, len(items)


def main():
    idx = build_index()                        # DocIndex (BM25 baseline)
    hyb = HybridRetriever(idx)                 # dense + RRF + rerank on the SAME chunks
    print("dense_on:", hyb.dense_on, "| chunks:", len(idx.chunks))

    inscope = [g for g in GOLDEN if g["a"] is not None]
    for name, items in [("golden in-scope", inscope), ("paraphrased (semantic)", HARD)]:
        b_rate, b_h, n = hit_rate(idx, items)
        h_rate, h_h, _ = hit_rate(hyb, items)
        print(f"\n{name}  (n={n}, hit@{K})")
        print(f"  BM25   : {b_rate*100:5.1f}%  ({b_h}/{n})")
        print(f"  Hybrid : {h_rate*100:5.1f}%  ({h_h}/{n})   delta {(h_rate-b_rate)*100:+.1f}pt")

    # show a couple of concrete wins on the hard set
    print("\nper-query (paraphrased):  expected | BM25 pages | Hybrid pages")
    for it in HARD:
        bp = sorted({c.get("page") for c in idx.retrieve(it["q"], k=K)})
        hp = sorted({c.get("page") for c in hyb.retrieve(it["q"], k=K)})
        bok = "hit " if set(it["pages"]) & set(bp) else "MISS"
        hok = "hit " if set(it["pages"]) & set(hp) else "MISS"
        print(f"  {it['q'][:48]:48} {it['pages']} | {bok} {bp} | {hok} {hp}")
        time.sleep(0.25)


if __name__ == "__main__":
    main()
