"""Phase-1b verification: deterministic table/number lookup floats exact value/period matches."""
from collections import Counter
from evals.run_eval import build_index
from app.engine.hybrid import HybridRetriever, lookup_terms

idx = build_index()
print("chunk types:", dict(Counter(c.get("type") for c in idx.chunks)))
hr = HybridRetriever(idx)

QUERIES = [
    ("What was revenue in FY25?", [4]),       # period anchor FY25
    ("Which line item shows 1,052?", [4]),    # explicit value -> operating profit
    ("capital expenditure 168", [6]),         # explicit value
    ("operating profit in FY26", [4]),        # period anchor FY26
]
for q, expected in QUERIES:
    terms = lookup_terms(q)
    hits = hr.retrieve(q, k=5)
    exact = [(h["page"], (h.get("content") or h["text"])[:40].replace("\n", " "))
             for h in hits if h.get("exact")]
    top_pages = [h["page"] for h in hits]
    got = "HIT " if set(expected) & set(top_pages) else "MISS"
    print(f"\nQ: {q}")
    print(f"   lookup_terms : {terms}")
    print(f"   exact-floated: {exact}")
    print(f"   top pages    : {top_pages}  expected {expected} -> {got}")
print("\nPHASE 1b OK")
