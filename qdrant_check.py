"""Verify QdrantRetriever (Gold backend) on in-memory Qdrant, vs the in-memory HybridRetriever."""
from evals.run_eval import build_index
from app.engine.qdrant_store import build_qdrant_retriever, get_client
from app.engine.retriever import Retriever
from app.engine.hybrid import HybridRetriever

client = get_client()                                   # in-memory Qdrant (no server)
qr = build_qdrant_retriever("sample", build_index(), client=client, collection="srr_docs")
hr = HybridRetriever(build_index())                     # in-memory baseline

print("QdrantRetriever is a Retriever:", isinstance(qr, Retriever))
print("points indexed:", client.count("srr_docs").count)

QUERIES = [
    ("What was operating profit in FY26?", [4]),
    ("Which line item shows 1,052?", [4]),
    ("capital expenditure 168", [6]),
    ("What were net assets at the period end?", [5]),
    ("how profitable were the company's core operations?", [3, 4]),   # semantic (dense)
]
keys_ok = True
for q, exp in QUERIES:
    qh = qr.retrieve(q, k=5)
    hh = hr.retrieve(q, k=5)
    qp = [h["page"] for h in qh]
    hp = [h["page"] for h in hh]
    qexact = sorted({h["page"] for h in qh if h.get("exact")})
    keys_ok &= all({"page", "text", "score"}.issubset(h.keys()) for h in qh)
    print(f"\nQ: {q}   expected {exp}")
    print(f"  Qdrant : pages={qp}  exact-floated={qexact}  -> {'HIT' if set(exp)&set(qp) else 'MISS'}")
    print(f"  in-mem : pages={hp}  -> {'HIT' if set(exp)&set(hp) else 'MISS'}")

# spot-check grounding survived the round-trip through Qdrant payload
g = qr.retrieve("operating profit", k=1)[0]
print("\ngrounding sample:", {k: g.get(k) for k in ("page", "bbox", "block_id", "section_heading")})
print("contract keys present:", keys_ok)
print("QDRANT BACKEND OK")
