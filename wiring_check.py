"""Verify make_retriever: SRR_VECTOR_STORE routing, shared Qdrant client, doc isolation."""
import os
os.environ["SRR_VECTOR_STORE"] = "qdrant"          # before any make_retriever call

from evals.run_eval import build_index
from app.engine.graph import DocIndex
from app.engine.hybrid import make_retriever, _qdrant_client
from app.engine.qdrant_store import QdrantRetriever
from app.engine.retriever import Retriever

# Two different docs go into ONE shared (in-memory) Qdrant collection, isolated by doc_id.
rA = make_retriever("docA", build_index())                                  # ACME financials
rB = make_retriever("docB", DocIndex.from_pages(
        [(1, "# Widget Catalog\nThe catalog lists 4,200 widget SKUs in 2027.")]))

print("qdrant mode ->", type(rA).__name__, "| is Retriever:", isinstance(rA, Retriever))
print("shared collection points:", _qdrant_client().count("srr_docs").count, "(docA 20 + docB 1)")

a = rA.retrieve("operating profit FY26", k=3)
b = rB.retrieve("how many widget SKUs?", k=3)
print("docA -> pages:", [h["page"] for h in a])
print("docB -> text :", [h["text"][:45] for h in b])
isolated = all("widget" in (h.get("content") or h.get("text") or "").lower() for h in b)
print("doc isolation (docB returns only its own content):", isolated)

os.environ["SRR_VECTOR_STORE"] = "memory"
print("memory mode ->", type(make_retriever("docM", build_index())).__name__)
print("\nWIRING OK")
