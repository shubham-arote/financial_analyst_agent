"""
langgraph_srr_agent.py
======================
Wires the SRR pipeline (srr_pipeline.py) into a LangGraph agentic-RAG loop.

The SRR parser is the *perception tool*: ingest a document once, turn it into
layout-faithful Markdown, chunk + embed it. The agent part is the cyclic graph
that plans -> retrieves -> grades -> (rewrites & retries | generates).

    ingest ──► index ──► retrieve ──► grade ─┬─(relevant)──► generate ──► END
                            ▲                 │
                            └──(weak)── rewrite

Why a small OCR model matters here: in an agentic loop the parser may be invoked
across a whole corpus, and re-ingest/re-chunk happens often during iteration. A
0.9B–3B model that runs on one local GPU changes the economics versus a frontier
VLM API call per page.

Deps:  pip install langgraph langchain-core langchain-openai
       (any embeddings + vector store; FAISS used here for a self-contained demo)
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from srr_pipeline import SRRPipeline, build_default_pipeline


# --------------------------------------------------------------------------- #
# Graph state — everything the loop needs to carry between nodes
# --------------------------------------------------------------------------- #
class RAGState(TypedDict, total=False):
    document_path: str
    question: str
    markdown: str
    retrieved: list[str]
    answer: str
    attempts: Annotated[int, operator.add]   # accumulates across rewrite loops
    grade: str                               # "relevant" | "weak"


# --------------------------------------------------------------------------- #
# Wire-once dependencies (parser, splitter, vector store, LLM)
# --------------------------------------------------------------------------- #
class Deps:
    """Holds the heavy objects so nodes stay pure-ish. Construct once at startup."""

    def __init__(self, pipeline: SRRPipeline | None = None,
                 llm=None, embeddings=None, k: int = 4):
        self.pipeline = pipeline or build_default_pipeline()
        self.k = k
        # >>> STUB-friendly: pass real objects in production. <<<
        # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        # self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # self.embeddings = embeddings or OpenAIEmbeddings()
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = None
        self.retriever = None

    def chunk(self, markdown: str) -> list[Document]:
        # Header-aware splitting preserves the structure the SRR parser recovered.
        try:
            from langchain_text_splitters import MarkdownHeaderTextSplitter
            splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
            )
            return splitter.split_text(markdown)
        except Exception:
            # Fallback: naive paragraph chunks (keeps the demo dependency-light).
            return [Document(page_content=p) for p in markdown.split("\n\n") if p.strip()]

    def build_index(self, docs: list[Document]):
        if self.embeddings is None:               # >>> STUB <<<
            self._stub_docs = docs
            return
        from langchain_community.vectorstores import FAISS
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

    def retrieve(self, query: str) -> list[str]:
        if self.retriever is None:                # >>> STUB <<< keyword fallback
            hits = [d.page_content for d in getattr(self, "_stub_docs", [])
                    if any(w.lower() in d.page_content.lower() for w in query.split())]
            return hits[: self.k] or [d.page_content for d in self._stub_docs[: self.k]]
        return [d.page_content for d in self.retriever.invoke(query)]


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def make_graph(deps: Deps):

    def ingest(state: RAGState) -> RAGState:
        """STRUCTURE+RECOGNITION+RELATION happen here, behind a single call."""
        path = state["document_path"]
        result = (deps.pipeline.parse_pdf(path) if path.lower().endswith(".pdf")
                  else _parse_image_path(deps, path))
        return {"markdown": result.markdown}

    def index(state: RAGState) -> RAGState:
        deps.build_index(deps.chunk(state["markdown"]))
        return {}

    def retrieve(state: RAGState) -> RAGState:
        return {"retrieved": deps.retrieve(state["question"]), "attempts": 1}

    def grade(state: RAGState) -> RAGState:
        """LLM-as-grader: is the retrieved context sufficient to answer?"""
        ctx = "\n\n".join(state.get("retrieved", []))
        if deps.llm is None:                                       # >>> STUB <<<
            ok = bool(ctx) and any(w.lower() in ctx.lower()
                                   for w in state["question"].split())
            return {"grade": "relevant" if ok else "weak"}
        prompt = (f"Question: {state['question']}\n\nContext:\n{ctx}\n\n"
                  "Can this context answer the question? Reply exactly 'relevant' or 'weak'.")
        verdict = deps.llm.invoke(prompt).content.strip().lower()
        return {"grade": "relevant" if "relevant" in verdict else "weak"}

    def rewrite(state: RAGState) -> RAGState:
        """Query rewriting to recover from a weak retrieval before retrying."""
        if deps.llm is None:                                       # >>> STUB <<<
            return {"question": state["question"] + " details specifics"}
        better = deps.llm.invoke(
            f"Rewrite this search query to be more specific and retrievable: "
            f"{state['question']}"
        ).content.strip()
        return {"question": better}

    def generate(state: RAGState) -> RAGState:
        ctx = "\n\n".join(state.get("retrieved", []))
        if deps.llm is None:                                       # >>> STUB <<<
            return {"answer": f"[answer grounded in {len(state.get('retrieved', []))} chunks]"}
        ans = deps.llm.invoke(
            f"Answer using ONLY this context. Cite nothing outside it.\n\n"
            f"Context:\n{ctx}\n\nQuestion: {state['question']}"
        ).content
        return {"answer": ans}

    # ---- conditional edge: loop on weak retrieval, with a budget ---- #
    def route_after_grade(state: RAGState) -> str:
        if state.get("grade") == "relevant":
            return "generate"
        if state.get("attempts", 0) >= 3:        # budget guard against infinite loops
            return "generate"
        return "rewrite"

    g = StateGraph(RAGState)
    g.add_node("ingest", ingest)
    g.add_node("index", index)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "index")
    g.add_edge("index", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade,
                            {"generate": "generate", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")            # the cycle
    g.add_edge("generate", END)
    return g.compile()


def _parse_image_path(deps: Deps, path: str):
    from PIL import Image
    return deps.pipeline.parse_image(Image.open(path).convert("RGB"))


# --------------------------------------------------------------------------- #
# Demo (runs on stubs end-to-end)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    deps = Deps()
    app = make_graph(deps)
    out = app.invoke({
        "document_path": "contract.png",   # any image; stub detector treats it as one block
        "question": "What is the termination clause?",
        "attempts": 0,
    })
    print("ANSWER:", out.get("answer"))
    print("ATTEMPTS:", out.get("attempts"))
