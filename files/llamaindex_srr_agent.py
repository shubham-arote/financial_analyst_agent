"""
llamaindex_srr_agent.py
=======================
Wires the SRR pipeline (srr_pipeline.py) into LlamaIndex's AgentWorkflow pattern.

Two integration points:
  1. SRRReader      : a custom reader/node-parser. SRR -> layout-faithful Markdown
                      -> LlamaIndex Document(s). The recovered headings/tables become
                      real structure the splitter and retriever can exploit.
  2. FunctionAgent  : an orchestrator that holds the parsed corpus as a query tool.
                      This mirrors the "Agentic Document Workflows" pattern: document
                      processing + retrieval + structured output under one agent that
                      decides when to look things up.

For multi-document setups, give the orchestrator several QueryEngineTools (one per
doc or per domain) and it routes — i.e. the Meta-Agent / Document-Agent pattern.

Deps:  pip install llama-index llama-index-llms-openai llama-index-embeddings-openai
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.tools import QueryEngineTool

from srr_pipeline import SRRPipeline, build_default_pipeline


# --------------------------------------------------------------------------- #
# 1. SRR as a LlamaIndex reader
# --------------------------------------------------------------------------- #
class SRRReader:
    """
    Turns a file path into LlamaIndex Documents via the SRR pipeline.
    One Document per page keeps page provenance in metadata (useful for citations).
    """

    def __init__(self, pipeline: SRRPipeline | None = None):
        self.pipeline = pipeline or build_default_pipeline()

    def load_data(self, file_path: str) -> list[Document]:
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            result = self.pipeline.parse_pdf(str(path))
            # split the assembled markdown back per page (parse_pdf joins on a rule)
            pages = result.markdown.split("\n\n---\n\n")
            return [
                Document(text=md, metadata={"source": path.name, "page": i})
                for i, md in enumerate(pages) if md.strip()
            ]
        from PIL import Image
        result = self.pipeline.parse_image(Image.open(file_path).convert("RGB"))
        return [Document(text=result.markdown, metadata={"source": path.name, "page": 0})]


# --------------------------------------------------------------------------- #
# 2. Build an index + expose it as a tool
# --------------------------------------------------------------------------- #
def build_query_tool(file_path: str, name: str, description: str,
                     pipeline: SRRPipeline | None = None) -> QueryEngineTool:
    docs = SRRReader(pipeline).load_data(file_path)

    # MarkdownNodeParser respects the heading hierarchy the SRR Relation stage
    # reconstructed — so chunks align with logical document sections.
    index = VectorStoreIndex.from_documents(
        docs, transformations=[MarkdownNodeParser()]
    )
    query_engine = index.as_query_engine(similarity_top_k=4)
    return QueryEngineTool.from_defaults(
        query_engine=query_engine, name=name, description=description
    )


# --------------------------------------------------------------------------- #
# 3. Orchestrating agent over one or many document tools
# --------------------------------------------------------------------------- #
def build_agent(tools: list[QueryEngineTool], llm=None):
    """
    A single FunctionAgent that decides which document tool to query. Swap in a
    multi-agent AgentWorkflow (one agent per domain) for larger corpora.
    """
    from llama_index.core.agent.workflow import FunctionAgent

    if llm is not None:
        Settings.llm = llm
    # else: configure Settings.llm / Settings.embed_model at startup, e.g.
    #   from llama_index.llms.openai import OpenAI
    #   Settings.llm = OpenAI(model="gpt-4o-mini")

    return FunctionAgent(
        tools=tools,
        system_prompt=(
            "You are a document-intelligence agent. Use the provided document "
            "tools to ground every answer. If a question spans multiple documents, "
            "query each relevant tool and synthesize. Never answer from prior "
            "knowledge when a tool can supply the fact."
        ),
    )


# --------------------------------------------------------------------------- #
# Demo (structure only — needs real LLM/embeddings + served VLM to run for real)
# --------------------------------------------------------------------------- #
async def _demo():
    pipeline = build_default_pipeline()
    tools = [
        build_query_tool(
            "msa_2025.pdf", name="master_services_agreement",
            description="The 2025 Master Services Agreement: scope, fees, termination, liability.",
            pipeline=pipeline,
        ),
        build_query_tool(
            "sow_alpha.pdf", name="statement_of_work_alpha",
            description="Statement of Work for Project Alpha: deliverables, milestones, acceptance.",
            pipeline=pipeline,
        ),
    ]
    agent = build_agent(tools)
    resp = await agent.run(
        "Do the SOW milestones conflict with the MSA termination notice period?"
    )
    print(resp)


if __name__ == "__main__":
    import asyncio
    # Requires Settings.llm/embed_model + a served VLM; otherwise this is a layout
    # reference. The SRR pipeline itself is exercisable standalone via srr_pipeline.py.
    asyncio.run(_demo())
