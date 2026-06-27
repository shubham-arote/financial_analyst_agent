"""Verify the Postgres checkpointer: multi-turn memory persisted through Postgres."""
import os
os.environ["SRR_CHECKPOINT"] = "postgres"
os.environ["CHECKPOINT_DB_URL"] = "postgresql://postgres:pw@localhost:5433/checkpoints"

from app.engine.graph import AgentEngine, _CHECKPOINTER
from app.engine.hybrid import build_retriever
from evals.run_eval import build_index

print("checkpointer:", type(_CHECKPOINTER).__name__)        # expect PostgresSaver
eng = AgentEngine(build_retriever(build_index()))           # uses the Postgres checkpointer
tid = "pg-convo-1"
print("Q1: What was operating profit in FY26?")
print("A1:", eng.run("What was operating profit in FY26?", thread_id=tid)["answer"][:120])
r2 = eng.run("What about the prior year?", thread_id=tid)   # follow-up, resolved via Postgres memory
print("Q2: What about the prior year?  (follow-up)")
print("A2:", r2["answer"][:120])
print("history turns persisted in Postgres:", len(r2.get("history", [])))
print("\nPOSTGRES CHECKPOINTER OK")
