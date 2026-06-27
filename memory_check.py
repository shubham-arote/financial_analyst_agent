"""Verify conversation memory: a follow-up that only resolves via prior-turn context."""
from evals.run_eval import build_index
from app.engine.graph import AgentEngine, _CHECKPOINTER
from app.engine.hybrid import build_retriever

print("checkpointer:", type(_CHECKPOINTER).__name__)
eng = AgentEngine(build_retriever(build_index()))

print("\n=== Thread t1 (conversation with memory) ===")
r1 = eng.run("What was operating profit in FY26?", thread_id="t1")
print("Q1: What was operating profit in FY26?")
print("A1:", r1["answer"][:150])
r2 = eng.run("What about the prior year?", thread_id="t1")     # <- follow-up, no subject named
print("Q2: What about the prior year?   (follow-up — only resolvable from history)")
print("A2:", r2["answer"][:150])
print("history turns accumulated:", len(r2.get("history", [])))

print("\n=== Thread t2 (fresh — same follow-up COLD, no history) ===")
r3 = eng.run("What about the prior year?", thread_id="t2")
print("A :", r3["answer"][:150])
print("\n(Expect: A1~1,052  A2~985 [FY25 op profit, resolved via memory]  t2 cold -> abstains)")
