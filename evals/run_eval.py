r"""
Evaluation harness — RAGAS-style metrics over a labelled Q&A set on the sample report.

Measures answer quality with an LLM-as-judge (the configured cloud provider):
  - answer_correctness : does the answer state the reference fact?           (in-scope)
  - faithfulness       : is every claim grounded in the retrieved context?    (in-scope)
  - retrieval_hit_rate : did retrieval surface the expected page?  (in-scope, objective)
  - abstain_accuracy   : did it correctly refuse out-of-scope questions?      (out-of-scope)

Run:  .venv\Scripts\python.exe -m evals.run_eval
Prints a summary and writes evals/report.md (portfolio evidence).
"""

from __future__ import annotations

import time
from pathlib import Path

import fitz

from app.engine.graph import AgentEngine, DocIndex
from app.engine.hybrid import build_retriever
from app.srr import cloud, pdf_textlayer
from app.srr.core import ColumnAwareReadingOrder

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample_report.pdf"
REPORT = Path(__file__).with_name("report.md")

# Labelled set grounded in the synthetic ACME PLC sample. a=None => out-of-scope (abstain).
GOLDEN = [
    {"q": "What was total revenue in FY26?", "a": "6,303 million pounds", "pages": [3, 4]},
    {"q": "What was operating profit in FY26?", "a": "1,052 million pounds", "pages": [4]},
    {"q": "What were net assets at the period end?", "a": "1,430 million pounds", "pages": [5]},
    {"q": "What total dividend per share was recommended for the year?", "a": "140 pence", "pages": [2]},
    {"q": "What was profit before tax in FY26?", "a": "1,011 million pounds", "pages": [3, 4]},
    {"q": "What was the effective tax rate for the year?", "a": "23.8%", "pages": [6]},
    {"q": "How much was capital expenditure?", "a": "168 million pounds", "pages": [6]},
    {"q": "What was the operating margin?", "a": "16.7%", "pages": [3]},
    {"q": "What were total assets in FY26?", "a": "4,520 million pounds", "pages": [5]},
    {"q": "What was cost of sales in FY26?", "a": "4,001 million pounds", "pages": [4]},
    {"q": "What was profit after tax in FY26?", "a": "770 million pounds", "pages": [4]},
    {"q": "What were earnings per share?", "a": "712 pence", "pages": [3]},
    {"q": "By how much did revenue grow year on year?", "a": "5.9%", "pages": [3]},
    {"q": "What were current assets in FY26?", "a": "1,980 million pounds", "pages": [5]},
    {"q": "What was gross profit in FY26?", "a": "2,302 million pounds", "pages": [4]},
    # harder: prior-year, comparison, and calculation (likely to surface weaknesses)
    {"q": "What was revenue in the prior year, FY25?", "a": "5,952 million pounds", "pages": [4]},
    {"q": "What were total liabilities in FY26?", "a": "3,090 million pounds", "pages": [5]},
    {"q": "What was profit before tax in the prior year, FY25?", "a": "933 million pounds", "pages": [4]},
    {"q": "By how much did net assets change year on year?", "a": "increased by 150 million pounds (1,430 vs 1,280)", "pages": [5]},
    {"q": "Did operating profit rise or fall versus the prior year, and by roughly how much?", "a": "rose by about 67 million pounds (1,052 vs 985)", "pages": [4]},
    {"q": "Approximately what was the gross profit margin in FY26?", "a": "about 36.5% (2,302 / 6,303)", "pages": [4]},
    {"q": "Who is the chief executive officer?", "a": None, "pages": []},
    {"q": "What was the company's share price at year end?", "a": None, "pages": []},
    {"q": "How many employees does the company have?", "a": None, "pages": []},
    {"q": "What is the company's carbon emissions reduction target?", "a": None, "pages": []},
]

ABSTAIN_HINTS = ("couldn't find", "could not find", "not in this document", "no information",
                 "not present", "isn't in", "not mentioned", "not contain", "not available",
                 "does not contain", "unable to find", "doesn't contain")


def build_index() -> DocIndex:
    doc = fitz.open(str(SAMPLE))
    rel = ColumnAwareReadingOrder()
    pages = [pdf_textlayer.extract_page(doc[n], n + 1, rel) for n in range(len(doc))]
    sizes = [(doc[n].rect.width, doc[n].rect.height) for n in range(len(doc))]
    pdf_textlayer.mark_repeated_furniture(pages, sizes)
    return DocIndex.from_blocks([b for pb in pages for b in pb])


def _yes(prompt: str) -> float:
    r = cloud.chat_text(prompt + "\n\nReply with ONLY 'yes' or 'no'.", max_tokens=3).strip().lower()
    return 1.0 if r.startswith("y") else 0.0


def judge_correct(q, ref, ans):
    return _yes(f"Question: {q}\nReference answer: {ref}\nCandidate answer: {ans}\n\n"
                "Does the candidate answer state the same key fact/number as the reference answer?")


def judge_faithful(ctx, ans):
    return _yes(f"Context:\n{ctx}\n\nAnswer: {ans}\n\n"
                "Is every factual claim in the answer supported by the context above?")


def is_abstain(ans: str) -> bool:
    a = (ans or "").lower()
    return any(h in a for h in ABSTAIN_HINTS)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def write_report(rows, m, n_in, n_oos):
    pct = lambda v: f"{v * 100:.0f}%"
    mark = lambda v: "—" if v is None else ("✓" if v == 1.0 else "✗")
    out = [
        "# Evaluation report", "",
        f"RAG answer quality on a labelled Q&A set — **{n_in} in-scope**, **{n_oos} out-of-scope** — "
        "over the sample annual report. Correctness/faithfulness are scored by LLM-as-judge "
        f"({cloud.provider_label()}); retrieval hit-rate and abstain are objective.", "",
        "## Summary", "",
        "| Metric | Score |", "|---|---|",
        f"| Answer correctness (in-scope) | **{pct(m['answer_correctness'])}** |",
        f"| Faithfulness — grounded, no hallucination (in-scope) | **{pct(m['faithfulness'])}** |",
        f"| Retrieval hit-rate — expected page retrieved (in-scope) | **{pct(m['retrieval_hit_rate'])}** |",
        f"| Abstain accuracy — out-of-scope correctly refused | **{pct(m['abstain_accuracy'])}** |",
        "", "## Per-question", "",
        "| Scope | Question | Correct | Faithful | Hit | Abstain |",
        "|---|---|:--:|:--:|:--:|:--:|",
    ]
    for r in rows:
        scope = "in" if r["in_scope"] else "oos"
        out.append(f"| {scope} | {r['q']} | {mark(r['correct'])} | {mark(r['faithful'])} "
                   f"| {mark(r['hit'])} | {mark(r['abstain'])} |")
    out += [
        "", "## Caveats (read before trusting the numbers)", "",
        "- The sample is a clean 6-page **synthetic** report. 100% here means the pipeline is sound "
        "on tidy input — **not** that it generalises to long, messy, real-world filings.",
        "- LLM-as-judge uses the **same model** that generated the answers (self-consistency bias). "
        "A stronger / independent judge model would be more rigorous.",
        "- Faithfulness counts a *derived* figure (e.g. a computed margin) as grounded when its inputs "
        "are in context — a lenient but defensible interpretation.",
        "- A production eval needs a larger, **human-labelled** set over real documents, plus "
        "per-commit tracking to catch regressions.",
    ]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    if not cloud.has_cloud():
        print("This eval needs a cloud key (LLM-as-judge + answers). Set GROQ_API_KEY in .env.")
        return
    print(f"judge/answers: {cloud.provider_label()}\nbuilding index from {SAMPLE.name}...")
    eng = AgentEngine(build_retriever(build_index()))      # hybrid when COHERE_API_KEY set, else BM25
    rows = []
    for i, item in enumerate(GOLDEN, 1):
        res = eng.run(item["q"])
        ans = res.get("answer", "")
        retrieved = res.get("retrieved", [])
        src_pages = {c.get("page") for c in retrieved}
        in_scope = item["a"] is not None
        row = {"q": item["q"], "in_scope": in_scope, "correct": None,
               "faithful": None, "hit": None, "abstain": None}
        if in_scope:
            row["correct"] = judge_correct(item["q"], item["a"], ans)
            row["faithful"] = judge_faithful(eng._context(retrieved), ans)
            row["hit"] = 1.0 if (set(item["pages"]) & src_pages) else 0.0
        else:
            row["abstain"] = 1.0 if is_abstain(ans) else 0.0
        rows.append(row)
        print(f"  [{i:>2}/{len(GOLDEN)}] {'OOS' if not in_scope else 'IN '} {item['q'][:40]:40} "
              f"corr={row['correct']} faith={row['faithful']} hit={row['hit']} abst={row['abstain']}")
        time.sleep(1.2)                                      # rate-limit friendly

    inscope = [r for r in rows if r["in_scope"]]
    oos = [r for r in rows if not r["in_scope"]]
    m = {
        "answer_correctness": _mean([r["correct"] for r in inscope]),
        "faithfulness": _mean([r["faithful"] for r in inscope]),
        "retrieval_hit_rate": _mean([r["hit"] for r in inscope]),
        "abstain_accuracy": _mean([r["abstain"] for r in oos]),
    }
    write_report(rows, m, len(inscope), len(oos))
    print("\n=== RESULTS ===")
    for k, v in m.items():
        print(f"  {k:20} {v * 100:5.0f}%")
    print(f"\nreport written -> {REPORT}")


if __name__ == "__main__":
    main()
