# Evaluation report

RAG answer quality on a labelled Q&A set — **21 in-scope**, **4 out-of-scope** — over the sample annual report. Correctness/faithfulness are scored by LLM-as-judge (Groq · meta-llama/llama-4-scout-17b-16e-instruct); retrieval hit-rate and abstain are objective.

## Summary

| Metric | Score |
|---|---|
| Answer correctness (in-scope) | **100%** |
| Faithfulness — grounded, no hallucination (in-scope) | **100%** |
| Retrieval hit-rate — expected page retrieved (in-scope) | **100%** |
| Abstain accuracy — out-of-scope correctly refused | **100%** |

## Per-question

| Scope | Question | Correct | Faithful | Hit | Abstain |
|---|---|:--:|:--:|:--:|:--:|
| in | What was total revenue in FY26? | ✓ | ✓ | ✓ | — |
| in | What was operating profit in FY26? | ✓ | ✓ | ✓ | — |
| in | What were net assets at the period end? | ✓ | ✓ | ✓ | — |
| in | What total dividend per share was recommended for the year? | ✓ | ✓ | ✓ | — |
| in | What was profit before tax in FY26? | ✓ | ✓ | ✓ | — |
| in | What was the effective tax rate for the year? | ✓ | ✓ | ✓ | — |
| in | How much was capital expenditure? | ✓ | ✓ | ✓ | — |
| in | What was the operating margin? | ✓ | ✓ | ✓ | — |
| in | What were total assets in FY26? | ✓ | ✓ | ✓ | — |
| in | What was cost of sales in FY26? | ✓ | ✓ | ✓ | — |
| in | What was profit after tax in FY26? | ✓ | ✓ | ✓ | — |
| in | What were earnings per share? | ✓ | ✓ | ✓ | — |
| in | By how much did revenue grow year on year? | ✓ | ✓ | ✓ | — |
| in | What were current assets in FY26? | ✓ | ✓ | ✓ | — |
| in | What was gross profit in FY26? | ✓ | ✓ | ✓ | — |
| in | What was revenue in the prior year, FY25? | ✓ | ✓ | ✓ | — |
| in | What were total liabilities in FY26? | ✓ | ✓ | ✓ | — |
| in | What was profit before tax in the prior year, FY25? | ✓ | ✓ | ✓ | — |
| in | By how much did net assets change year on year? | ✓ | ✓ | ✓ | — |
| in | Did operating profit rise or fall versus the prior year, and by roughly how much? | ✓ | ✓ | ✓ | — |
| in | Approximately what was the gross profit margin in FY26? | ✓ | ✓ | ✓ | — |
| oos | Who is the chief executive officer? | — | — | — | ✓ |
| oos | What was the company's share price at year end? | — | — | — | ✓ |
| oos | How many employees does the company have? | — | — | — | ✓ |
| oos | What is the company's carbon emissions reduction target? | — | — | — | ✓ |

## Caveats (read before trusting the numbers)

- The sample is a clean 6-page **synthetic** report. 100% here means the pipeline is sound on tidy input — **not** that it generalises to long, messy, real-world filings.
- LLM-as-judge uses the **same model** that generated the answers (self-consistency bias). A stronger / independent judge model would be more rigorous.
- Faithfulness counts a *derived* figure (e.g. a computed margin) as grounded when its inputs are in context — a lenient but defensible interpretation.
- A production eval needs a larger, **human-labelled** set over real documents, plus per-commit tracking to catch regressions.
