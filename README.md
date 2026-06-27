# Grounded Document-QA Agent

Ask questions about any PDF — **born-digital or scanned** — and get answers that **cite the exact
page and spot** they came from. A document-intelligence pipeline (layout-model parsing + cloud OCR)
feeding a **LangGraph** agentic-RAG loop, with the production layer that turns a demo into something
measurable and deployable: an **evaluation harness**, an **abstain path**, **guardrails**, and
**observability**.

**Pipeline:** PDF → parse (Docling layout model · cloud-VLM OCR for scans · text-layer fast path,
**auto-routed** by document type) → parent-child chunks + reranking → agent loop *retrieve → grade →
rewrite↺ → generate* → **cited answer** with click-to-source highlighting. Parsing runs in the
**background with a live progress bar**, so you can query pages as they finish.

> Demoed on financial reports (10-K / annual-report style): multi-column tables, charts, scanned pages.

---

## Evaluation (measured, not claimed)

`evals/run_eval.py` runs a labelled Q&A set through the **full agent** and scores it with an
LLM-as-judge (RAGAS-style: correctness, faithfulness, context recall):

| Metric | Score — sample report, 25 questions |
|---|---|
| Answer correctness | **100%** |
| Faithfulness (grounded — no hallucination) | **100%** |
| Retrieval hit-rate (expected page retrieved) | **100%** |
| Abstain accuracy (out-of-scope refused, not confabulated) | **100%** |

The set deliberately includes prior-year figures, year-on-year comparisons, a **margin
calculation** (the model must divide 2,302 / 6,303), and out-of-scope questions the agent should
**refuse** — and it does, via an explicit abstain path. Full per-question results and honest
caveats (clean synthetic data, self-judging bias): **[`evals/report.md`](evals/report.md)**.
Reproduce: `python -m evals.run_eval`.

---

## Production layer (what separates this from a notebook demo)

- **Evaluation** — `evals/run_eval.py` scores correctness, faithfulness, retrieval hit-rate, and
  abstain accuracy (LLM-as-judge) → [`evals/report.md`](evals/report.md).
- **Abstain** — when retrieval grades weak after retries, the agent says *"I couldn't find that in
  this document"* instead of confabulating (measured 100% on out-of-scope questions).
- **Guardrails** ([`app/guards.py`](app/guards.py)) — input validation blocks prompt-injection in the
  query; a **retrieval rail** flags injection hidden in document text; the generate prompt is hardened
  to treat document content as *data, never instructions*.
- **Observability** ([`app/obs.py`](app/obs.py)) — every answer logs a structured JSONL trace
  (query, retrieval candidates + scores, grades, rewrites, latency, injection flags), served at
  `GET /api/traces`. Drop-in target for Langfuse / LangSmith.
- **Ops** — non-blocking background parsing with progress, `GET /healthz`, a `Dockerfile`, and
  swappable parsers / retrievers / LLM providers behind small interfaces.

---

## Why this shape (the architecture)

```
          ┌──────────────── SRR pipeline (perception) ─────────────────┐
 page ──► │  STRUCTURE          RECOGNITION            RELATION         │ ──► Markdown
          │  "where is it?"     "what is it?"          "how ordered?"   │
          │  tiny detector  ►   small VLM (per block,  ►  reading order │
          │  (XY-cut/YOLO)      parallel)                (XY-cut)       │
          └────────────────────────────────────────────────────────────┘
                                                              │ chunk + index (BM25)
                                                              ▼
                 ┌──────────── LangGraph loop (reasoning) ───────────┐
   question ───► │  retrieve ─► grade ─┬─(relevant)─► generate ─► ✔  │
                 │      ▲              └─(weak)─► rewrite ─┘          │
                 │      └───────────────────────────────────────────┘ (attempts budget)
                 └────────────────────────────────────────────────────┘
```

Each stage hides behind a small `Protocol`, so any component is **swappable without touching
the orchestration** — that modularity is the whole argument for this design over a single
monolithic VLM. The detector is tiny and runs on CPU; recognition is the slow stage but is
**embarrassingly parallel** (one independent call per block); a small VLM on a *cropped region*
solves a far easier problem than one giant VLM decoding a whole dense page.

---

## Quickstart

```powershell
# from the project root
./run.ps1
```

First run creates an isolated `.venv`, installs the lean dependency set (no torch/CUDA),
generates the sample page, and starts the server. Then open **http://127.0.0.1:8000** and
click **“Load sample report.”**

Manual equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe samples\make_sample.py
.\.venv\Scripts\python.exe -m uvicorn app.server:app --port 8000
```

Or with Docker (bundles Docling + models; large image):

```bash
docker build -t docqa-agent .
docker run -p 8000:8000 --env-file .env docqa-agent   # open http://localhost:8000
```

### Two ways to run

| Mode | Setup | Recognition | Q&A answers |
|------|-------|-------------|-------------|
| **Offline** (default) | nothing | bundled sample's ground-truth text; stub for other docs | extractive (top BM25 chunks) |
| **Cloud** (recommended) | put a free key in `.env` | real **VLM vision OCR** per block on *any* doc | synthesized + cited by an LLM |

To enable cloud mode, copy `.env.example` → `.env` and set **one** free key
([Groq](https://console.groq.com/keys) or [OpenRouter](https://openrouter.ai/keys)). Both are
OpenAI-compatible; the recognizer sends each cropped block to a vision model and the agent
loop uses a text model. The local **layout-detection side window needs no key and no GPU.**

### Multi-page born-digital PDFs (annual reports, 10-Ks)

Upload a born-digital PDF (selectable text — exported reports, filings, accounts) and it takes
a **separate, faster path**: PyMuPDF extracts text + bounding boxes for **every page in seconds**
— no OCR, no VLM, no key, no GPU — and the figures are **exact** (not transcribed), which matters
for financial numbers. All pages go into one doc-wide BM25 index; pages render on demand (never
holding hundreds of images in RAM).

- **Page navigator** (◀ / ▶ / jump) over the whole document.
- **Q&A spans all pages**; each answer source carries its `(page, block_id, bbox)`.
- **Click a source → it opens that page and highlights the exact block** it came from.

Try it without your own file: `samples/make_sample_pdf.py` builds a 6-page mock annual report
(`samples/sample_report.pdf`); the server also serves it at `GET /sample.pdf`. Scanned PDFs (no
text layer) are detected and rejected with a clear message — OCR-for-scanned isn't wired in yet.

---

## Hardware note

This was built for a machine with a 2 GB Maxwell GPU — too small for a local VLM. So the
default path is **CPU-only and cloud-first**: the tiny detector runs locally on CPU (its
natural home), and recognition calls a *free* hosted small VLM. No torch, no CUDA needed.

---

## Project map

```
app/
  srr/
    core.py        data model + Protocols + prompt routing + reading order + assembly
    detector.py    HeuristicDetector (XY-cut, default) | DocLayoutYOLODetector (opt)
    recognizer.py  CloudVLM / StubVLM / EasyOCRVLM  +  VLMRecognizer / GroundTruthRecognizer
    cloud.py       Groq / OpenRouter / custom OpenAI-compatible providers
    streaming.py   StreamingSRRPipeline — emits events as it runs (the live source)
    factory.py     build_pipeline() — assembles the right tier from SRR_* env vars
  engine/
    graph.py       LangGraph: BM25 retrieve → grade → rewrite → generate (streamed)
  server.py        FastAPI: upload / sample / page.png / ws_parse / ws_ask
  web/             index.html · styles.css · app.js  (canvas side window + chat)
samples/
  make_sample.py   generates report_page.png + a ground-truth sidecar
files/             the original stubbed reference (srr_pipeline / langgraph / llamaindex)
```

The reusable primitives in `app/srr/core.py` are adapted from the reference
`files/srr_pipeline.py`; the agent nodes in `app/engine/graph.py` from
`files/langgraph_srr_agent.py`.

---

## Swap any component (one env var)

| Stage | Default | Upgrade |
|------|---------|---------|
| Structure | `SRR_DETECTOR=heuristic` (XY-cut, local) | `=doclayout` → DocLayoutYOLO (`pip install doclayout-yolo`) |
| Recognition | `SRR_RECOGNIZER=auto` (cloud if key, else GT/stub) | `=cloud` / `=easyocr` / `=groundtruth` / `=stub` |
| Retrieve | BM25 (built in) | sentence-transformers + FAISS (same `retrieve()` shape) |
| Grade/Generate | heuristic + extractive | cloud LLM (auto when a key is set) |

Run the pipeline headless to see the event stream:
```powershell
.\.venv\Scripts\python.exe -m app.srr.streaming samples\report_page.png --no-delay
.\.venv\Scripts\python.exe -m app.engine.graph     # parses the sample, runs two questions
```

---

## Honest limitations

- **Heuristic typing is rough.** The XY-cut detector segments well but labels blocks only
  approximately (short bold text can read as a heading, etc.). Offline sample mode corrects
  types from ground truth; for arbitrary docs, enable `SRR_DETECTOR=doclayout` for real labels.
- **Reading order is geometric** (recursive XY-cut). Great on Manhattan layouts, brittle on
  wrapped/nested ones — which is exactly why MonkeyOCR *trains* a relation model. The
  `RelationPredictor` interface is ready for a learned drop-in.
- **Tables/formulas** are reproduced properly only by the VLM tier (via per-type prompts);
  plain OCR/ground-truth approximates them.
- **Single page** per upload (PDF → first page). The pipeline is per-page; multi-page is a
  small loop away.
```
