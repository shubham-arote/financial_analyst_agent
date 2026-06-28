# CLAUDE.md — project conventions & guardrails

> Read this before changing anything. It keeps work consistent and prevents drift.
> A modular restructure is in progress on branch `refactor/modular-structure`
> (plan: `~/.claude/plans/breezy-meandering-stroustrup.md`). The architecture below
> is the **target**; update this file as each phase lands.

## What this is

A **grounded document-QA agent**. Upload a PDF (born-digital or scanned) → ask questions →
get answers that **cite the exact page + block**, with click-to-source highlighting in the UI.
It is NOT (yet) a financial *analyst* agent — no calculator/verifier/multi-agent layer.

**Data flow:**
```
PDF ─▶ parse (auto-routed)      srr/  →  born-digital: PyMuPDF text-layer
       │                                  scanned:      cloud-VLM OCR (Groq)
       │                                  (optional:    Docling layout model)
       ▼
   parent-child chunks + index  retrieval/index.py (DocIndex: blocks→children, sections→parents, BM25)
       ▼
   hybrid retrieve              retrieval/  BM25 + Cohere dense → query-type RRF → Cohere rerank
       │                                    → deterministic table/number lookup (exact, floated on top)
       ▼
   LangGraph agent              agent/graph.py  contextualize → retrieve → grade → rewrite↺ → generate
       │                                        (+ abstain when context is weak)
       ▼
   cited answer + highlight     server.py / web/  WS stream of node events + Evidence
```

## Architecture map (target layout)

```
app/
  config.py        single Settings object — the ONLY place env vars are read
  server.py        thin FastAPI routes; delegates to services/
  services/        orchestration: documents.py (doc lifecycle), chat.py (ask)
  srr/             parsing: core (data model), detector, recognizer, streaming,
                   factory, cloud (LLM client), parsers/{textlayer,docling,cloudvlm}
  retrieval/       base.py (Retriever Protocol + Evidence), index.py (DocIndex),
                   hybrid.py, qdrant.py, cohere.py
  agent/           graph.py (AgentEngine + LangGraph)
  storage/         docstore.py (DocStore ABC + Sqlite + GCS)
  guards.py obs.py guardrails + observability (JSONL traces)
  web/             index.html, app.js, styles.css
tests/             pytest suite — fast offline gate (test_smoke.py) + units
evals/             RAGAS-style eval harness + retrieval A/B
docs/              deploy.md, retrieval.md
```

### The two seams (depend on these, not concretes)
- **`retrieval.base.Retriever`** — Protocol with `.retrieve(query, k) -> list[Evidence]`.
  `Evidence` is a TypedDict carrying `page, bbox, block_id, section_id, type, text, content,
  heading, parent_text, section_heading, score, exact`. The agent depends on this, never on a
  concrete retriever. `make_retriever()` selects the backend (in-process hybrid vs Qdrant) by config.
- **`storage.docstore.DocStore`** — ABC for persisted docs (meta + blob + per-page blocks).
  `SqliteDocStore` (local default) ↔ `GcsDocStore` (stateless cloud), selected by config.

## Hard constraints (non-negotiable)

- **No local ML/VLM models.** Dev machine is a 2 GB GPU (GeForce 920M). Everything heavy is
  **cloud-first**: Groq / OpenRouter (chat + vision OCR) and Cohere (embed-v4 + rerank-v3.5),
  all free tier. Docling/torch are *optional* installs, never required, never in the lean image.
- **Always run in the venv** (`.venv`). Never install into system Python.
- **`.env` is never committed** (gitignored) and never baked into the Docker image
  (`.dockerignore`). Secrets reach the cloud only via `--set-secrets` / host secrets.
- **Keep the lean Docker image lean** — no torch / Docling / easyocr in `Dockerfile`.

## Conventions

- **Config:** read env ONLY through `app.config.settings`. Do not call `os.environ` / `os.getenv`
  anywhere else.
- **Public API:** each package exposes its surface via its `__init__.py`. Import from the package,
  not deep modules, across layer boundaries.
- **Key-optional degradation:** any cloud dependency must degrade gracefully when its key is
  absent (no Cohere → BM25; no chat key → extractive answers; no Qdrant → in-process). Never
  hard-crash on a missing key.
- **Verify, don't assume:** every change is gated by `pytest tests/` (see below).
- **Windows dev:** no Python hot-reload — restart the server after code changes. Static assets
  are no-cache; bump the `?v=` query in `index.html` when changing `app.js`/`styles.css`.

## Run & verify

```powershell
# run (loads .env, serves http://127.0.0.1:8000)
./run.ps1

# fast offline gate — MUST be green after every change/phase
.venv/Scripts/python.exe -m pytest tests/ -q

# retrieval A/B + RAGAS-style eval (uses keys if present)
.venv/Scripts/python.exe -m evals.compare_retrieval
.venv/Scripts/python.exe -m evals.run_eval
```

## Don'ts

- Don't add local ML models or make any key *required*.
- Don't read env vars outside `app/config.py`.
- Don't bake secrets into code or the image.
- Don't bloat the lean image (no torch/Docling).
- Don't reach across layers into deep modules — go through package `__init__` and the seams.
