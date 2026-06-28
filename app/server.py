r"""
server.py — thin FastAPI surface for the document-QA / analyst agent.

Routes only: receive uploads + questions, render pages, stream events. All orchestration is
delegated to `app.services` (documents = lifecycle, chat = ask) over the core packages
(srr / retrieval / agent / storage). Two ingestion paths:

  * single image / sample  -> live SRR pipeline, streamed over /ws/parse
  * PDF                     -> background parse (auto-routed), pages served on demand

Endpoints
  GET  /                          the split-view UI
  GET  /api/status               provider/recognition mode for the header badge
  POST /upload                   image -> single page; .pdf -> multi-page doc (parses in bg)
  POST /load-sample              the bundled synthetic page (offline ground-truth demo)
  GET  /sample.pdf               the bundled multi-page sample PDF
  GET  /api/traces               recent agent traces (observability)
  GET  /healthz                  liveness + doc counts + cloud flag
  GET  /doc/{id}/page.png        single-image page
  GET  /doc/{id}/page/{n}.png    PDF page n, rendered on demand
  GET  /doc/{id}/status          PDF parse progress
  GET  /doc/{id}/page/{n}        PDF page n blocks + markdown (JSON)
  WS   /ws/parse/{id}            single-image: streams SRR stage events; builds the index
  WS   /ws/ask/{id}              streams the agent's node events + the grounded answer

Run:  .venv\Scripts\python.exe -m uvicorn app.server:app
"""

from __future__ import annotations

import asyncio
import io
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import guards, obs
from .srr import cloud
from .srr.core import block_to_dict
from .srr.factory import build_pipeline
from .srr.parsers import textlayer
from .services import chat, documents

WEB_DIR = Path(__file__).parent / "web"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
SAMPLE_PNG = SAMPLES_DIR / "report_page.png"
SAMPLE_JSON = SAMPLE_PNG.with_suffix(".regions.json")
SAMPLE_PDF = SAMPLES_DIR / "sample_report.pdf"

app = FastAPI(title="Financial Analyst Agent")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp


@app.on_event("startup")
def _startup():
    documents.load_persisted_docs()       # rebuild DOCS from the store (survive restart)


async def _bridge(gen_factory):
    """Run a blocking generator in a thread, yield its events into the event loop."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    DONE = object()

    def worker():
        try:
            for ev in gen_factory():
                loop.call_soon_threadsafe(queue.put_nowait, ev)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "error": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, DONE)

    loop.run_in_executor(None, worker)
    while True:
        ev = await queue.get()
        if ev is DONE:
            return
        yield ev


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status():
    return {"cloud": cloud.has_cloud(), "provider": cloud.provider_label(),
            "sample_available": SAMPLE_PNG.exists(), "sample_pdf": SAMPLE_PDF.exists()}


@app.post("/upload")
async def upload(file: UploadFile):
    data = await file.read()
    if (file.filename or "").lower().endswith(".pdf"):
        try:
            return documents.start_pdf(data)                    # non-blocking; parses in background
        except Exception as e:
            return JSONResponse({"error": f"could not open PDF: {e}"}, status_code=400)
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"could not read file: {e}"}, status_code=400)
    return documents.store_image(image)


@app.post("/load-sample")
def load_sample():
    if not SAMPLE_PNG.exists():
        return JSONResponse({"error": "sample not generated; run samples/make_sample.py"},
                            status_code=404)
    import json
    gt = json.loads(SAMPLE_JSON.read_text(encoding="utf-8")) if SAMPLE_JSON.exists() else None
    return documents.store_image(Image.open(SAMPLE_PNG).convert("RGB"), ground_truth=gt)


@app.get("/sample.pdf")
def sample_pdf():
    if SAMPLE_PDF.exists():
        return FileResponse(SAMPLE_PDF, media_type="application/pdf", filename="sample_report.pdf")
    return Response(status_code=404)


@app.get("/api/traces")
def api_traces(n: int = 20):
    """Recent agent traces (query, retrieval candidates+scores, grades, latency, injection flags)."""
    return {"traces": obs.recent(n)}


@app.get("/healthz")
def healthz():
    return {"status": "ok", "docs_loaded": len(documents.DOCS),
            "docs_persisted": documents.STORE.count(), "cloud": cloud.has_cloud()}


@app.get("/doc/{doc_id}/page.png")
def page_png(doc_id: str):
    doc = documents.get(doc_id)
    if not doc or doc.get("kind") != "image":
        return Response(status_code=404)
    buf = io.BytesIO()
    doc["image"].save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/doc/{doc_id}/page/{n}.png")
def pdf_page_png(doc_id: str, n: int):
    doc = documents.get(doc_id)
    if not doc or doc.get("kind") != "pdf" or n < 1 or n > doc["page_count"]:
        return Response(status_code=404)
    pix = doc["fitz"][n - 1].get_pixmap(dpi=130)
    return Response(content=pix.tobytes("png"), media_type="image/png")


@app.get("/doc/{doc_id}/status")
def pdf_status(doc_id: str):
    doc = documents.get(doc_id)
    if not doc or doc.get("kind") != "pdf":
        return JSONResponse({"error": "unknown doc"}, status_code=404)
    return {"status": doc.get("status", "ready"),
            "parsed_pages": doc.get("parsed_pages", doc["page_count"]),
            "page_count": doc["page_count"], "chunks": len(doc["index"].chunks),
            "error": doc.get("error")}


@app.get("/doc/{doc_id}/page/{n}")
def pdf_page_data(doc_id: str, n: int):
    doc = documents.get(doc_id)
    if not doc or doc.get("kind") != "pdf" or n < 1 or n > doc["page_count"]:
        return JSONResponse({"error": "unknown page"}, status_code=404)
    blocks = doc["pages_blocks"][n - 1]
    w, h = doc["sizes"][n - 1]
    if not blocks and doc.get("status") == "parsing" and n > doc.get("parsed_pages", 0):
        return {"page": n, "page_w": int(w), "page_h": int(h), "status": "pending"}
    return {"page": n, "page_w": int(w), "page_h": int(h), "status": "ready",
            "blocks": [block_to_dict(b) for b in blocks],
            "markdown": textlayer.page_markdown(blocks)}


# --------------------------------------------------------------------------- #
# WebSockets
# --------------------------------------------------------------------------- #
@app.websocket("/ws/parse/{doc_id}")
async def ws_parse(ws: WebSocket, doc_id: str):
    await ws.accept()
    doc = documents.get(doc_id)
    if not doc or doc.get("kind") != "image":
        await ws.send_json({"type": "error", "error": "not a single-image doc"})
        await ws.close()
        return

    pipeline = build_pipeline(ground_truth=doc["ground_truth"])
    captured: dict = {}

    def gen():
        for ev in pipeline.parse_page_streaming(doc["image"]):
            if ev["type"] == "result":
                captured["markdown"] = ev["markdown"]
            yield ev

    try:
        async for ev in _bridge(gen):
            await ws.send_json(ev)
        blocks = getattr(pipeline, "last_ordered_blocks", [])
        chunks = documents.finalize_image(doc_id, doc, blocks, captured.get("markdown", ""))
        await ws.send_json({"type": "indexed", "chunks": chunks})
    except WebSocketDisconnect:
        return
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass


@app.websocket("/ws/ask/{doc_id}")
async def ws_ask(ws: WebSocket, doc_id: str):
    await ws.accept()
    thread_id = uuid.uuid4().hex                      # one conversation per ws connection
    try:
        while True:
            msg = await ws.receive_json()
            question = (msg.get("question") or "").strip()
            doc = documents.get(doc_id)
            if not question:
                continue
            ok, reason = guards.check_question(question)        # input guardrail
            if not ok:
                await ws.send_json({"type": "error", "error": reason})
                continue
            if not doc or not doc.get("index") or not doc["index"].chunks:
                await ws.send_json({"type": "error", "error": "still parsing — ask again in a moment"})
                continue
            engine = chat.get_engine(doc, doc_id)               # retriever cached per finalized index

            def gen():
                yield from engine.run_streaming(question, thread_id=thread_id)

            async for ev in _bridge(gen):
                await ws.send_json(ev)
    except WebSocketDisconnect:
        return
