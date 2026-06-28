r"""
server.py — FastAPI app tying the live SRR pipeline + LangGraph engine to a browser UI.

Two ingestion paths:
  * single image / sample  -> image SRR pipeline, streamed live over /ws/parse
  * born-digital PDF        -> PyMuPDF text-layer extraction of ALL pages at upload,
                               one doc-wide block-aware index, pages rendered on demand

Endpoints
  GET  /                          the split-view UI
  GET  /api/status               provider/recognition mode for the header badge
  POST /upload                   image -> single page; .pdf -> multi-page text-layer doc
  POST /load-sample              the bundled synthetic page (offline ground-truth demo)
  GET  /sample.pdf               the bundled multi-page sample PDF (for testing upload)
  GET  /doc/{id}/page.png        single-image page
  GET  /doc/{id}/page/{n}.png    PDF page n, rendered on demand
  GET  /doc/{id}/page/{n}        PDF page n blocks + markdown (JSON)
  WS   /ws/parse/{id}            single-image: streams SRR stage events; builds index
  WS   /ws/ask/{id}              streams LangGraph node events + the grounded answer

Run:  .venv\Scripts\python.exe -m uvicorn app.server:app
"""

from __future__ import annotations

import asyncio
import io
import threading
import uuid
from pathlib import Path

import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from . import guards, obs, store
from .config import settings
from .agent.graph import AgentEngine
from .retrieval import DocIndex, make_retriever
from .srr import cloud, pdf_textlayer
from .srr.core import ColumnAwareReadingOrder, block_from_dict, block_to_dict
from .srr.factory import build_pipeline

WEB_DIR = Path(__file__).parent / "web"
SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"
SAMPLE_PNG = SAMPLES_DIR / "report_page.png"
SAMPLE_JSON = SAMPLE_PNG.with_suffix(".regions.json")
SAMPLE_PDF = SAMPLES_DIR / "sample_report.pdf"
MAX_W = 1600  # downscale huge image uploads (canvas perf + VLM cost)

app = FastAPI(title="Live SRR Document Engine")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static") or request.url.path == "/":
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return resp

DOCS: dict[str, dict] = {}   # doc_id -> live per-doc state (reconstructable from STORE)
STORE = store.get_store()    # durable persistence (survives restart; SQLite -> GCS/Firestore on GCP)


# --------------------------------------------------------------------------- #
# Store helpers
# --------------------------------------------------------------------------- #
def _store_image(image: Image.Image, ground_truth=None) -> dict:
    if image.width > MAX_W:
        image = image.resize((MAX_W, round(image.height * MAX_W / image.width)))
    doc_id = uuid.uuid4().hex[:12]
    DOCS[doc_id] = {"kind": "image", "image": image, "page_w": image.width,
                    "page_h": image.height, "ground_truth": ground_truth,
                    "markdown": "", "index": None}
    buf = io.BytesIO(); image.save(buf, format="PNG")
    STORE.save_doc(doc_id, "image", {"page_w": image.width, "page_h": image.height,
                   "ground_truth": ground_truth, "markdown": "", "status": "ready",
                   "parsed_pages": 1, "page_count": 1}, buf.getvalue())
    return {"doc_id": doc_id, "multipage": False, "page_w": image.width, "page_h": image.height}


def _start_pdf(data: bytes) -> dict:
    """Create the doc entry and kick off background parsing; return immediately so the UI
    can show pages + a progress bar and answer over pages as they finish (Docling is slow)."""
    fitz_doc = fitz.open(stream=data, filetype="pdf")
    n = len(fitz_doc)
    sizes = [(int(fitz_doc[i].rect.width), int(fitz_doc[i].rect.height)) for i in range(n)]
    doc_id = uuid.uuid4().hex[:12]
    DOCS[doc_id] = {"kind": "pdf", "fitz": fitz_doc, "page_count": n, "sizes": sizes,
                    "pages_blocks": [[] for _ in range(n)], "index": DocIndex.from_blocks([]),
                    "status": "parsing", "parsed_pages": 0, "error": None, "lock": threading.Lock()}
    STORE.save_doc(doc_id, "pdf", {"page_count": n, "sizes": sizes, "status": "parsing",
                   "parsed_pages": 0, "error": None}, data)
    threading.Thread(target=_parse_pdf_bg, args=(doc_id, data), daemon=True).start()
    w, h = sizes[0] if sizes else (595, 842)
    return {"doc_id": doc_id, "multipage": True, "page_count": n,
            "page_w": w, "page_h": h, "status": "parsing"}


def _rebuild_index(doc: dict) -> None:
    idx = DocIndex.from_blocks([b for pb in doc["pages_blocks"] for b in pb])
    with doc["lock"]:
        doc["index"] = idx


def _save_pages(doc_id: str, doc: dict, lo: int, hi: int) -> None:
    """Persist pages [lo, hi) (0-based) + the current parse status to the DocStore."""
    for i in range(lo, hi):
        STORE.save_page(doc_id, i + 1, [block_to_dict(b) for b in doc["pages_blocks"][i]])
    STORE.update_status(doc_id, doc.get("status", "parsing"), doc.get("parsed_pages", 0), doc.get("error"))


def _fail(doc_id: str, doc: dict, msg: str) -> None:
    doc["status"], doc["error"] = "error", msg
    STORE.update_status(doc_id, "error", doc.get("parsed_pages", 0), msg)


def _docling_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("docling") is not None


def _parse_pdf_bg(doc_id: str, data: bytes) -> None:
    """Background worker: fill pages_blocks incrementally, rebuild the index, and persist as it goes."""
    doc = DOCS.get(doc_id)
    if not doc:
        return
    parser = settings.pdf_parser
    relation = ColumnAwareReadingOrder()
    n = doc["page_count"]
    try:
        if parser == "auto":                              # best available per document
            from .srr import pdf_cloudvlm
            scanned = not pdf_textlayer.has_text_layer(doc["fitz"])
            if scanned:
                parser = "cloudvlm" if pdf_cloudvlm.available() else "docling"
            else:                                         # born-digital: Docling if installed, else fast text layer
                parser = "docling" if _docling_available() else "textlayer"
        if parser == "textlayer":
            fitz_doc = doc["fitz"]
            if not pdf_textlayer.has_text_layer(fitz_doc):
                return _fail(doc_id, doc, "This PDF has no text layer (looks scanned).")
            pb = [pdf_textlayer.extract_page(fitz_doc[i], i + 1, relation) for i in range(n)]
            pdf_textlayer.mark_repeated_furniture(pb, doc["sizes"])
            doc["pages_blocks"] = pb
            doc["parsed_pages"] = n
            _rebuild_index(doc)
            _save_pages(doc_id, doc, 0, n)
        elif parser == "cloudvlm":
            from .srr import pdf_cloudvlm
            if not pdf_cloudvlm.available():
                return _fail(doc_id, doc, "Cloud VLM OCR needs GROQ_API_KEY or OPENROUTER_API_KEY in .env.")
            chunk = settings.vlm_chunk
            for a in range(0, n, chunk):
                b = min(a + chunk, n)
                for gp0, blocks in pdf_cloudvlm.parse_pages(data, a, b).items():
                    if 0 <= gp0 < n:
                        doc["pages_blocks"][gp0] = blocks
                doc["parsed_pages"] = b
                _rebuild_index(doc)
                _save_pages(doc_id, doc, a, b)
        else:
            from .srr import pdf_docling
            env_ocr = settings.docling_ocr
            do_ocr = (not pdf_textlayer.has_text_layer(doc["fitz"])) if env_ocr is None else (env_ocr != "0")
            chunk = settings.docling_chunk
            for a in range(0, n, chunk):
                b = min(a + chunk, n)
                for gp0, blocks in pdf_docling.parse_pages(data, a, b, do_ocr=do_ocr).items():
                    if 0 <= gp0 < n:
                        doc["pages_blocks"][gp0] = blocks
                doc["parsed_pages"] = b
                _rebuild_index(doc)                        # query-as-you-go
                _save_pages(doc_id, doc, a, b)
        doc["status"] = "ready"
        STORE.update_status(doc_id, "ready", n, None)
    except Exception as e:
        _fail(doc_id, doc, f"{type(e).__name__}: {e}")


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
# Startup: reload persisted documents (survive restart)
# --------------------------------------------------------------------------- #
@app.on_event("startup")
def _load_persisted_docs():
    """Reconstruct DOCS from the store: fitz from stored PDF bytes, BM25 index from saved blocks."""
    for doc_id in STORE.all_ids():
        try:
            meta = STORE.get_meta(doc_id) or {}
            kind, blob, pages = meta.get("kind"), STORE.get_blob(doc_id), STORE.get_pages(doc_id)
            if kind == "pdf" and blob:
                fdoc = fitz.open(stream=blob, filetype="pdf")
                npg = meta.get("page_count", len(fdoc))
                pblocks = [[] for _ in range(npg)]
                for pno, blks in pages.items():
                    if 1 <= pno <= npg:
                        pblocks[pno - 1] = [block_from_dict(d) for d in blks]
                st = meta.get("status", "ready")
                if st == "parsing":                           # stale at restart; saved pages usable
                    st = "ready"
                DOCS[doc_id] = {"kind": "pdf", "fitz": fdoc, "page_count": npg,
                                "sizes": [tuple(s) for s in meta.get("sizes", [])],
                                "pages_blocks": pblocks,
                                "index": DocIndex.from_blocks([b for pb in pblocks for b in pb]),
                                "status": st, "parsed_pages": meta.get("parsed_pages", npg),
                                "error": meta.get("error"), "lock": threading.Lock()}
            elif kind == "image" and blob:
                img = Image.open(io.BytesIO(blob)).convert("RGB")
                blks = [block_from_dict(d) for d in pages.get(1, [])]
                DOCS[doc_id] = {"kind": "image", "image": img, "page_w": img.width,
                                "page_h": img.height, "ground_truth": meta.get("ground_truth"),
                                "markdown": meta.get("markdown", ""),
                                "index": DocIndex.from_blocks(blks)}
        except Exception:
            continue


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
            return _start_pdf(data)                             # non-blocking; parses in background
        except Exception as e:
            return JSONResponse({"error": f"could not open PDF: {e}"}, status_code=400)
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"could not read file: {e}"}, status_code=400)
    return _store_image(image)


@app.post("/load-sample")
def load_sample():
    if not SAMPLE_PNG.exists():
        return JSONResponse({"error": "sample not generated; run samples/make_sample.py"},
                            status_code=404)
    import json
    gt = json.loads(SAMPLE_JSON.read_text(encoding="utf-8")) if SAMPLE_JSON.exists() else None
    return _store_image(Image.open(SAMPLE_PNG).convert("RGB"), ground_truth=gt)


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
    return {"status": "ok", "docs_loaded": len(DOCS), "docs_persisted": STORE.count(),
            "cloud": cloud.has_cloud()}


@app.get("/doc/{doc_id}/page.png")
def page_png(doc_id: str):
    doc = DOCS.get(doc_id)
    if not doc or doc.get("kind") != "image":
        return Response(status_code=404)
    buf = io.BytesIO()
    doc["image"].save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/doc/{doc_id}/page/{n}.png")
def pdf_page_png(doc_id: str, n: int):
    doc = DOCS.get(doc_id)
    if not doc or doc.get("kind") != "pdf" or n < 1 or n > doc["page_count"]:
        return Response(status_code=404)
    pix = doc["fitz"][n - 1].get_pixmap(dpi=130)
    return Response(content=pix.tobytes("png"), media_type="image/png")


@app.get("/doc/{doc_id}/status")
def pdf_status(doc_id: str):
    doc = DOCS.get(doc_id)
    if not doc or doc.get("kind") != "pdf":
        return JSONResponse({"error": "unknown doc"}, status_code=404)
    return {"status": doc.get("status", "ready"),
            "parsed_pages": doc.get("parsed_pages", doc["page_count"]),
            "page_count": doc["page_count"], "chunks": len(doc["index"].chunks),
            "error": doc.get("error")}


@app.get("/doc/{doc_id}/page/{n}")
def pdf_page_data(doc_id: str, n: int):
    doc = DOCS.get(doc_id)
    if not doc or doc.get("kind") != "pdf" or n < 1 or n > doc["page_count"]:
        return JSONResponse({"error": "unknown page"}, status_code=404)
    blocks = doc["pages_blocks"][n - 1]
    w, h = doc["sizes"][n - 1]
    if not blocks and doc.get("status") == "parsing" and n > doc.get("parsed_pages", 0):
        return {"page": n, "page_w": int(w), "page_h": int(h), "status": "pending"}
    return {"page": n, "page_w": int(w), "page_h": int(h), "status": "ready",
            "blocks": [block_to_dict(b) for b in blocks],
            "markdown": pdf_textlayer.page_markdown(blocks)}


# --------------------------------------------------------------------------- #
# WebSockets
# --------------------------------------------------------------------------- #
@app.websocket("/ws/parse/{doc_id}")
async def ws_parse(ws: WebSocket, doc_id: str):
    await ws.accept()
    doc = DOCS.get(doc_id)
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
        doc["markdown"] = captured.get("markdown", "")
        blocks = getattr(pipeline, "last_ordered_blocks", [])
        doc["index"] = (DocIndex.from_blocks(blocks) if blocks
                        else DocIndex.from_pages([(1, doc["markdown"])]))
        STORE.save_page(doc_id, 1, [block_to_dict(b) for b in blocks])       # persist for reload
        STORE.save_doc(doc_id, "image", {"page_w": doc["page_w"], "page_h": doc["page_h"],
                       "ground_truth": doc.get("ground_truth"), "markdown": doc["markdown"],
                       "status": "ready", "parsed_pages": 1, "page_count": 1})
        await ws.send_json({"type": "indexed", "chunks": len(doc["index"].chunks)})
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
            doc = DOCS.get(doc_id)
            if not question:
                continue
            ok, reason = guards.check_question(question)        # input guardrail
            if not ok:
                await ws.send_json({"type": "error", "error": reason})
                continue
            if not doc or not doc.get("index") or not doc["index"].chunks:
                await ws.send_json({"type": "error", "error": "still parsing — ask again in a moment"})
                continue
            # Wrap the BM25 DocIndex with the hybrid retriever (Cohere dense + RRF + rerank)
            # once per finalized index, cached on the doc so we embed only once, not per question.
            if doc.get("_retr_for") is not doc["index"]:
                doc["_retriever"] = make_retriever(doc_id, doc["index"])
                doc["_retr_for"] = doc["index"]
            engine = AgentEngine(doc["_retriever"])

            def gen():
                yield from engine.run_streaming(question, thread_id=thread_id)

            async for ev in _bridge(gen):
                await ws.send_json(ev)
    except WebSocketDisconnect:
        return
