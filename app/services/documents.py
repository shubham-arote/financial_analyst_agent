"""
documents.py — document lifecycle service.

Owns the in-memory registry of live documents (`DOCS`) and the durable `DocStore`, plus
everything that creates, parses, persists, and reloads a document. The server's routes are
thin wrappers over the functions here; nothing in this module knows about FastAPI/WebSockets.

  upload image -> store_image()           (single page, ready immediately)
  upload PDF   -> start_pdf()              (returns at once; parses in a background thread)
                  -> _parse_pdf_bg()       (textlayer | cloudvlm | docling, auto-routed)
  live image parse finishes -> finalize_image()
  process restart -> load_persisted_docs() (rebuild DOCS from the DocStore)
"""

from __future__ import annotations

import importlib.util
import io
import threading
import uuid

import fitz  # PyMuPDF
from PIL import Image

from ..config import settings
from ..retrieval import DocIndex
from ..srr import cloud  # noqa: F401  (kept for parity; provider checks live in parsers)
from ..srr.core import ColumnAwareReadingOrder, block_from_dict, block_to_dict
from ..srr.parsers import textlayer
from ..storage import get_store

MAX_W = 1600  # downscale huge image uploads (canvas perf + VLM cost)

DOCS: dict[str, dict] = {}   # doc_id -> live per-doc state (reconstructable from STORE)
STORE = get_store()          # durable persistence (survives restart; SQLite -> GCS on cloud)


def get(doc_id: str) -> dict | None:
    return DOCS.get(doc_id)


def docling_available() -> bool:
    return importlib.util.find_spec("docling") is not None


# --------------------------------------------------------------------------- #
# Create
# --------------------------------------------------------------------------- #
def store_image(image: Image.Image, ground_truth=None) -> dict:
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


def start_pdf(data: bytes) -> dict:
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


def finalize_image(doc_id: str, doc: dict, blocks: list, markdown: str) -> int:
    """After the live single-image SRR parse: build the index, persist, return chunk count."""
    doc["markdown"] = markdown
    doc["index"] = (DocIndex.from_blocks(blocks) if blocks
                    else DocIndex.from_pages([(1, markdown)]))
    STORE.save_page(doc_id, 1, [block_to_dict(b) for b in blocks])
    STORE.save_doc(doc_id, "image", {"page_w": doc["page_w"], "page_h": doc["page_h"],
                   "ground_truth": doc.get("ground_truth"), "markdown": markdown,
                   "status": "ready", "parsed_pages": 1, "page_count": 1})
    return len(doc["index"].chunks)


# --------------------------------------------------------------------------- #
# Background PDF parsing
# --------------------------------------------------------------------------- #
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


def _parse_pdf_bg(doc_id: str, data: bytes) -> None:
    """Background worker: fill pages_blocks incrementally, rebuild the index, persist as it goes."""
    doc = DOCS.get(doc_id)
    if not doc:
        return
    parser = settings.pdf_parser
    relation = ColumnAwareReadingOrder()
    n = doc["page_count"]
    try:
        if parser == "auto":                              # best available per document
            from ..srr.parsers import cloudvlm
            scanned = not textlayer.has_text_layer(doc["fitz"])
            if scanned:
                parser = "cloudvlm" if cloudvlm.available() else "docling"
            else:                                         # born-digital: Docling if installed, else fast text layer
                parser = "docling" if docling_available() else "textlayer"
        if parser == "textlayer":
            fitz_doc = doc["fitz"]
            if not textlayer.has_text_layer(fitz_doc):
                return _fail(doc_id, doc, "This PDF has no text layer (looks scanned).")
            pb = [textlayer.extract_page(fitz_doc[i], i + 1, relation) for i in range(n)]
            textlayer.mark_repeated_furniture(pb, doc["sizes"])
            doc["pages_blocks"] = pb
            doc["parsed_pages"] = n
            _rebuild_index(doc)
            _save_pages(doc_id, doc, 0, n)
        elif parser == "cloudvlm":
            from ..srr.parsers import cloudvlm
            if not cloudvlm.available():
                return _fail(doc_id, doc, "Cloud VLM OCR needs GROQ_API_KEY or OPENROUTER_API_KEY in .env.")
            chunk = settings.vlm_chunk
            for a in range(0, n, chunk):
                b = min(a + chunk, n)
                for gp0, blocks in cloudvlm.parse_pages(data, a, b).items():
                    if 0 <= gp0 < n:
                        doc["pages_blocks"][gp0] = blocks
                doc["parsed_pages"] = b
                _rebuild_index(doc)
                _save_pages(doc_id, doc, a, b)
        else:
            from ..srr.parsers import docling
            env_ocr = settings.docling_ocr
            do_ocr = (not textlayer.has_text_layer(doc["fitz"])) if env_ocr is None else (env_ocr != "0")
            chunk = settings.docling_chunk
            for a in range(0, n, chunk):
                b = min(a + chunk, n)
                for gp0, blocks in docling.parse_pages(data, a, b, do_ocr=do_ocr).items():
                    if 0 <= gp0 < n:
                        doc["pages_blocks"][gp0] = blocks
                doc["parsed_pages"] = b
                _rebuild_index(doc)                        # query-as-you-go
                _save_pages(doc_id, doc, a, b)
        doc["status"] = "ready"
        STORE.update_status(doc_id, "ready", n, None)
    except Exception as e:
        _fail(doc_id, doc, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# Restart recovery
# --------------------------------------------------------------------------- #
def load_persisted_docs() -> None:
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
