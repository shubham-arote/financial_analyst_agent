"""
pdf_cloudvlm.py — OCR scanned PDFs with a cloud vision model (Groq / OpenRouter).

Why: local Docling OCR (RapidOCR on CPU) is ~8 min/page on a no-GPU machine and merges
words. A hosted vision model reads a rendered page in ~5s with better quality. For
*scanned* documents this is the right tool; born-digital docs should still use Docling
(precise bboxes) or the text-layer (exact text).

Each page is rendered to an image and transcribed to Markdown, then split into typed
`Block`s. Citations are page-level (the model returns text without precise boxes), so a
cited block highlights the whole page — an honest trade-off for scanned input.
Selected via SRR_PDF_PARSER=cloudvlm (needs a cloud key).
"""

from __future__ import annotations

import re

import fitz
from PIL import Image

from . import cloud
from .core import BBox, Block, BlockType

OCR_PROMPT = (
    "Transcribe this document page to clean Markdown. Use '#'/'##' for headings, keep "
    "paragraphs as paragraphs, and convert any tables to Markdown table syntax. Output "
    "ONLY the transcribed content — no commentary, no code fences."
)


def available() -> bool:
    return cloud.has_cloud()


def _md_to_blocks(md: str, pw: float, ph: float, page_no: int) -> list[Block]:
    full = (0.0, 0.0, pw, ph)
    blocks: list[Block] = []
    table_buf: list[str] = []

    def flush_table():
        if table_buf:
            blocks.append(Block(BlockType.TABLE, BBox(*full), content="\n".join(table_buf).strip(), page=page_no))
            table_buf.clear()

    for para in re.split(r"\n\s*\n", md or ""):
        para = para.strip()
        if not para:
            continue
        if para.lstrip().startswith("|"):
            table_buf.append(para)
            continue
        flush_table()
        if para.startswith("#"):
            blocks.append(Block(BlockType.TITLE, BBox(*full), content=para.lstrip("# ").strip(), page=page_no))
        else:
            blocks.append(Block(BlockType.TEXT, BBox(*full), content=para, page=page_no))
    flush_table()

    for j, b in enumerate(blocks):
        b.order = j
        b.id = j
    return blocks


def parse_pages(data: bytes, start: int, end: int, do_ocr: bool | None = None) -> dict[int, list[Block]]:
    """OCR pages [start, end) (0-based) with the cloud VLM -> {global_page_0based: [Block]}."""
    src = fitz.open(stream=data, filetype="pdf")
    out: dict[int, list[Block]] = {}
    for i in range(start, min(end, len(src))):
        page = src[i]
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        md = cloud.chat_vision(img, OCR_PROMPT, max_tokens=4096)
        if md.startswith("[") and "error" in md[:20].lower():     # cloud error -> empty page
            md = ""
        out[i] = _md_to_blocks(md, page.rect.width, page.rect.height, i + 1)
    return out
