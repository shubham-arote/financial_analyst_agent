"""
docling.py — document parsing via Docling (IBM's open-source layout parser).

Why: PyMuPDF's text layer + hand-rolled heuristics (font-size headings, find_tables,
XY-cut order) is brittle on design-heavy reports and can't read scanned / outlined-font
pages at all. Docling runs real **layout** + **table-structure** models (and OCR for
scanned pages), so headings, tables, reading order, and figures come from models rather
than guesses — and it still gives **bounding boxes**, so block-level citations keep working.

This is an *adapter*: it emits the same `Block` objects the rest of the SRR pipeline uses
(`from_blocks` -> sections -> index -> citations -> UI all stay identical). Heavier and
slower than the text-layer path; selected via SRR_PDF_PARSER=docling.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO

from ..core import BBox, Block, BlockType, ColumnAwareReadingOrder

logger = logging.getLogger("srr.docling")

# Docling DocItemLabel -> our BlockType
LABEL_MAP = {
    "title": BlockType.TITLE, "section_header": BlockType.TITLE,
    "text": BlockType.TEXT, "paragraph": BlockType.TEXT, "code": BlockType.TEXT,
    "list_item": BlockType.LIST, "document_index": BlockType.LIST,
    "table": BlockType.TABLE, "picture": BlockType.FIGURE, "caption": BlockType.CAPTION,
    "page_header": BlockType.HEADER, "page_footer": BlockType.FOOTER,
    "footnote": BlockType.FOOTER, "formula": BlockType.FORMULA,
}

_converters: dict[bool, object] = {}    # cached by do_ocr (models load once each)


def _get_converter(do_ocr: bool):
    """Construct (and cache) a Docling converter. OCR on = handles scanned (slower);
    OCR off = born-digital text + layout model only (~2x faster)."""
    if do_ocr not in _converters:
        from docling.document_converter import DocumentConverter
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption
            opts = PdfPipelineOptions()
            opts.do_ocr = do_ocr
            if do_ocr:                                    # fully-scanned pages: OCR the whole page,
                try:                                      # not just regions the layout model flags
                    opts.ocr_options.force_full_page_ocr = True
                except Exception:
                    pass
            opts.do_table_structure = True
            try:
                opts.table_structure_options.do_cell_matching = True
            except Exception:
                pass
            _converters[do_ocr] = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
            logger.info("Docling converter ready (ocr=%s)", do_ocr)
        except Exception as e:
            logger.warning("Docling options unavailable (%s); using defaults.", e)
            _converters[do_ocr] = DocumentConverter()
    return _converters[do_ocr]


def _stream(data: bytes):
    from docling.datamodel.base_models import DocumentStream
    return DocumentStream(name="document.pdf", stream=BytesIO(data))


def _label(item) -> str:
    lab = getattr(item, "label", None)
    return getattr(lab, "value", None) or (str(lab) if lab is not None else "text")


def _content(item, doc, label: str) -> str:
    if label == "table":
        for call in (lambda: item.export_to_markdown(doc), lambda: item.export_to_markdown()):
            try:
                md = call()
                if md and md.strip():
                    return md.strip()
            except Exception:
                continue
        return ""
    return (getattr(item, "text", "") or "").strip()


def _bbox_topleft(prov_item, page_h: float):
    """Return (x0, y0, x1, y1) in top-left pixel/point space."""
    bbox = prov_item.bbox
    try:
        tl = bbox.to_top_left_origin(page_h)
        x0, y0, x1, y1 = tl.l, tl.t, tl.r, tl.b
    except Exception:
        x0, y0, x1, y1 = bbox.l, bbox.t, bbox.r, bbox.b
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def parse(data: bytes, relation: ColumnAwareReadingOrder | None = None,
          page_count: int | None = None, do_ocr: bool = True) -> tuple[list[list[Block]], list[tuple]]:
    """Parse a PDF with Docling -> per-page lists of Blocks + page sizes (points)."""
    relation = relation or ColumnAwareReadingOrder()
    result = _get_converter(do_ocr).convert(_stream(data))
    doc = result.document

    pmax = page_count or (max(doc.pages.keys()) if getattr(doc, "pages", None) else 1)
    pages_blocks: list[list[Block]] = [[] for _ in range(pmax)]
    sizes: list[tuple | None] = [None] * pmax
    page_h: dict[int, float] = {}
    for pno, pg in (doc.pages or {}).items():
        try:
            w, h = float(pg.size.width), float(pg.size.height)
            page_h[pno] = h
            if 1 <= pno <= pmax:
                sizes[pno - 1] = (w, h)
        except Exception:
            pass

    for item, _level in doc.iterate_items():
        prov = getattr(item, "prov", None)
        if not prov:
            continue
        label = _label(item)
        btype = LABEL_MAP.get(label, BlockType.TEXT)
        content = _content(item, doc, label)
        if btype != BlockType.FIGURE and not content:
            continue
        p = prov[0]
        pno = getattr(p, "page_no", 1)
        if not (1 <= pno <= pmax):
            continue
        x0, y0, x1, y1 = _bbox_topleft(p, page_h.get(pno, 842.0))
        pages_blocks[pno - 1].append(Block(btype, BBox(x0, y0, x1, y1), content=content, page=pno))

    for i in range(pmax):
        if sizes[i] is None:
            sizes[i] = (595.0, 842.0)                      # A4 fallback
    for blocks in pages_blocks:                            # docling order = reading order
        for j, b in enumerate(blocks):
            b.order = j
            b.id = j
    return pages_blocks, [(int(w), int(h)) for (w, h) in sizes]


def parse_pages(data: bytes, start: int, end: int, do_ocr: bool = True) -> dict[int, list[Block]]:
    """Parse only pages [start, end) (0-based) for incremental/background processing.
    Returns {global_page_index_0based: [Block,...]} with each Block.page set 1-based global."""
    import fitz
    src = fitz.open(stream=data, filetype="pdf")
    sub = fitz.open()
    sub.insert_pdf(src, from_page=start, to_page=end - 1)
    pages_blocks, _sizes = parse(sub.tobytes(), page_count=end - start, do_ocr=do_ocr)
    out: dict[int, list[Block]] = {}
    for local_idx, blocks in enumerate(pages_blocks):
        gpage = start + local_idx + 1                       # 1-based global page number
        for b in blocks:
            b.page = gpage
        out[start + local_idx] = blocks
    return out
