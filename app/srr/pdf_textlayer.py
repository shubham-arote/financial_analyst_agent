"""
pdf_textlayer.py — born-digital PDF ingestion via PyMuPDF's text layer.

For PDFs that already carry a text layer (exported reports, 10-Ks, annual accounts),
this reads embedded text + bounding boxes + font sizes directly — no OCR, no VLM, and the
numbers are *exact*. It produces the same `Block` objects the rest of the SRR pipeline
uses, so reading order, assembly, indexing, and the canvas/answer linking work unchanged.

Fidelity work (v2):
  * tables  -> `find_tables` (text strategy + numeric guard) become real TABLE blocks
               with header-aware Markdown; overlapping text is removed so it isn't doubled.
  * headings -> font-size *mode* (page body size) + bold flag, not a single global median.
  * furniture -> running headers/footers/page-numbers that repeat across pages are detected
               document-wide and dropped from the index (they otherwise pollute retrieval).
  * reading order -> column-aware XY-cut tuned for PDF point-space gutters.
"""

from __future__ import annotations

import re
from collections import Counter

from ..config import settings
from .core import BBox, Block, BlockType, ColumnAwareReadingOrder, assemble_markdown

_NUM = re.compile(r"^[-(]?\d[\d,.\s]*\)?%?$")


def has_text_layer(doc, sample: int = 14) -> bool:
    """Born-digital pages carry real text; scanned pages are ~empty.

    Samples pages *spread across the whole document* (not just the first few), because
    annual reports open with graphical cover / contents / highlights pages that carry
    almost no extractable text — counting only those would wrongly flag the file as
    scanned. Born-digital if any sampled page has real text, or the total is substantial.
    """
    n = len(doc)
    if n == 0:
        return False
    if n <= sample:
        idxs = list(range(n))
    else:
        idxs = sorted({round(i * (n - 1) / (sample - 1)) for i in range(sample)})
    lengths = [len(doc[i].get_text("text").strip()) for i in idxs]
    return max(lengths) > 150 or sum(lengths) > 600


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _has_table_rules(page, min_segments: int = 6) -> bool:
    """Cheap gate: `find_tables()` is slow, so only run it on pages that actually contain
    enough ruled line/box segments to hold a ruled table. Prose/notes pages skip it,
    which keeps extraction fast on long reports."""
    try:
        draws = page.get_drawings()
    except Exception:
        return True                                      # if unsure, attempt detection
    segs = 0
    for d in draws:
        for it in d.get("items", []):
            op = it[0]
            segs += 1 if op == "l" else 4 if op in ("re", "qu") else 0
        if segs >= min_segments:
            return True
    return False


def _detect_tables(page, page_no: int) -> list[Block]:
    """Detect ruled, mostly-numeric tables; return TABLE blocks with header-aware md.

    Uses the conservative 'lines' strategy (real tables only) plus guards, so a table
    can't swallow a page's heading/prose the way the 'text' strategy does on uniformly
    left-aligned pages. Borderless tables stay as text (numbers still retrievable) — a
    VLM-on-table-region pass is the future upgrade for those. Gated by _has_table_rules
    (speed) and the SRR_TABLES env (set to 0 to disable table detection entirely).
    """
    out: list[Block] = []
    if not settings.tables or not _has_table_rules(page):
        return out
    page_area = page.rect.width * page.rect.height
    try:
        finder = page.find_tables()                     # default 'lines' strategy
    except Exception:
        return out
    for t in getattr(finder, "tables", []):
        try:
            cells = t.extract()
        except Exception:
            continue
        nrows = len(cells)
        ncols = max((len(r) for r in cells), default=0)
        if nrows < 2 or ncols < 2:
            continue
        x0, y0, x1, y1 = t.bbox
        if (x1 - x0) * (y1 - y0) > 0.85 * page_area:     # don't swallow the whole page
            continue
        flat = [c.strip() for row in cells for c in row if c and c.strip()]
        if not flat or sum(_NUM.match(c) is not None for c in flat) / len(flat) < 0.15:
            continue                                     # not numeric enough -> prose, skip
        try:
            md = t.to_markdown().strip()
        except Exception:
            md = "\n".join(" | ".join(c or "" for c in row) for row in cells)
        out.append(Block(BlockType.TABLE, BBox(x0, y0, x1, y1), content=md, page=page_no))
    return out


def _overlaps(b: tuple, rects: list[BBox], frac: float = 0.5) -> bool:
    x0, y0, x1, y1 = b
    area = max(1.0, (x1 - x0) * (y1 - y0))
    for r in rects:
        iw = max(0.0, min(x1, r.x1) - max(x0, r.x0))
        ih = max(0.0, min(y1, r.y1) - max(y0, r.y0))
        if iw * ih / area > frac:
            return True
    return False


# --------------------------------------------------------------------------- #
# Headings
# --------------------------------------------------------------------------- #
def _body_size(size_chars: Counter) -> float:
    """Body text size = the size covering the most *characters* (robust when a page has
    few text blocks, where a span-count mode would wrongly pick the heading size)."""
    return size_chars.most_common(1)[0][0] if size_chars else 0.0


def _classify(text: str, size: float, bold: bool, body: float, y1: float, page_h: float) -> BlockType:
    if y1 >= 0.94 * page_h and len(text) < 80:
        return BlockType.FOOTER
    if not body:
        return BlockType.TEXT
    t = text.rstrip()
    # A heading is a short, single-line, title-like label — not a sentence and not body.
    if "\n" in text or len(t) > 90 or t.endswith((".", ",", ";", ":")):
        return BlockType.TEXT
    if size >= body * 1.20:
        return BlockType.TITLE                           # clearly larger than body
    if bold and size >= body * 1.12:
        return BlockType.TITLE                           # bold AND larger (bold at body size = emphasis)
    return BlockType.TEXT


def _is_bold(span: dict) -> bool:
    return bool(span.get("flags", 0) & 16) or any(
        w in span.get("font", "").lower() for w in ("bold", "black", "semibold", "heavy"))


# --------------------------------------------------------------------------- #
# Per-page extraction
# --------------------------------------------------------------------------- #
def extract_page(page, page_no: int, relation: ColumnAwareReadingOrder | None = None) -> list[Block]:
    relation = relation or ColumnAwareReadingOrder(min_gap=10)   # tuned for point-space
    pw, ph = page.rect.width, page.rect.height
    data = page.get_text("dict")

    tables = _detect_tables(page, page_no)
    table_rects = [b.bbox for b in tables]

    raw: list[tuple] = []          # (kind, bbox, text, size, bold)
    size_chars: Counter = Counter()
    for blk in data.get("blocks", []):
        bbox = tuple(blk["bbox"])
        if blk.get("type") == 1:                         # image
            if (bbox[2] - bbox[0]) >= 40 and (bbox[3] - bbox[1]) >= 40 and not _overlaps(bbox, table_rects):
                raw.append(("figure", bbox, "", 0.0, False))
            continue
        if _overlaps(bbox, table_rects):                 # text owned by a table -> skip
            continue
        lines, bsizes, bold = [], [], False
        for line in blk.get("lines", []):
            txt = "".join(s["text"] for s in line.get("spans", []))
            if txt.strip():
                lines.append(txt)
            for s in line.get("spans", []):
                st = s["text"].strip()
                if st:
                    bsizes.append(s["size"])
                    size_chars[round(s["size"] * 2) / 2] += len(st)
                    bold = bold or _is_bold(s)
        text = "\n".join(lines).strip()
        if not text:
            continue
        raw.append(("text", bbox, text, max(bsizes) if bsizes else 0.0, bold))

    body = _body_size(size_chars)
    blocks: list[Block] = list(tables)
    title_sizes: list[tuple[Block, float]] = []
    for kind, (x0, y0, x1, y1), text, size, bold in raw:
        if kind == "figure":
            blocks.append(Block(BlockType.FIGURE, BBox(x0, y0, x1, y1), content="", page=page_no))
            continue
        bt = _classify(text, size, bold, body, y1, ph)
        b = Block(bt, BBox(x0, y0, x1, y1), content=text, page=page_no)
        blocks.append(b)
        if bt == BlockType.TITLE:
            title_sizes.append((b, size))

    # over-firing guard: a real page rarely has >4 headings; if it does, keep only the
    # largest size level and demote the rest back to text.
    if len(title_sizes) > 4:
        top = max(s for _, s in title_sizes)
        for b, s in title_sizes:
            if s < top * 0.92:
                b.type = BlockType.TEXT

    relation.order(blocks, (pw, ph))
    for i, b in enumerate(blocks):
        b.id = i
    return blocks


# --------------------------------------------------------------------------- #
# Document-wide: drop repeated running headers / footers / page numbers
# --------------------------------------------------------------------------- #
def _norm(t: str) -> str:
    return re.sub(r"\d+", "#", (t or "").strip().lower())[:60]


def _band(b: Block, page_h: float) -> str | None:
    if b.bbox.y1 <= 0.10 * page_h:
        return "top"
    if b.bbox.y0 >= 0.90 * page_h:
        return "bot"
    return None


def mark_repeated_furniture(pages_blocks: list[list[Block]], sizes: list[tuple]) -> int:
    """Text repeating in the top/bottom band across many pages is furniture -> exclude."""
    n = len(pages_blocks)
    if n < 4:
        return 0
    counts: Counter = Counter()
    for blocks, (_pw, ph) in zip(pages_blocks, sizes):
        for b in blocks:
            band = _band(b, ph)
            if band and (b.content or "").strip():
                counts[(_norm(b.content), band)] += 1
    threshold = max(3, int(0.3 * n))
    repeated = {k for k, c in counts.items() if c >= threshold and k[0]}
    marked = 0
    for blocks, (_pw, ph) in zip(pages_blocks, sizes):
        for b in blocks:
            band = _band(b, ph)
            if band and (_norm(b.content), band) in repeated:
                b.type = BlockType.HEADER if band == "top" else BlockType.FOOTER
                b.order = None
                marked += 1
    return marked


def page_markdown(blocks: list[Block]) -> str:
    body = sorted([b for b in blocks if b.order is not None], key=lambda b: b.order)
    return assemble_markdown(body)
