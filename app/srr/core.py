"""
core.py — stable, model-agnostic SRR primitives.

Adapted from the reference `files/srr_pipeline.py` (MonkeyOCR-style SRR paradigm):
the data model, the Protocols that keep every stage swappable, the prompt routing,
the column-aware reading-order heuristic, and the Markdown/JSON assembly. Logic is
preserved; this copy lives inside the app package so the MVP is self-contained and
doesn't import across a space-containing reference path.

    Structure   -> "Where is it?"   : layout detection           (LayoutDetector)
    Recognition -> "What is it?"     : per-block content          (BlockRecognizer / VLMClient)
    Relation    -> "How organized?"  : reading order              (RelationPredictor)
    Assembly    -> ordered+recognized blocks -> Markdown / JSON
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from PIL import Image


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
class BlockType(str, Enum):
    TEXT = "text"
    TITLE = "title"
    LIST = "list"
    TABLE = "table"
    FORMULA = "formula"
    FIGURE = "figure"
    CAPTION = "caption"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


# "Furniture" — page chrome excluded from the main reading flow.
FURNITURE = {BlockType.HEADER, BlockType.FOOTER, BlockType.PAGE_NUMBER}


@dataclass
class BBox:
    """Pixel-space bounding box, origin top-left."""
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def x_overlap(self, other: "BBox") -> float:
        lo, hi = max(self.x0, other.x0), min(self.x1, other.x1)
        return max(0.0, hi - lo)

    def y_overlap(self, other: "BBox") -> float:
        lo, hi = max(self.y0, other.y0), min(self.y1, other.y1)
        return max(0.0, hi - lo)

    def iou(self, other: "BBox") -> float:
        inter = self.x_overlap(other) * self.y_overlap(other)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass
class Block:
    type: BlockType
    bbox: BBox
    score: float = 1.0
    page: int = 0
    id: int | None = None           # assigned by the streaming orchestrator
    content: str | None = None      # filled by the Recognition stage
    order: int | None = None        # filled by the Relation stage


def block_to_dict(b: Block) -> dict:
    return {
        "id": b.id,
        "type": b.type.value,
        "bbox": [int(b.bbox.x0), int(b.bbox.y0), int(b.bbox.x1), int(b.bbox.y1)],
        "score": round(float(b.score), 3),
        "page": b.page,
        "order": b.order,
        "content": b.content,
    }


def block_from_dict(d: dict) -> Block:
    """Reconstruct a Block from block_to_dict() output (for persistence/reload)."""
    b = Block(type=BlockType(d["type"]), bbox=BBox(*d["bbox"]),
              score=float(d.get("score", 1.0)), page=int(d.get("page", 0)))
    b.id = d.get("id")
    b.order = d.get("order")
    b.content = d.get("content")
    return b


# --------------------------------------------------------------------------- #
# Stage protocols — every stage is swappable behind these
# --------------------------------------------------------------------------- #
@runtime_checkable
class LayoutDetector(Protocol):
    label: str
    def detect(self, image: Image.Image, page: int = 0) -> list[Block]: ...


@runtime_checkable
class VLMClient(Protocol):
    """Minimal image+prompt -> text backend (cloud VLM, EasyOCR, stub, ...)."""
    def generate(self, image: Image.Image, prompt: str) -> str: ...


@runtime_checkable
class BlockRecognizer(Protocol):
    """Stage-level recognizer: turn one detected block into text/markdown/html."""
    label: str
    def recognize_one(self, page_img: Image.Image, block: Block) -> str: ...


@runtime_checkable
class RelationPredictor(Protocol):
    def order(self, blocks: Sequence[Block], page_size: tuple[int, int]) -> list[Block]: ...


# --------------------------------------------------------------------------- #
# Prompt routing — one VLM, but the *task* changes per block type
# --------------------------------------------------------------------------- #
PROMPTS: dict[BlockType, str] = {
    BlockType.TEXT:    "OCR the text in this image region. Output plain Markdown, preserve meaningful line breaks. No commentary.",
    BlockType.TITLE:   "OCR this heading. Output only the heading text, no Markdown hashes, no commentary.",
    BlockType.LIST:    "OCR this list. Output as a Markdown bulleted or numbered list. No commentary.",
    BlockType.CAPTION: "OCR this caption verbatim. Output plain text only.",
    BlockType.TABLE:   "Convert this table to clean Markdown table syntax. Preserve every cell and header. Output only the table.",
    BlockType.FORMULA: "Convert this formula to LaTeX. Output only the LaTeX, no $ delimiters, no commentary.",
    BlockType.FIGURE:  "Describe this figure in one concise sentence suitable as alt text. No commentary.",
}


# --------------------------------------------------------------------------- #
# Relation — column-aware reading order (XY-cut flavour)
# --------------------------------------------------------------------------- #
class ColumnAwareReadingOrder:
    """
    Reading order via recursive XY-cut over the *detected blocks* (not pixels).
    Repeatedly split the block set by the widest whitespace gap — horizontal first
    (top-to-bottom bands), then vertical (left-to-right columns) — and recurse. This
    is the classic, robust ordering for Manhattan layouts: a full-width title splits
    off first; the two-column body then splits at the gutter and each column reads
    top-to-bottom; full-width sections below read in order. Swap in a *learned*
    predictor (MonkeyOCR trains one) behind this same interface for messy layouts.
    """

    label = "recursive XY-cut (heuristic)"

    def __init__(self, min_gap: int = 14):
        self.min_gap = min_gap

    def _gap_split(self, blocks: list[Block], axis: str) -> list[list[Block]]:
        """Split blocks where a whitespace gap on `axis` exceeds min_gap."""
        if axis == "y":
            lo, hi = (lambda b: b.bbox.y0), (lambda b: b.bbox.y1)
        else:
            lo, hi = (lambda b: b.bbox.x0), (lambda b: b.bbox.x1)
        sb = sorted(blocks, key=lo)
        groups: list[list[Block]] = [[sb[0]]]
        cur_end = hi(sb[0])
        for b in sb[1:]:
            if lo(b) - cur_end > self.min_gap:
                groups.append([b])
                cur_end = hi(b)
            else:
                groups[-1].append(b)
                cur_end = max(cur_end, hi(b))
        return groups

    def _cut(self, blocks: list[Block], out: list[Block]) -> None:
        if len(blocks) <= 1:
            out.extend(blocks)
            return
        rows = self._gap_split(blocks, "y")        # horizontal cut: top -> bottom
        if len(rows) > 1:
            for g in rows:
                self._cut(g, out)
            return
        cols = self._gap_split(blocks, "x")        # vertical cut: left -> right
        if len(cols) > 1:
            for g in cols:
                self._cut(g, out)
            return
        out.extend(sorted(blocks, key=lambda b: (b.bbox.y0, b.bbox.x0)))  # atomic

    def order(self, blocks: Sequence[Block], page_size: tuple[int, int]) -> list[Block]:
        body = [b for b in blocks if b.type not in FURNITURE]
        ordered: list[Block] = []
        if body:
            self._cut(list(body), ordered)
        for i, b in enumerate(ordered):
            b.order = i
        return ordered


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def assemble_markdown(ordered: Sequence[Block]) -> str:
    parts: list[str] = []
    for b in ordered:
        if not b.content:
            continue
        content = b.content.strip()
        if not content:
            continue
        if b.type == BlockType.TITLE:
            parts.append(f"## {content}")
        elif b.type == BlockType.FORMULA:
            parts.append(f"$$\n{content}\n$$")
        elif b.type == BlockType.CAPTION:
            parts.append(f"*{content}*")
        else:  # TEXT, LIST, TABLE, FIGURE
            parts.append(content)
    return "\n\n".join(parts)


def assemble_json(ordered: Sequence[Block]) -> str:
    return json.dumps([block_to_dict(b) for b in ordered], ensure_ascii=False, indent=2)
