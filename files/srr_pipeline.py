"""
srr_pipeline.py
================
A MonkeyOCR-style Structure-Recognition-Relation (SRR) document parsing pipeline.

Paradigm (Li et al., MonkeyOCR 2025):
    Structure   -> "Where is it?"   : block-level layout detection  (tiny detector, e.g. DocLayoutYOLO)
    Recognition -> "What is it?"     : per-block content recognition (small VLM, run in parallel)
    Relation    -> "How organized?"  : reading-order / logical ordering over the blocks
    Assembly    -> stitch ordered, recognized blocks into Markdown / JSON

Why this shape:
    - The detector is tiny and *swappable* (YOLO-class). Adding a new element type
      = retrain a detection head, not the whole VLM.
    - Recognition is the slow stage but is embarrassingly parallel: every cropped
      block is an independent VLM call, so we batch / thread them.
    - A small VLM on a *cropped region* solves a far easier problem than a giant
      VLM decoding a whole dense page in one long autoregressive sequence.

The heavy model calls (detector weights, VLM inference) are isolated behind
Protocols and marked  # >>> STUB <<<  so you can plug in real backends
(vLLM OpenAI-compatible server, transformers, doclayout_yolo, etc.) without
touching the orchestration logic.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

from PIL import Image

logger = logging.getLogger("srr")


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
    HEADER = "header"   # running header / page furniture
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"


# Block types that are "furniture" and usually excluded from the main reading flow.
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
    def width(self) -> float:
        return self.x1 - self.x0

    def x_overlap(self, other: "BBox") -> float:
        lo, hi = max(self.x0, other.x0), min(self.x1, other.x1)
        return max(0.0, hi - lo)


@dataclass
class Block:
    type: BlockType
    bbox: BBox
    score: float = 1.0
    page: int = 0
    content: str | None = None      # filled by the Recognition stage
    order: int | None = None        # filled by the Relation stage


# --------------------------------------------------------------------------- #
# Stage 1 — STRUCTURE : layout detection (the tiny model)
# --------------------------------------------------------------------------- #
@runtime_checkable
class LayoutDetector(Protocol):
    def detect(self, image: Image.Image, page: int = 0) -> list[Block]: ...


class DocLayoutYOLODetector:
    """
    Wraps a YOLO-class layout detector (MonkeyOCR-3B uses DocLayoutYOLO here).
    Real backend: `pip install doclayout-yolo` then load the .pt checkpoint.
    """

    # Map the detector's class names -> our BlockType. Adjust to your checkpoint.
    LABEL_MAP = {
        "plain text": BlockType.TEXT,
        "title": BlockType.TITLE,
        "list": BlockType.LIST,
        "table": BlockType.TABLE,
        "isolate_formula": BlockType.FORMULA,
        "figure": BlockType.FIGURE,
        "figure_caption": BlockType.CAPTION,
        "table_caption": BlockType.CAPTION,
        "abandon": BlockType.FOOTER,   # headers/footers/page-nums often "abandon"
    }

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 1024):
        self.conf, self.imgsz = conf, imgsz
        # >>> STUB <<< load once, reuse. Kept lazy so the file imports without weights.
        # from doclayout_yolo import YOLOv10
        # self.model = YOLOv10(weights)
        self.weights = weights
        self.model = None

    def detect(self, image: Image.Image, page: int = 0) -> list[Block]:
        if self.model is None:
            # >>> STUB <<< return a single full-page text block so the rest of the
            # pipeline is exercisable end-to-end without real weights.
            w, h = image.size
            return [Block(BlockType.TEXT, BBox(0, 0, w, h), page=page)]

        # res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz)[0]
        # blocks = []
        # for b in res.boxes:
        #     name = res.names[int(b.cls)]
        #     btype = self.LABEL_MAP.get(name, BlockType.TEXT)
        #     x0, y0, x1, y1 = b.xyxy[0].tolist()
        #     blocks.append(Block(btype, BBox(x0, y0, x1, y1), float(b.conf), page))
        # return blocks
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Stage 2 — RECOGNITION : per-block content (the small VLM)
# --------------------------------------------------------------------------- #
@runtime_checkable
class VLMClient(Protocol):
    """A minimal image+prompt -> text interface. Implement for vLLM / transformers / API."""
    def generate(self, image: Image.Image, prompt: str) -> str: ...


class OpenAICompatVLM:
    """
    Talks to any OpenAI-compatible /v1/chat/completions endpoint.
    This is how MonkeyOCR / PaddleOCR-VL / Qwen-VL are typically served via vLLM:

        vllm serve echo840/MonkeyOCR --trust-remote-code   # or PaddlePaddle/PaddleOCR-VL
    """

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 max_tokens: int = 4096, temperature: float = 0.0):
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature
        # >>> STUB <<< construct the client lazily.
        # from openai import OpenAI
        # self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_url, self.client = base_url, None

    @staticmethod
    def _data_url(image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def generate(self, image: Image.Image, prompt: str) -> str:
        if self.client is None:
            return "[recognized-text-stub]"   # >>> STUB <<<
        # resp = self.client.chat.completions.create(
        #     model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
        #     messages=[{"role": "user", "content": [
        #         {"type": "text", "text": prompt},
        #         {"type": "image_url", "image_url": {"url": self._data_url(image)}},
        #     ]}],
        # )
        # return resp.choices[0].message.content.strip()
        raise NotImplementedError


# Prompt routing: one unified VLM, but the *task* changes per block type.
# This is the core of "Recognition": ask the right question for each region.
PROMPTS: dict[BlockType, str] = {
    BlockType.TEXT:    "OCR the text in this image region. Output plain Markdown, preserve line breaks where meaningful. No commentary.",
    BlockType.TITLE:   "OCR this heading. Output only the heading text as Markdown (#-level appropriate to its size).",
    BlockType.LIST:    "OCR this list. Output as a Markdown bulleted or numbered list.",
    BlockType.CAPTION: "OCR this caption verbatim. Output plain text only.",
    BlockType.TABLE:   "Convert this table to clean HTML (<table>...). Preserve all cells, row/col spans, and merged headers. Output only HTML.",
    BlockType.FORMULA: "Convert this formula to LaTeX. Output only the LaTeX, no $ delimiters.",
    BlockType.FIGURE:  "Describe this figure in one concise sentence for an alt-text caption.",
}


@dataclass
class Recognizer:
    """
    Crops each block and recognizes it with the small VLM. Parallel by design:
    MonkeyOCR notes the block-level recognition stage is inherently parallelizable.
    """
    vlm: VLMClient
    max_workers: int = 8
    pad: int = 4   # px padding around crops to avoid clipping glyphs/super-scripts

    def _recognize_one(self, page_img: Image.Image, block: Block) -> Block:
        if block.type in FURNITURE:
            block.content = ""          # drop furniture from output
            return block
        if block.type == BlockType.FIGURE:
            # Keep a placeholder + alt text; the raw crop can be saved separately.
            crop = self._crop(page_img, block.bbox)
            alt = self.vlm.generate(crop, PROMPTS[BlockType.FIGURE])
            block.content = f"![{alt}](figure_p{block.page}_{int(block.bbox.x0)}.png)"
            return block

        crop = self._crop(page_img, block.bbox)
        prompt = PROMPTS.get(block.type, PROMPTS[BlockType.TEXT])
        block.content = self.vlm.generate(crop, prompt)
        return block

    def _crop(self, img: Image.Image, b: BBox) -> Image.Image:
        w, h = img.size
        box = (max(0, b.x0 - self.pad), max(0, b.y0 - self.pad),
               min(w, b.x1 + self.pad), min(h, b.y1 + self.pad))
        return img.crop(tuple(map(int, box)))

    def recognize(self, page_img: Image.Image, blocks: Sequence[Block]) -> list[Block]:
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda b: self._recognize_one(page_img, b), blocks))


# --------------------------------------------------------------------------- #
# Stage 3 — RELATION : reading order ("How is it organized?")
# --------------------------------------------------------------------------- #
@runtime_checkable
class RelationPredictor(Protocol):
    def order(self, blocks: Sequence[Block], page_size: tuple[int, int]) -> list[Block]: ...


class ColumnAwareReadingOrder:
    """
    Default heuristic relation model: detect columns by horizontal overlap, sort
    columns left-to-right, and within each column sort top-to-bottom (an XY-cut
    flavour). For complex/unconstrained layouts, swap in a *learned* relation
    predictor (MonkeyOCR trains one) behind this same interface.
    """

    def __init__(self, col_overlap_ratio: float = 0.5):
        self.col_overlap_ratio = col_overlap_ratio

    def order(self, blocks: Sequence[Block], page_size: tuple[int, int]) -> list[Block]:
        body = [b for b in blocks if b.type not in FURNITURE]
        body.sort(key=lambda b: (b.bbox.y0, b.bbox.x0))

        columns: list[list[Block]] = []
        for b in body:
            placed = False
            for col in columns:
                ref = col[0].bbox
                ov = ref.x_overlap(b.bbox)
                if ov >= self.col_overlap_ratio * min(ref.width, b.bbox.width):
                    col.append(b)
                    placed = True
                    break
            if not placed:
                columns.append([b])

        columns.sort(key=lambda col: min(bl.bbox.cx for bl in col))  # left -> right
        ordered: list[Block] = []
        for col in columns:
            col.sort(key=lambda bl: bl.bbox.y0)                      # top -> bottom
            ordered.extend(col)

        for i, b in enumerate(ordered):
            b.order = i
        return ordered


# --------------------------------------------------------------------------- #
# Assembly — ordered, recognized blocks -> Markdown / JSON
# --------------------------------------------------------------------------- #
def assemble_markdown(ordered: Sequence[Block]) -> str:
    parts: list[str] = []
    for b in ordered:
        if not b.content:
            continue
        if b.type == BlockType.TITLE:
            parts.append(f"## {b.content}")
        elif b.type == BlockType.FORMULA:
            parts.append(f"$$\n{b.content}\n$$")
        else:  # TEXT, LIST, TABLE(html), FIGURE(md img), CAPTION
            parts.append(b.content)
    return "\n\n".join(parts)


def assemble_json(ordered: Sequence[Block]) -> str:
    return json.dumps(
        [{"page": b.page, "order": b.order, "type": b.type.value,
          "bbox": [b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1],
          "content": b.content} for b in ordered],
        ensure_ascii=False, indent=2,
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
@dataclass
class SRRResult:
    markdown: str
    blocks: list[Block]

    def to_json(self) -> str:
        return assemble_json(self.blocks)


class SRRPipeline:
    def __init__(self, detector: LayoutDetector, recognizer: Recognizer,
                 relation: RelationPredictor | None = None):
        self.detector = detector
        self.recognizer = recognizer
        self.relation = relation or ColumnAwareReadingOrder()

    def parse_page(self, image: Image.Image, page: int = 0) -> list[Block]:
        blocks = self.detector.detect(image, page)                 # STRUCTURE
        blocks = self.recognizer.recognize(image, blocks)          # RECOGNITION (parallel)
        blocks = self.relation.order(blocks, image.size)           # RELATION
        logger.info("page %d: %d blocks", page, len(blocks))
        return blocks

    def parse_image(self, image: Image.Image) -> SRRResult:
        blocks = self.parse_page(image, 0)
        return SRRResult(assemble_markdown(blocks), blocks)

    def parse_pdf(self, pdf_path: str, dpi: int = 200) -> SRRResult:
        """Rasterize each page, run SRR, then concatenate. Needs `pdf2image` + poppler."""
        from pdf2image import convert_from_path  # local import: optional dep
        pages = convert_from_path(pdf_path, dpi=dpi)
        all_blocks: list[Block] = []
        md_pages: list[str] = []
        for i, page_img in enumerate(pages):
            blocks = self.parse_page(page_img.convert("RGB"), page=i)
            all_blocks.extend(blocks)
            md_pages.append(assemble_markdown(blocks))
        return SRRResult("\n\n---\n\n".join(md_pages), all_blocks)


# --------------------------------------------------------------------------- #
# Factory + demo
# --------------------------------------------------------------------------- #
def build_default_pipeline(
    detector_weights: str = "doclayout_yolo_docstructbench.pt",
    vlm_base_url: str = "http://localhost:8000/v1",
    vlm_model: str = "echo840/MonkeyOCR",
    max_workers: int = 8,
) -> SRRPipeline:
    detector = DocLayoutYOLODetector(detector_weights)
    vlm = OpenAICompatVLM(base_url=vlm_base_url, model=vlm_model)
    recognizer = Recognizer(vlm=vlm, max_workers=max_workers)
    return SRRPipeline(detector, recognizer, ColumnAwareReadingOrder())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Runs end-to-end on stubs (no weights needed) so you can see the data flow:
    pipe = build_default_pipeline()
    img = Image.new("RGB", (1024, 1448), "white")
    result = pipe.parse_image(img)
    print("---- MARKDOWN ----")
    print(result.markdown)
    print("---- JSON ----")
    print(result.to_json())
