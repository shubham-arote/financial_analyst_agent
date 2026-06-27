"""
detector.py — Stage 1, STRUCTURE ("where is it?"). The tiny, swappable detector.

Default: a pure-Python **recursive XY-cut** layout detector (numpy + PIL). No model
weights, no GPU, no downloads — so the live "layout detection side window" animates on
day one. It alternates horizontal/vertical whitespace cuts to segment the page into
blocks, then applies light heuristics to label them (title / table / figure / caption /
text / footer).

Upgrade (env `SRR_DETECTOR=doclayout`): DocLayoutYOLO — the exact YOLO-class detector
MonkeyOCR-3B uses. Same `LayoutDetector` interface, so nothing downstream changes.
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from .core import BBox, Block, BlockType

logger = logging.getLogger("srr.detector")


# --------------------------------------------------------------------------- #
# Default: pure-Python recursive XY-cut
# --------------------------------------------------------------------------- #
class HeuristicDetector:
    """Recursive XY-cut segmentation + heuristic block typing. CPU-only, no deps."""

    label = "heuristic XY-cut (local, no model)"

    def __init__(self, ink_threshold: int = 200, min_gap_v: int = 18,
                 min_gap_h: int = 30, min_block: int = 14, max_depth: int = 6):
        self.ink_threshold = ink_threshold
        self.min_gap_v = min_gap_v        # row-gap that separates paragraphs (not lines)
        self.min_gap_h = min_gap_h        # col-gap that separates columns (not words)
        self.min_block = min_block
        self.max_depth = max_depth

    # ---- public API ---- #
    def detect(self, image: Image.Image, page: int = 0) -> list[Block]:
        gray = np.asarray(image.convert("L"))
        ink = (gray < self.ink_threshold).astype(np.uint8)  # 1 = dark pixel
        H, W = ink.shape

        regions: list[tuple[int, int, int, int]] = []
        self._xycut(ink, 0, 0, regions, depth=0)

        blocks: list[Block] = []
        for (x0, y0, x1, y1) in regions:
            sub = ink[y0:y1, x0:x1]
            btype = self._classify(sub, x0, y0, x1, y1, W, H)
            blocks.append(Block(type=btype, bbox=BBox(x0, y0, x1, y1), score=1.0, page=page))

        self._tag_captions(blocks)
        # natural top-left ordering for stable ids (Relation re-orders later)
        blocks.sort(key=lambda b: (round(b.bbox.y0 / 20), b.bbox.x0))
        return blocks

    # ---- XY-cut ---- #
    def _split(self, is_content: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
        """Group runs of content cells, merging across gaps shorter than min_gap."""
        idx = np.nonzero(is_content)[0]
        if len(idx) == 0:
            return []
        groups: list[tuple[int, int]] = []
        start = prev = int(idx[0])
        for i in idx[1:]:
            i = int(i)
            if i - prev > min_gap:
                groups.append((start, prev + 1))
                start = i
            prev = i
        groups.append((start, prev + 1))
        return [(a, b) for (a, b) in groups if (b - a) >= self.min_block]

    def _xycut(self, sub: np.ndarray, ox: int, oy: int,
               out: list[tuple[int, int, int, int]], depth: int) -> None:
        h, w = sub.shape
        if h < self.min_block or w < self.min_block:
            self._emit(sub, ox, oy, out)
            return

        if depth < self.max_depth:
            rthr = max(1.0, 0.02 * w)
            rows = self._split(sub.sum(axis=1) >= rthr, self.min_gap_v)
            if len(rows) > 1:
                for (a, b) in rows:
                    self._xycut(sub[a:b, :], ox, oy + a, out, depth + 1)
                return

            cthr = max(1.0, 0.02 * h)
            cols = self._split(sub.sum(axis=0) >= cthr, self.min_gap_h)
            if len(cols) > 1:
                for (a, b) in cols:
                    self._xycut(sub[:, a:b], ox + a, oy, out, depth + 1)
                return

        self._emit(sub, ox, oy, out)

    def _emit(self, sub: np.ndarray, ox: int, oy: int,
              out: list[tuple[int, int, int, int]]) -> None:
        ys, xs = np.nonzero(sub)
        if len(xs) == 0:
            return
        x0, x1 = ox + int(xs.min()), ox + int(xs.max()) + 1
        y0, y1 = oy + int(ys.min()), oy + int(ys.max()) + 1
        if (x1 - x0) >= self.min_block and (y1 - y0) >= 4:
            out.append((x0, y0, x1, y1))

    # ---- typing heuristics ---- #
    def _classify(self, sub: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                  W: int, H: int) -> BlockType:
        h, w = y1 - y0, x1 - x0
        density = float(sub.mean()) if sub.size else 0.0
        row_frac = sub.mean(axis=1) if sub.size else np.zeros(1)
        rules = int((row_frac > 0.85).sum())                 # near-full-width ink = a rule
        text_lines = len(self._split(row_frac >= 0.04, 3))   # rough line count

        if y1 >= 0.95 * H:                                   # bottom strip -> footer
            return BlockType.FOOTER
        if y1 <= 0.16 * H and w > 0.5 * W:                   # big banner at top
            return BlockType.TITLE
        if rules >= 2:                                       # >=2 drawn horizontal rules
            return BlockType.TABLE
        if density > 0.16 and text_lines <= 2 and h > 120 and w > 120:
            return BlockType.FIGURE                          # large dense 2-D ink blob
        if text_lines <= 1 and h < 0.05 * H and density > 0.04 and w < 0.7 * W:
            return BlockType.TITLE                           # short bold heading
        return BlockType.TEXT

    def _tag_captions(self, blocks: list[Block]) -> None:
        """A short text block just under a figure/table becomes a caption."""
        anchors = [b for b in blocks if b.type in (BlockType.FIGURE, BlockType.TABLE)]
        for b in blocks:
            if b.type != BlockType.TEXT or b.bbox.height > 60:
                continue
            for a in anchors:
                gap = b.bbox.y0 - a.bbox.y1
                if 0 <= gap <= 40 and a.bbox.x_overlap(b.bbox) > 0.3 * b.bbox.width:
                    b.type = BlockType.CAPTION
                    break


# --------------------------------------------------------------------------- #
# Optional upgrade: DocLayoutYOLO (the MonkeyOCR detector)
# --------------------------------------------------------------------------- #
class DocLayoutYOLODetector:
    """`pip install doclayout-yolo` + a .pt checkpoint. Lazy-loaded."""

    label = "DocLayoutYOLO (YOLOv10)"

    LABEL_MAP = {
        "plain text": BlockType.TEXT, "title": BlockType.TITLE, "list": BlockType.LIST,
        "table": BlockType.TABLE, "isolate_formula": BlockType.FORMULA,
        "figure": BlockType.FIGURE, "figure_caption": BlockType.CAPTION,
        "table_caption": BlockType.CAPTION, "abandon": BlockType.FOOTER,
    }

    def __init__(self, weights: str, conf: float = 0.25, imgsz: int = 1024):
        self.weights, self.conf, self.imgsz = weights, conf, imgsz
        from doclayout_yolo import YOLOv10  # raises if not installed -> factory falls back
        self.model = YOLOv10(weights)

    def detect(self, image: Image.Image, page: int = 0) -> list[Block]:
        res = self.model.predict(image, conf=self.conf, imgsz=self.imgsz, device="cpu")[0]
        blocks: list[Block] = []
        for b in res.boxes:
            name = res.names[int(b.cls)]
            x0, y0, x1, y1 = b.xyxy[0].tolist()
            blocks.append(Block(self.LABEL_MAP.get(name, BlockType.TEXT),
                                BBox(x0, y0, x1, y1), float(b.conf), page))
        return blocks
