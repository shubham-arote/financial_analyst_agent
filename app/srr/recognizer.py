"""
recognizer.py — Stage 2, RECOGNITION ("what is it?"). The small VLM (or fallback).

Two layers:
  * VLMClient backends (image+prompt -> text): CloudVLM (Groq/OpenRouter), StubVLM,
    EasyOCRVLM (optional local).
  * BlockRecognizer (stage-level): crops each detected block, routes the right prompt
    per block type, and returns Markdown/table/LaTeX. GroundTruthRecognizer is a special
    offline backend for the bundled sample (uses the known text we drew into it), so the
    end-to-end Q&A demo works with zero network and zero key.
"""

from __future__ import annotations

from PIL import Image

from . import cloud
from .core import BBox, Block, BlockType, PROMPTS, FURNITURE


# --------------------------------------------------------------------------- #
# VLMClient backends
# --------------------------------------------------------------------------- #
class CloudVLM:
    """Routes the crop to a free OpenAI-compatible vision model."""
    label = property(lambda self: f"cloud VLM ({cloud.provider_label()})")

    def generate(self, image: Image.Image, prompt: str) -> str:
        return cloud.chat_vision(image, prompt)


class StubVLM:
    """Last-resort: keeps the UI alive when there's no key and no ground truth."""
    label = "stub (no recognition backend)"

    def generate(self, image: Image.Image, prompt: str) -> str:
        return "[set GROQ_API_KEY or OPENROUTER_API_KEY for real recognition]"


class EasyOCRVLM:
    """Optional fully-offline OCR. `pip install easyocr` into the venv to enable."""
    label = "EasyOCR (local CPU)"
    _reader = None

    def _get_reader(self):
        if EasyOCRVLM._reader is None:
            import easyocr  # raises if not installed
            EasyOCRVLM._reader = easyocr.Reader(["en"], gpu=False)
        return EasyOCRVLM._reader

    def generate(self, image: Image.Image, prompt: str) -> str:
        import numpy as np
        lines = self._get_reader().readtext(np.asarray(image.convert("RGB")),
                                            detail=0, paragraph=True)
        return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Stage-level recognizers
# --------------------------------------------------------------------------- #
class VLMRecognizer:
    """Crop -> per-type prompt -> VLMClient. The reference Recognizer, block-level."""

    def __init__(self, vlm, pad: int = 6):
        self.vlm = vlm
        self.pad = pad

    @property
    def label(self) -> str:
        lab = getattr(type(self.vlm), "label", None)
        return lab.fget(self.vlm) if isinstance(lab, property) else getattr(self.vlm, "label", "vlm")

    def _crop(self, img: Image.Image, b: BBox) -> Image.Image:
        w, h = img.size
        box = (max(0, b.x0 - self.pad), max(0, b.y0 - self.pad),
               min(w, b.x1 + self.pad), min(h, b.y1 + self.pad))
        return img.crop(tuple(map(int, box)))

    def recognize_one(self, page_img: Image.Image, block: Block) -> str:
        if block.type in FURNITURE:
            return ""
        crop = self._crop(page_img, block.bbox)
        if block.type == BlockType.FIGURE:
            alt = self.vlm.generate(crop, PROMPTS[BlockType.FIGURE]).replace("\n", " ").strip()
            return f"![{alt or 'figure'}](figure_p{block.page}_{int(block.bbox.x0)}.png)"
        prompt = PROMPTS.get(block.type, PROMPTS[BlockType.TEXT])
        return self.vlm.generate(crop, prompt).strip()


class GroundTruthRecognizer:
    """
    Offline backend for the bundled sample: matches each detected block to the known
    region we drew (by IoU) and returns its real text — and corrects the block type so
    offline labels look right. Transparent demo aid, only used when no cloud key is set.
    """

    label = "offline sample (ground-truth text)"

    def __init__(self, regions: list[dict]):
        self.regions = regions

    def recognize_one(self, page_img: Image.Image, block: Block) -> str:
        best, best_iou = None, 0.0
        for r in self.regions:
            iou = block.bbox.iou(BBox(*r["bbox"]))
            if iou > best_iou:
                best, best_iou = r, iou
        if best and best_iou > 0.05:
            try:
                block.type = BlockType(best["type"])
            except ValueError:
                pass
            if block.type in FURNITURE:
                return ""
            return str(best.get("text", "")).strip()
        return ""
