"""
streaming.py — the SRR orchestrator that *emits events as it runs*.

parse_page_streaming() is a generator yielding JSON-able events in pipeline order:
structure (one per detected block) -> recognition (one per block, as each finishes) ->
relation (reading order) -> assembled result. The FastAPI WebSocket relays these to the
browser canvas, which is what makes the "layout detection side window" feel live.

Recognition runs in a thread pool and emits via as_completed, so boxes fill in as each
block's VLM/OCR call returns — the natural staggering *is* the animation.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

from PIL import Image

from .core import Block, assemble_markdown

logger = logging.getLogger("srr.streaming")


class StreamingSRRPipeline:
    def __init__(self, detector, recognizer, relation,
                 stream_delay: float = 0.12, max_workers: int = 6):
        self.detector = detector
        self.recognizer = recognizer
        self.relation = relation
        self.stream_delay = stream_delay
        self.max_workers = max_workers
        self.last_ordered_blocks: list[Block] = []

    @property
    def info(self) -> dict:
        return {
            "detector": getattr(self.detector, "label", type(self.detector).__name__),
            "recognizer": getattr(self.recognizer, "label", type(self.recognizer).__name__),
            "relation": getattr(self.relation, "label", type(self.relation).__name__),
        }

    # ---- streaming (drives the UI) ---- #
    def parse_page_streaming(self, image: Image.Image, page: int = 0) -> Iterator[dict]:
        yield {"type": "info", **self.info, "page_w": image.size[0], "page_h": image.size[1]}

        # STRUCTURE
        yield {"type": "stage", "stage": "structure", "status": "start"}
        blocks = self.detector.detect(image, page)
        for i, b in enumerate(blocks):
            b.id = i
            yield {"type": "block", "id": b.id, "block_type": b.type.value,
                   "bbox": [int(b.bbox.x0), int(b.bbox.y0), int(b.bbox.x1), int(b.bbox.y1)],
                   "score": round(float(b.score), 3)}
            if self.stream_delay:
                time.sleep(self.stream_delay)
        yield {"type": "stage", "stage": "structure", "status": "end", "count": len(blocks)}

        # RECOGNITION (parallel; emit as each finishes)
        yield {"type": "stage", "stage": "recognition", "status": "start"}
        if blocks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futs = {pool.submit(self.recognizer.recognize_one, image, b): b for b in blocks}
                for fut in as_completed(futs):
                    b = futs[fut]
                    try:
                        b.content = fut.result()
                    except Exception as e:  # keep the stream alive
                        b.content = f"[recognition-error: {e}]"
                    yield {"type": "recognized", "id": b.id,
                           "block_type": b.type.value, "content": b.content or ""}
        yield {"type": "stage", "stage": "recognition", "status": "end"}

        # RELATION
        yield {"type": "stage", "stage": "relation", "status": "start"}
        ordered = self.relation.order(blocks, image.size)
        self.last_ordered_blocks = ordered   # exposed for block-aware indexing
        yield {"type": "order", "order": [b.id for b in ordered]}
        yield {"type": "stage", "stage": "relation", "status": "end"}

        # ASSEMBLY
        markdown = assemble_markdown(ordered)
        yield {"type": "result", "markdown": markdown,
               "page_w": image.size[0], "page_h": image.size[1]}
        yield {"type": "done"}

    # ---- headless (agent / tests) ---- #
    def parse_page(self, image: Image.Image, page: int = 0) -> tuple[list[Block], str]:
        blocks = self.detector.detect(image, page)
        for i, b in enumerate(blocks):
            b.id = i
        if blocks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for b, res in zip(blocks, pool.map(
                        lambda bb: self.recognizer.recognize_one(image, bb), blocks)):
                    b.content = res
        ordered = self.relation.order(blocks, image.size)
        return ordered, assemble_markdown(ordered)


# --------------------------------------------------------------------------- #
# CLI: python -m app.srr.streaming <image_or_pdf> [--no-delay]
# --------------------------------------------------------------------------- #
def _load_ground_truth(path: str):
    import json
    from pathlib import Path
    side = Path(path).with_suffix(".regions.json")
    if side.exists():
        return json.loads(side.read_text(encoding="utf-8"))
    return None


if __name__ == "__main__":
    import sys
    from .factory import build_pipeline

    logging.basicConfig(level=logging.INFO)
    src = sys.argv[1] if len(sys.argv) > 1 else "samples/report_page.png"
    delay = 0.0 if "--no-delay" in sys.argv else None

    gt = _load_ground_truth(src)
    pipe = build_pipeline(ground_truth=gt, stream_delay=delay)
    print("pipeline:", pipe.info)

    img = Image.open(src).convert("RGB")
    n_blocks = 0
    for ev in pipe.parse_page_streaming(img):
        t = ev["type"]
        if t == "block":
            n_blocks += 1
            print(f"  [structure] #{ev['id']:>2} {ev['block_type']:<8} {ev['bbox']}")
        elif t == "recognized":
            preview = (ev["content"][:70] + "…") if len(ev["content"]) > 70 else ev["content"]
            print(f"  [recognize] #{ev['id']:>2} {ev['block_type']:<8} {preview!r}")
        elif t == "order":
            print(f"  [relation ] reading order: {ev['order']}")
        elif t == "result":
            print("\n---- ASSEMBLED MARKDOWN ----\n")
            print(ev["markdown"])

    assert n_blocks > 0, "FAIL: detector produced no blocks"
    print(f"\nOK: {n_blocks} blocks parsed end-to-end.")
