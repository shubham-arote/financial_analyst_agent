"""
factory.py — assemble the right SRR pipeline tier from environment variables.

    SRR_DETECTOR     heuristic (default) | doclayout
    SRR_RECOGNIZER   auto (default) | cloud | groundtruth | easyocr | stub
    SRR_STREAM_DELAY seconds between structure events (default 0.12; UI pacing)
    SRR_MAX_WORKERS  recognition thread pool size (default 6)

"auto" picks the best available: cloud VLM if a key is set, else the bundled sample's
ground-truth text if provided, else a stub. Every choice falls back cleanly so the app
never hard-fails.
"""

from __future__ import annotations

import logging
import os

from . import cloud
from .core import ColumnAwareReadingOrder
from .detector import DocLayoutYOLODetector, HeuristicDetector
from .recognizer import (CloudVLM, EasyOCRVLM, GroundTruthRecognizer, StubVLM,
                         VLMRecognizer)
from .streaming import StreamingSRRPipeline

logger = logging.getLogger("srr.factory")


def _build_detector():
    choice = os.getenv("SRR_DETECTOR", "heuristic").lower()
    if choice == "doclayout":
        weights = os.getenv("SRR_DETECTOR_WEIGHTS", "doclayout_yolo_docstructbench.pt")
        try:
            return DocLayoutYOLODetector(weights)
        except Exception as e:
            logger.warning("DocLayoutYOLO unavailable (%s); using heuristic detector.", e)
    return HeuristicDetector()


def _build_recognizer(ground_truth):
    choice = os.getenv("SRR_RECOGNIZER", "auto").lower()

    if choice == "stub":
        return VLMRecognizer(StubVLM())
    if choice == "easyocr":
        return VLMRecognizer(EasyOCRVLM())
    if choice == "groundtruth":
        if ground_truth:
            return GroundTruthRecognizer(ground_truth)
        logger.warning("SRR_RECOGNIZER=groundtruth but no sidecar; falling back.")
    if choice in ("cloud", "vlm"):
        return VLMRecognizer(CloudVLM())

    # auto: best available
    if cloud.has_cloud():
        return VLMRecognizer(CloudVLM())
    if ground_truth:
        return GroundTruthRecognizer(ground_truth)
    return VLMRecognizer(StubVLM())


def build_pipeline(ground_truth=None, stream_delay: float | None = None) -> StreamingSRRPipeline:
    detector = _build_detector()
    recognizer = _build_recognizer(ground_truth)
    relation = ColumnAwareReadingOrder()

    delay = (float(os.getenv("SRR_STREAM_DELAY", "0.12"))
             if stream_delay is None else stream_delay)
    workers = int(os.getenv("SRR_MAX_WORKERS", "6"))
    return StreamingSRRPipeline(detector, recognizer, relation,
                                stream_delay=delay, max_workers=workers)
