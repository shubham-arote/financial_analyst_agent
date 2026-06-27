"""
obs.py — lightweight observability: a structured JSONL trace of every Q&A through the agent.

Production teams pipe this to Langfuse/LangSmith; here it's a self-contained, dependency-free
JSONL log (+ a /api/traces endpoint) so every run's retrieval candidates, grade verdicts,
rewrites, answer, latency, and any injection flags are inspectable and debuggable.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "logs" / "traces.jsonl"
_lock = threading.Lock()


def log_trace(record: dict) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with _lock, LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def recent(n: int = 20) -> list[dict]:
    if not LOG.exists():
        return []
    try:
        lines = LOG.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(ln) for ln in lines if ln.strip()]
    except Exception:
        return []
