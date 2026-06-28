"""Central configuration — the ONE place the app reads the environment.

Every tunable and secret lives here as an attribute of the `settings` singleton.
Modules import `from app.config import settings` and read `settings.<name>`; they
never touch `os.environ` directly. This keeps configuration discoverable, typed,
and easy to override in tests (set env vars before importing `app.*`).

`.env` is loaded once here (python-dotenv, optional). Existing process env wins
over `.env` (override=False), so test/CI/deploy env vars take precedence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional; the app still runs without it
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

_ROOT = Path(__file__).resolve().parents[1]   # repo root (app/ is parents[0])
_DATA = _ROOT / "data"


def _str(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _lower(name: str, default: str = "") -> str:
    return os.getenv(name, default).lower()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _secret(name: str) -> str | None:
    """A key/token: stripped of stray whitespace, empty -> None."""
    v = os.getenv(name)
    return v.strip() if v and v.strip() else None


@dataclass(frozen=True)
class Settings:
    # ── secrets / keys (None when unset; callers degrade gracefully) ──────────
    cohere_api_key: str | None
    groq_api_key: str | None
    openrouter_api_key: str | None
    vlm_api_key: str | None
    qdrant_api_key: str | None

    # ── models / endpoints ───────────────────────────────────────────────────
    cohere_embed_model: str
    cohere_rerank_model: str
    vision_model: str | None
    text_model: str | None
    vlm_base_url: str | None
    vlm_model: str

    # ── parsing pipeline ─────────────────────────────────────────────────────
    pdf_parser: str          # auto | docling | cloudvlm | textlayer
    vlm_chunk: int           # pages per cloud-VLM OCR chunk
    docling_ocr: str | None  # None=auto, "1"=always, "0"=never
    docling_chunk: int       # pages per Docling background chunk
    detector: str            # heuristic | doclayout
    detector_weights: str
    recognizer: str          # auto | cloud | groundtruth | easyocr | stub
    stream_delay: float      # UI pacing between structure events
    max_workers: int         # recognition thread-pool size
    tables: bool             # detect ruled tables (SRR_TABLES != "0")

    # ── retrieval ────────────────────────────────────────────────────────────
    rerank: str              # heuristic | llm | off
    vector_store: str        # memory | qdrant

    # ── qdrant (Gold backend) ────────────────────────────────────────────────
    qdrant_mode: str         # "" (local) | cloud
    qdrant_url: str | None
    qdrant_location: str     # ":memory:" or a local URL
    qdrant_collection: str

    # ── storage (DocStore) ───────────────────────────────────────────────────
    docstore: str            # "" (sqlite) | gcs
    gcs_bucket: str | None
    gcs_endpoint: str | None # emulator endpoint (fake-gcs-server), else None
    db_path: str             # sqlite DocStore path

    # ── conversation memory (checkpointer) ───────────────────────────────────
    checkpoint: str          # sqlite | postgres | memory | off
    checkpoint_db_url: str    # postgres / Cloud SQL URL
    checkpoint_db: str        # sqlite checkpointer path

    # convenience -------------------------------------------------------------
    @property
    def has_chat(self) -> bool:
        """True when some chat/vision LLM is reachable (synthesized answers + VLM OCR)."""
        return bool(self.groq_api_key or self.openrouter_api_key
                    or (self.vlm_base_url and self.vlm_api_key))

    @property
    def has_cohere(self) -> bool:
        return bool(self.cohere_api_key)


def _load() -> Settings:
    return Settings(
        cohere_api_key=_secret("COHERE_API_KEY"),
        groq_api_key=_secret("GROQ_API_KEY"),
        openrouter_api_key=_secret("OPENROUTER_API_KEY"),
        vlm_api_key=_secret("VLM_API_KEY"),
        qdrant_api_key=_secret("QDRANT_API_KEY"),

        cohere_embed_model=_str("COHERE_EMBED_MODEL", "embed-v4.0"),
        cohere_rerank_model=_str("COHERE_RERANK_MODEL", "rerank-v3.5"),
        vision_model=os.getenv("VISION_MODEL"),
        text_model=os.getenv("TEXT_MODEL"),
        vlm_base_url=os.getenv("VLM_BASE_URL"),
        vlm_model=_str("VLM_MODEL", "gpt-4o-mini"),

        pdf_parser=_lower("SRR_PDF_PARSER", "auto"),
        vlm_chunk=max(1, _int("SRR_VLM_CHUNK", 4)),
        docling_ocr=os.getenv("SRR_DOCLING_OCR"),
        docling_chunk=max(1, _int("SRR_DOCLING_CHUNK", 6)),
        detector=_lower("SRR_DETECTOR", "heuristic"),
        detector_weights=_str("SRR_DETECTOR_WEIGHTS", "doclayout_yolo_docstructbench.pt"),
        recognizer=_lower("SRR_RECOGNIZER", "auto"),
        stream_delay=_float("SRR_STREAM_DELAY", 0.12),
        max_workers=_int("SRR_MAX_WORKERS", 6),
        tables=_str("SRR_TABLES", "1") != "0",

        rerank=_lower("SRR_RERANK", "heuristic"),
        vector_store=_lower("SRR_VECTOR_STORE", "memory"),

        qdrant_mode=_lower("QDRANT_MODE", ""),
        qdrant_url=_secret("QDRANT_URL"),
        qdrant_location=_str("QDRANT_LOCATION", ":memory:"),
        qdrant_collection=_str("QDRANT_COLLECTION", "srr_docs"),

        docstore=_lower("SRR_DOCSTORE", ""),
        gcs_bucket=os.getenv("GCS_BUCKET"),
        gcs_endpoint=os.getenv("STORAGE_EMULATOR_HOST") or os.getenv("GCS_ENDPOINT"),
        db_path=_str("SRR_DB", str(_DATA / "docs.db")),

        checkpoint=_lower("SRR_CHECKPOINT", "sqlite"),
        checkpoint_db_url=_str("CHECKPOINT_DB_URL", ""),
        checkpoint_db=_str("SRR_CHECKPOINT_DB", str(_DATA / "checkpoints.db")),
    )


settings = _load()
