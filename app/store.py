"""
store.py — durable storage for parsed documents (DocStore interface + SQLite backend).

The in-memory `DOCS` dict is lost on restart and can't be shared across instances. DocStore
externalizes the *durable* representation — the original PDF/image bytes, per-page block JSON,
and parse status — while live objects (the open fitz Document, the BM25 DocIndex) are
reconstructed from it on demand. The interface is backend-agnostic, so the SQLite backend can
be swapped for GCS (blobs) + Firestore (metadata/pages) on Cloud Run without touching server.py.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .config import settings


class DocStore(ABC):
    @abstractmethod
    def save_doc(self, doc_id: str, kind: str, meta: dict, blob: bytes | None = None) -> None: ...
    @abstractmethod
    def save_page(self, doc_id: str, page_no: int, blocks: list[dict]) -> None: ...
    @abstractmethod
    def update_status(self, doc_id: str, status: str, parsed_pages: int, error: str | None) -> None: ...
    @abstractmethod
    def get_meta(self, doc_id: str) -> dict | None: ...
    @abstractmethod
    def get_blob(self, doc_id: str) -> bytes | None: ...
    @abstractmethod
    def get_pages(self, doc_id: str) -> dict[int, list[dict]]: ...
    @abstractmethod
    def all_ids(self) -> list[str]: ...
    @abstractmethod
    def count(self) -> int: ...


class SqliteDocStore(DocStore):
    """File-based store. WAL + a write lock keeps the background parse thread and request
    threads from stepping on each other."""

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS docs("
                      "doc_id TEXT PRIMARY KEY, kind TEXT, meta TEXT, blob BLOB, updated REAL)")
            c.execute("CREATE TABLE IF NOT EXISTS pages("
                      "doc_id TEXT, page_no INTEGER, blocks TEXT, PRIMARY KEY(doc_id, page_no))")

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def save_doc(self, doc_id, kind, meta, blob=None):
        meta = {**meta, "kind": kind}
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO docs(doc_id, kind, meta, blob, updated) VALUES(?,?,?,?,?) "
                      "ON CONFLICT(doc_id) DO UPDATE SET kind=excluded.kind, meta=excluded.meta, "
                      "blob=COALESCE(excluded.blob, docs.blob), updated=excluded.updated",
                      (doc_id, kind, json.dumps(meta), blob, time.time()))

    def save_page(self, doc_id, page_no, blocks):
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO pages(doc_id, page_no, blocks) VALUES(?,?,?) "
                      "ON CONFLICT(doc_id, page_no) DO UPDATE SET blocks=excluded.blocks",
                      (doc_id, page_no, json.dumps(blocks)))

    def update_status(self, doc_id, status, parsed_pages, error):
        with self._lock, self._conn() as c:
            row = c.execute("SELECT meta FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
            if not row:
                return
            meta = json.loads(row[0])
            meta.update(status=status, parsed_pages=parsed_pages, error=error)
            c.execute("UPDATE docs SET meta=?, updated=? WHERE doc_id=?",
                      (json.dumps(meta), time.time(), doc_id))

    def get_meta(self, doc_id):
        with self._conn() as c:
            row = c.execute("SELECT meta FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def get_blob(self, doc_id):
        with self._conn() as c:
            row = c.execute("SELECT blob FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
        return row[0] if row else None

    def get_pages(self, doc_id):
        with self._conn() as c:
            rows = c.execute("SELECT page_no, blocks FROM pages WHERE doc_id=? ORDER BY page_no",
                             (doc_id,)).fetchall()
        return {r[0]: json.loads(r[1]) for r in rows}

    def all_ids(self):
        with self._conn() as c:
            return [r[0] for r in c.execute("SELECT doc_id FROM docs ORDER BY updated").fetchall()]

    def count(self):
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM docs").fetchone()[0]


class GcsDocStore(DocStore):
    """Pure-GCS backend — the Cloud-Run *stateless* store. One bucket, objects per doc:
    `{prefix}/{doc_id}/meta.json`, `/blob`, `/pages/{n}.json`. No DB, nothing on the instance
    disk, so the service can scale to zero and run many instances. Verifiable locally against
    fake-gcs-server by pointing `STORAGE_EMULATOR_HOST` (or `GCS_ENDPOINT`) at it."""

    def __init__(self, bucket: str, prefix: str = "docs", endpoint: str | None = None):
        from google.cloud import storage
        if endpoint:                                      # emulator / fake-gcs-server
            from google.auth.credentials import AnonymousCredentials
            self._client = storage.Client(project="srr-local",
                                          credentials=AnonymousCredentials(),
                                          client_options={"api_endpoint": endpoint})
        else:
            self._client = storage.Client()               # real GCS (ADC on Cloud Run)
        self.prefix = prefix.strip("/")
        self._bucket = self._client.bucket(bucket)
        if not self._bucket.exists():
            try:
                self._bucket = self._client.create_bucket(bucket)
            except Exception:
                pass

    def _b(self, *parts):
        return self._bucket.blob("/".join([self.prefix, *parts]))

    def save_doc(self, doc_id, kind, meta, blob=None):
        meta = {**meta, "kind": kind}
        self._b(doc_id, "meta.json").upload_from_string(json.dumps(meta),
                                                        content_type="application/json")
        if blob is not None:                              # COALESCE: only (re)write blob when given
            self._b(doc_id, "blob").upload_from_string(blob)

    def save_page(self, doc_id, page_no, blocks):
        self._b(doc_id, "pages", f"{page_no}.json").upload_from_string(
            json.dumps(blocks), content_type="application/json")

    def update_status(self, doc_id, status, parsed_pages, error):
        meta = self.get_meta(doc_id)
        if not meta:
            return
        meta.update(status=status, parsed_pages=parsed_pages, error=error)
        self._b(doc_id, "meta.json").upload_from_string(json.dumps(meta),
                                                        content_type="application/json")

    def get_meta(self, doc_id):
        b = self._b(doc_id, "meta.json")
        return json.loads(b.download_as_bytes()) if b.exists() else None

    def get_blob(self, doc_id):
        b = self._b(doc_id, "blob")
        return b.download_as_bytes() if b.exists() else None

    def get_pages(self, doc_id):
        out = {}
        base = "/".join([self.prefix, doc_id, "pages"]) + "/"
        for b in self._client.list_blobs(self._bucket, prefix=base):
            out[int(b.name.rsplit("/", 1)[-1][:-len(".json")])] = json.loads(b.download_as_bytes())
        return dict(sorted(out.items()))

    def all_ids(self):
        base = self.prefix + "/"
        ids = {b.name[len(base):].split("/", 1)[0]
               for b in self._client.list_blobs(self._bucket, prefix=base)}
        return sorted(i for i in ids if i)

    def count(self):
        return len(self.all_ids())


def get_store() -> DocStore:
    """Pick the durable store. `SRR_DOCSTORE=gcs` (or `GCS_BUCKET` set) -> GcsDocStore (the
    stateless Cloud-Run backend); else SQLite (default, local). `SRR_DB` overrides the sqlite path."""
    if settings.docstore == "gcs" or settings.gcs_bucket:
        return GcsDocStore(settings.gcs_bucket or "srr-docs", endpoint=settings.gcs_endpoint)
    return SqliteDocStore(settings.db_path)
