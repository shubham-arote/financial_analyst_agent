"""Verify GcsDocStore against fake-gcs-server: full DocStore round-trip + COALESCE semantics."""
import os
os.environ["SRR_DOCSTORE"] = "gcs"
os.environ["GCS_BUCKET"] = "srr-test"
os.environ["STORAGE_EMULATOR_HOST"] = "http://localhost:4443"

from app.store import get_store

s = get_store()
print("store:", type(s).__name__)

s.save_doc("d1", "pdf", {"status": "parsing", "page_count": 2, "parsed_pages": 0}, blob=b"%PDF-1.4 fake")
s.save_page("d1", 1, [{"type": "text", "content": "hello", "bbox": [0, 0, 10, 10]}])
s.save_page("d1", 2, [{"type": "title", "content": "world", "bbox": [0, 0, 10, 10]}])
s.update_status("d1", "ready", 2, None)

meta = s.get_meta("d1")
print("meta status/parsed:", meta.get("status"), meta.get("parsed_pages"), "| kind:", meta.get("kind"))
print("blob:", s.get_blob("d1"))
print("pages:", list(s.get_pages("d1").keys()), "->", s.get_pages("d1")[1])
print("all_ids:", s.all_ids(), "| count:", s.count())

# COALESCE: a meta-only re-save must NOT wipe the stored blob
s.save_doc("d1", "pdf", {"status": "ready", "page_count": 2, "parsed_pages": 2})
print("blob after meta-only save (COALESCE):", s.get_blob("d1"))

assert s.get_blob("d1") == b"%PDF-1.4 fake"
assert list(s.get_pages("d1").keys()) == [1, 2]
assert s.all_ids() == ["d1"] and s.count() == 1
print("\nGCS DOCSTORE OK")
