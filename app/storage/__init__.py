"""Storage layer — the DocStore seam for persisted documents (meta + blob + per-page blocks).

`get_store()` selects `SqliteDocStore` (local default) or `GcsDocStore` (stateless cloud)
by config. The rest of the app depends on the `DocStore` ABC, not a concrete backend.
"""

from .docstore import DocStore, SqliteDocStore, GcsDocStore, get_store

__all__ = ["DocStore", "SqliteDocStore", "GcsDocStore", "get_store"]
