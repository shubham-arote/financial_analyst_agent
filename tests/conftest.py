"""Shared pytest setup.

Two jobs, both done at import time BEFORE the app is imported anywhere:

1. Put the repo root on sys.path so `import app...` works no matter where pytest
   is invoked from.
2. Isolate tests from the developer's real state — point the DocStore and the
   conversation checkpointer at throwaway locations so running the suite never
   touches `data/docs.db` / `data/checkpoints.db`, and startup stays instant.
"""

import os
import sys
import tempfile
from pathlib import Path

# 1. repo root on the import path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 2. test isolation — must be set before `app.*` modules read the environment
_TMP = Path(tempfile.gettempdir()) / "srr_tests"
_TMP.mkdir(exist_ok=True)
os.environ.setdefault("SRR_DOCSTORE", "sqlite")
os.environ.setdefault("SRR_DB", str(_TMP / "docs.db"))
os.environ.setdefault("SRR_CHECKPOINT", "memory")     # no checkpoint file on disk
