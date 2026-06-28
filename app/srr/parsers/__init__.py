"""PDF parsers, auto-routed by config (SRR_PDF_PARSER):

  * textlayer — born-digital fast path (PyMuPDF text layer); light, always available
  * docling   — layout-model parser (optional, heavy: pulls torch/Docling)
  * cloudvlm  — scanned-page OCR via a cloud vision model (needs a cloud key)

Heavy/optional parsers are imported lazily by callers, so importing this package never
pulls torch/Docling. Import the submodule you need directly, e.g.:

    from app.srr.parsers import textlayer
"""
