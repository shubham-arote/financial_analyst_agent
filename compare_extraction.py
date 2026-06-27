"""Compare OUR extraction (text-layer + Docling) vs the reference proprietary
extraction on the AI Index Report 2026, on matched front-matter pages."""
import sys, fitz
from collections import Counter
sys.path.insert(0, r"E:\PROJECTS\ocr pipeline")
from app.srr.pdf_textlayer import extract_page
from app.srr import pdf_docling

PDF = r"E:\PROJECTS\ocr pipeline\ai_index_report_2026.pdf"
data = open(PDF, "rb").read()
doc = fitz.open(PDF)

PAGES = {2: "Contents/ToC", 5: "Steering Committee (4-col photo grid)",
         7: "Partners (markdown table + logos)", 10: "Top Takeaways"}
# reference block counts + whether reference emitted a table on that page
REF = {2: (7, False), 5: (51, False), 7: (25, True), 10: (7, False)}


def report(name, getblocks):
    print(f"\n===== {name} =====")
    for p, lbl in PAGES.items():
        blocks = getblocks(p)
        hist = Counter(b.type.value for b in blocks)
        has_tbl = any(b.type.value == "table" for b in blocks)
        ref_n, ref_tbl = REF[p]
        print(f"p{p:>2} {lbl}")
        print(f"     ours: {len(blocks):>2} blocks  {dict(hist)}  table={has_tbl}")
        print(f"     ref : {ref_n:>2} blocks  table={ref_tbl}")


report("TEXT-LAYER (fast path)", lambda p: extract_page(doc[p - 1], p))

# Docling: born-digital -> do_ocr=False (use embedded text + layout model)
_dcache = {}
def _docling(p):
    if p not in _dcache:
        _dcache[p] = pdf_docling.parse_pages(data, p - 1, p, do_ocr=False).get(p - 1, [])
    return _dcache[p]
report("DOCLING (layout model)", _docling)

# Table fidelity on p7 — show what each path produced for the partners table
print("\n===== TABLE FIDELITY (p7) =====")
for nm, blocks in [("text-layer", extract_page(doc[6], 7)), ("docling", _docling(7))]:
    tbls = [b for b in blocks if b.type.value == "table"]
    print(f"  {nm}: {len(tbls)} table block(s)")
    for t in tbls[:1]:
        print("   ", (t.content or "")[:300].replace("\n", " "))
