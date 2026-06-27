"""
make_sample.py — generate a synthetic multi-column financial report page.

Produces:
  samples/report_page.png            the page image (input to the SRR pipeline)
  samples/report_page.regions.json   ground-truth regions (type + bbox + text)

The sidecar lets the offline GroundTruthRecognizer return real text per block, so the
end-to-end Q&A demo works with no API key. Layout uses generous whitespace gaps (paragraph
spacing, a 60px column gutter) so the heuristic XY-cut detector can segment it cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1000, 1400
MARGIN = 60
INK = (17, 17, 17)
GRAY = (210, 210, 210)
BAR = (90, 90, 90)

OUT_PNG = Path(__file__).with_name("report_page.png")
OUT_JSON = Path(__file__).with_name("report_page.regions.json")

regions: list[dict] = []


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for path in (f"C:/Windows/Fonts/{name}", name):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font("arialbd.ttf", 34)
F_H2 = font("arialbd.ttf", 21)
F_BODY = font("arial.ttf", 16)
F_TBL = font("arial.ttf", 15)
F_TBLB = font("arialbd.ttf", 15)
F_FORMULA = font("ariali.ttf", 19)
F_SMALL = font("arial.ttf", 12)


def record(rtype: str, bbox: tuple[int, int, int, int], text: str) -> None:
    regions.append({"type": rtype, "bbox": [int(v) for v in bbox], "text": text})


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def paragraph(draw, x, y, text, fnt, max_w, rtype, line_gap=6) -> int:
    """Draw wrapped text; record a region; return the bottom y."""
    lines = wrap(draw, text, fnt, max_w)
    asc, desc = fnt.getmetrics()
    lh = asc + desc + line_gap
    widest = 0
    for i, ln in enumerate(lines):
        draw.text((x, y + i * lh), ln, font=fnt, fill=INK)
        widest = max(widest, int(draw.textlength(ln, font=fnt)))
    bottom = y + len(lines) * lh - line_gap
    record(rtype, (x, y, x + widest, bottom), text)
    return bottom


def draw_table(draw, x, y, w) -> int:
    rows = [["Segment", "Revenue ($M)", "YoY %"],
            ["Cloud", "29.5", "+24%"],
            ["Hardware", "12.1", "+5%"],
            ["Services", "6.6", "+11%"],
            ["Total", "48.2", "+18%"]]
    rh = 30
    col_x = [x, x + int(w * 0.46), x + int(w * 0.76), x + w]
    table_h = rh * len(rows)
    # horizontal rules (these give the detector >=2 long ink rows -> TABLE)
    for i in range(len(rows) + 1):
        yy = y + i * rh
        draw.line([(x, yy), (x + w, yy)], fill=INK, width=1)
    for cx in col_x:
        draw.line([(cx, y), (cx, y + table_h)], fill=INK, width=1)
    for r, row in enumerate(rows):
        fnt = F_TBLB if r == 0 or row[0] == "Total" else F_TBL
        for c, cell in enumerate(row):
            draw.text((col_x[c] + 8, y + r * rh + 7), cell, font=fnt, fill=INK)
    md = ("| Segment | Revenue ($M) | YoY % |\n| --- | --- | --- |\n"
          "| Cloud | 29.5 | +24% |\n| Hardware | 12.1 | +5% |\n"
          "| Services | 6.6 | +11% |\n| Total | 48.2 | +18% |")
    record("table", (x, y, x + w, y + table_h), md)
    return y + table_h


def draw_figure(draw, x, y, w) -> int:
    h = 200
    bars = [("Q3'25", 40.8), ("Q4'25", 43.1), ("Q1'26", 45.6), ("Q2'26", 48.2)]
    base_y = y + h - 28
    draw.line([(x + 30, y + 6), (x + 30, base_y)], fill=INK, width=6)   # y axis
    # thick x axis bridges the gaps between bars so XY-cut keeps the figure as ONE block
    draw.line([(x + 30, base_y), (x + w - 6, base_y)], fill=INK, width=8)  # x axis
    span = w - 60
    bw = int(span / len(bars) * 0.55)
    maxv = 52
    for i, (lab, val) in enumerate(bars):
        bx = x + 40 + int(span / len(bars) * i) + 6
        bh = int((val / maxv) * (h - 50))
        draw.rectangle([bx, base_y - bh, bx + bw, base_y], fill=BAR)
        draw.text((bx, base_y + 6), lab, font=F_SMALL, fill=INK)
    record("figure", (x, y, x + w, y + h),
           "Bar chart of quarterly revenue: Q3 2025 $40.8M, Q4 2025 $43.1M, "
           "Q1 2026 $45.6M, Q2 2026 $48.2M.")
    return y + h


def main() -> None:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # ---- Title ----
    title = "Acme Corp - Q2 2026 Financial Report"
    d.text((MARGIN, 46), title, font=F_TITLE, fill=INK)
    record("title", (MARGIN, 46, MARGIN + int(d.textlength(title, font=F_TITLE)), 86), title)
    d.line([(MARGIN, 100), (W - MARGIN, 100)], fill=GRAY, width=2)  # faint, not detected

    # ---- Left column ----
    lx, lw = MARGIN, 410
    y = 132
    d.text((lx, y), "Financial Highlights", font=F_H2, fill=INK)
    record("title", (lx, y, lx + int(d.textlength("Financial Highlights", font=F_H2)), y + 24),
           "Financial Highlights")
    y += 44
    paras = [
        "Total revenue for the second quarter of 2026 was $48.2 million, an increase "
        "of 18% compared with the same quarter last year.",
        "By segment, Cloud contributed $29.5 million, Hardware $12.1 million, and "
        "Services $6.6 million to total quarterly revenue.",
        "Operating margin expanded to 22.4% in the quarter, up from 19.1% in the second "
        "quarter of 2025, driven by higher-margin cloud revenue.",
        "Net income was $7.3 million, or $0.41 per diluted share, compared with $5.0 "
        "million, or $0.28 per share, a year earlier.",
    ]
    for p in paras:
        y = paragraph(d, lx, y, p, F_BODY, lw, "text") + 28

    # ---- Right column ----
    rx, rw = 530, 410
    ry = 132
    ry = paragraph(d, rx, ry, "During the quarter the company repurchased 1.2 million "
                   "shares for a total of $14 million under its existing buyback program.",
                   F_BODY, rw, "text") + 30
    ry = draw_table(d, rx, ry, rw) + 34
    ry = draw_figure(d, rx, ry, rw) + 10
    cap = "Figure 1. Quarterly revenue trend, Q3 2025 to Q2 2026."
    d.text((rx, ry), cap, font=F_SMALL, fill=INK)
    record("caption", (rx, ry, rx + int(d.textlength(cap, font=F_SMALL)), ry + 16), cap)

    # ---- Outlook (full width, below both columns) ----
    oy = max(y, ry) + 40
    d.text((MARGIN, oy), "Outlook", font=F_H2, fill=INK)
    record("title", (MARGIN, oy, MARGIN + int(d.textlength("Outlook", font=F_H2)), oy + 24), "Outlook")
    oy += 44
    full_w = W - 2 * MARGIN
    oy = paragraph(d, MARGIN, oy,
                   "Management raised full-year 2026 revenue guidance to a range of $196 "
                   "million to $202 million, reflecting continued momentum in the Cloud segment.",
                   F_BODY, full_w, "text") + 24
    oy = paragraph(d, MARGIN, oy,
                   "Cash and equivalents stood at $61.8 million at quarter end, and research "
                   "and development expense rose to $8.9 million.",
                   F_BODY, full_w, "text") + 40

    formula = "Gross Margin = (Revenue - COGS) / Revenue x 100%"
    d.text((MARGIN, oy), formula, font=F_FORMULA, fill=INK)
    record("formula", (MARGIN, oy, MARGIN + int(d.textlength(formula, font=F_FORMULA)), oy + 26), formula)

    # ---- Footer ----
    footer = "Acme Corporation - Confidential   |   Page 1 of 1"
    fy = H - 38
    d.text((MARGIN, fy), footer, font=F_SMALL, fill=INK)
    record("footer", (MARGIN, fy, MARGIN + int(d.textlength(footer, font=F_SMALL)), fy + 16), footer)

    img.save(OUT_PNG)
    OUT_JSON.write_text(json.dumps(regions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT_PNG}  ({W}x{H})")
    print(f"wrote {OUT_JSON}  ({len(regions)} regions: "
          f"{', '.join(sorted({r['type'] for r in regions}))})")


if __name__ == "__main__":
    main()
