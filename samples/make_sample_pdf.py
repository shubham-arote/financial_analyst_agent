"""
make_sample_pdf.py — a small multi-page, born-digital PDF that mimics an annual report,
for testing the text-layer ingestion path. Output: samples/sample_report.pdf

The financial statements are drawn as *ruled* tables (grid lines) so PyMuPDF's
conservative line-based `find_tables()` detects them as real tables (not loose text),
while headings and prose stay separate.
"""

from __future__ import annotations

from pathlib import Path

import fitz

OUT = Path(__file__).with_name("sample_report.pdf")
COLX = [72, 320, 410, 500]            # label | FY26 | FY25 column edges


def heading(page, y, text):
    page.insert_text((72, y), text, fontsize=22, fontname="hebo")   # large + bold


def paragraphs(page, y, paras):
    for p in paras:
        rect = fitz.Rect(72, y, 523, y + 140)
        ret = page.insert_textbox(rect, p, fontsize=11, fontname="helv")
        y += (rect.height - ret if ret > 0 else rect.height) + 16
    return y


def table(page, y, title, header, rows):
    page.insert_text((72, y), title, fontsize=13, fontname="hebo")
    top = y + 14
    rh = 22
    allrows = [header] + rows
    n = len(allrows)
    for r, row in enumerate(allrows):
        ry = top + r * rh
        font = "hebo" if (r == 0 or row[0] == "Net assets" or row[0].startswith("Profit after")) else "helv"
        for c, cell in enumerate(row):
            page.insert_text((COLX[c] + 6, ry + 15), cell, fontsize=10, fontname=font)
    bottom = top + n * rh
    for r in range(n + 1):                              # horizontal rules
        page.draw_line(fitz.Point(COLX[0], top + r * rh), fitz.Point(COLX[-1], top + r * rh),
                       color=(0, 0, 0), width=0.7)
    for cx in COLX:                                     # vertical rules
        page.draw_line(fitz.Point(cx, top), fitz.Point(cx, bottom), color=(0, 0, 0), width=0.7)
    return bottom + 16


def page_number(page, doc):
    page.insert_text((500, 805), f"Page {doc.page_count}", fontsize=9, fontname="helv")


def main():
    doc = fitz.open()

    p = doc.new_page(width=595, height=842); heading(p, 90, "ACME PLC")
    paragraphs(p, 130, ["Annual Report and Accounts", "January 2026"]); page_number(p, doc)

    p = doc.new_page(width=595, height=842); heading(p, 90, "Chairman's Statement")
    paragraphs(p, 130, [
        "I am pleased to report another year of progress for the Group. Trading conditions "
        "remained challenging, yet the business delivered resilient results, driven by "
        "disciplined cost control and continued investment in our online platform.",
        "The Board is recommending a final dividend of 75 pence per share, bringing the total "
        "dividend for the year to 140 pence, an increase of 6% on the prior year."])
    page_number(p, doc)

    p = doc.new_page(width=595, height=842); heading(p, 90, "Financial Review")
    paragraphs(p, 130, [
        "Group revenue increased by 5.9% to 6,303 million pounds, with full price sales up "
        "7.1%. Operating margin improved to 16.7%, reflecting operational leverage and lower "
        "markdown activity during the period.",
        "Profit before tax was 1,011 million pounds, up 8.4% on the prior year. Earnings per "
        "share rose to 712 pence and the Group generated strong free cash flow."])
    page_number(p, doc)

    p = doc.new_page(width=595, height=842); heading(p, 90, "Consolidated Income Statement")
    y = paragraphs(p, 130, ["The income statement for the 52 weeks to January 2026 is set out below."])
    table(p, y, "Income Statement (GBP m)", ["Segment", "FY26", "FY25"], [
        ["Revenue", "6,303", "5,952"], ["Cost of sales", "(4,001)", "(3,820)"],
        ["Gross profit", "2,302", "2,132"], ["Operating profit", "1,052", "985"],
        ["Profit before tax", "1,011", "933"], ["Profit after tax", "770", "712"]])
    page_number(p, doc)

    p = doc.new_page(width=595, height=842); heading(p, 90, "Consolidated Balance Sheet")
    y = paragraphs(p, 130, ["Net assets increased during the period as summarised below."])
    table(p, y, "Balance Sheet (GBP m)", ["Item", "FY26", "FY25"], [
        ["Non-current assets", "2,540", "2,410"], ["Current assets", "1,980", "1,850"],
        ["Total assets", "4,520", "4,260"], ["Total liabilities", "(3,090)", "(2,980)"],
        ["Net assets", "1,430", "1,280"]])
    page_number(p, doc)

    p = doc.new_page(width=595, height=842); heading(p, 90, "Notes to the Accounts")
    paragraphs(p, 130, [
        "Revenue is recognised when control of goods passes to the customer. The Group "
        "operates retail stores and an online platform across multiple territories.",
        "The effective tax rate for the year was 23.8%. Capital expenditure totalled 168 "
        "million pounds, primarily on warehouse automation and technology systems."])
    page_number(p, doc)

    doc.save(OUT)
    print(f"wrote {OUT}  ({doc.page_count} pages)")


if __name__ == "__main__":
    main()
