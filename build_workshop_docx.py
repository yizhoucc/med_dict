#!/usr/bin/env python3
"""Build the physician-first Word review manuscript from its locked JSON source."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "manuscript/physician_text/2026-11-21/WORKSHOP_PAPER_WORD_SOURCE.json"
OUTPUT = ROOT / "WORKSHOP_PAPER_PHYSICIAN_REVIEW.docx"
SKILL_ROOT = Path(
    "/Users/yizhoucc/.codex/plugins/cache/openai-primary-runtime/"
    "documents/26.1119.11507/skills/documents"
)

FIGURE_DIR = ROOT / "results/extraction_comparison/clinician_ratings/draft_figures"
FIGURES = {
    "[Insert Figure 1 here]": (
        FIGURE_DIR / "figure1_development_pathway_print.png",
        "Figure 1 development and adaptation workflow with four panels.",
    ),
    "[Insert Figure 2 here]": (
        FIGURE_DIR / "figure2_evaluator_distribution_print.png",
        "Figure 2 physician preference distributions overall and by cancer type.",
    ),
    "[Insert Figure 3 here]": (
        FIGURE_DIR / "figure3_interrater_matrix_print.png",
        "Figure 3 pairwise exact agreement and Cohen kappa among four physicians.",
    ),
    "[Insert Figure 4 here]": (
        FIGURE_DIR / "figure4_effect_profile_print.png",
        "Figure 4 physician preference by core oncology field.",
    ),
    "[Insert Supplementary Figure S1 here]": (
        FIGURE_DIR / "supplementary_figure_s1_note_margins_print.png",
        "Supplementary Figure S1 note level preference margins by cancer type.",
    ),
    "[Insert Supplementary Figure S2 here]": (
        FIGURE_DIR / "supplementary_figure_s2_staged_ablation_print.png",
        "Supplementary Figure S2 cumulative ablation configurations and adjacent comparisons.",
    ),
}

FONT = "Times New Roman"
INK = "000000"
NOTE_GRAY = RGBColor(89, 99, 110)
HEADER_FILL = "DCE6F1"
ALT_FILL = "F7FAFC"
BORDER = "D9D9D9"


def _set_run_font(run, size: float | None = None, *, bold: bool | None = None,
                  italic: bool | None = None, color: RGBColor | None = None) -> None:
    run.font.name = FONT
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), FONT)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), FONT)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def _set_paragraph_spacing(paragraph, *, before: float = 0, after: float = 6,
                           line: float = 1.15) -> None:
    fmt = paragraph.paragraph_format
    fmt.space_before = Pt(before)
    fmt.space_after = Pt(after)
    fmt.line_spacing = line
    fmt.widow_control = True


def _set_cell_border(cell, color: str = BORDER, size: str = "4") -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def _set_cell_fill(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def _set_cell_margins(cell, top: int = 80, start: int = 90,
                      bottom: int = 80, end: int = 90) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def _repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    header = OxmlElement("w:tblHeader")
    header.set(qn("w:val"), "true")
    tr_pr.append(header)


def _prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = tr_pr.find(qn("w:cantSplit"))
    if cant_split is None:
        cant_split = OxmlElement("w:cantSplit")
        tr_pr.append(cant_split)
    cant_split.set(qn("w:val"), "true")


def _set_repeat_table_header(row) -> None:
    _repeat_header(row)


def _set_col_width(cell, width_in: float) -> None:
    cell.width = Inches(width_in)
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(int(width_in * 1440)))
    tc_w.set(qn("w:type"), "dxa")


def _set_table_borders(table) -> None:
    for row in table.rows:
        for cell in row.cells:
            _set_cell_border(cell)


def _set_alt_text(inline_shape, title: str, description: str) -> None:
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("title", title)
    doc_pr.set("descr", description)


def _page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    _set_run_font(run, 9, color=RGBColor(90, 90, 90))
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    run._r.addnext(fld)


def _add_hyperlink(paragraph, text: str, url: str) -> None:
    part = paragraph.part
    rel_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), rel_id)
    run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "2F6690")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), FONT)
    fonts.set(qn("w:hAnsi"), FONT)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "19")
    r_pr.extend([fonts, color, underline, size])
    run.append(r_pr)
    node = OxmlElement("w:t")
    node.text = text
    run.append(node)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _reference_link_label(url: str) -> str:
    if url.startswith("https://doi.org/"):
        return f"doi: {url.removeprefix('https://doi.org/')}"
    arxiv_match = re.match(r"https?://arxiv\.org/(?:abs|pdf)/([^/?#]+)", url)
    if arxiv_match:
        return f"arXiv: {arxiv_match.group(1).removesuffix('.pdf')}"
    pubmed_match = re.match(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?", url)
    if pubmed_match:
        return f"PMID: {pubmed_match.group(1)}"
    host = urlparse(url).netloc.removeprefix("www.")
    return f"Source: {host}" if host else "Source link"


def _configure_styles(doc: Document) -> None:
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), FONT)
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), FONT)
    normal.font.size = Pt(11)
    normal.font.color.rgb = RGBColor(0, 0, 0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.15

    title = styles["Title"]
    title.font.name = FONT
    title.font.size = Pt(16)
    title.font.bold = False
    title.font.color.rgb = RGBColor(0, 0, 0)
    title.paragraph_format.space_before = Pt(0)
    title.paragraph_format.space_after = Pt(12)
    title_p_pr = title._element.get_or_add_pPr()
    title_border = title_p_pr.find(qn("w:pBdr"))
    if title_border is not None:
        title_p_pr.remove(title_border)

    for style_name, size in (("Heading 1", 14), ("Heading 2", 12)):
        style = styles[style_name]
        style.font.name = FONT
        style._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), FONT)
        style._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True
        style.paragraph_format.space_before = Pt(12 if style_name == "Heading 1" else 8)
        style.paragraph_format.space_after = Pt(4)

    if "Editorial Note" not in [s.name for s in styles]:
        note = styles.add_style("Editorial Note", WD_STYLE_TYPE.PARAGRAPH)
    else:
        note = styles["Editorial Note"]
    note.font.name = FONT
    note._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), FONT)
    note._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), FONT)
    note.font.size = Pt(10)
    note.font.italic = True
    note.font.color.rgb = NOTE_GRAY
    note.paragraph_format.space_before = Pt(0)
    note.paragraph_format.space_after = Pt(6)
    note.paragraph_format.line_spacing = 1.1
    note.paragraph_format.keep_together = True

    caption = styles["Caption"]
    caption.font.name = FONT
    caption._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:ascii"), FONT)
    caption._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:hAnsi"), FONT)
    caption.font.size = Pt(9.5)
    caption.font.bold = False
    caption.font.italic = False
    caption.font.color.rgb = RGBColor(0, 0, 0)
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.line_spacing = 1.05
    caption.paragraph_format.keep_together = True


def _configure_sections(doc: Document) -> None:
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)
        section.page_width = Inches(8.5)
        section.page_height = Inches(11)
        section.header_distance = Inches(0.5)
        section.footer_distance = Inches(0.5)
        footer_p = section.footer.paragraphs[0]
        _page_number(footer_p)


TABLE_WIDTHS = {
    "Table 1.": [1.35, 2.35, 2.80],
    "Table 2.": [0.50, 6.00],
    "Table 3.": [0.65, 1.15, 0.65, 0.65, 0.65, 2.75],
    "Table 4.": [0.65, 1.15, 0.65, 0.65, 0.65, 2.75],
    "Table 5.": [0.68, 0.70, 0.55, 0.55, 0.55, 1.45, 1.40, 0.62],
    "Supplementary Table S1.": [1.30, 1.10, 1.55, 2.55],
    "Supplementary Table S2.": [0.82, 1.90, 0.82, 1.05, 0.82],
    "Supplementary Table S3.": [0.95, 0.72, 0.50, 1.55, 1.50, 0.82],
}


def _add_table(doc: Document, block: dict) -> None:
    caption = doc.add_paragraph(style="Caption")
    caption.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = caption.add_run(block["caption"])
    _set_run_font(run, 9.5, bold=True)
    caption.paragraph_format.keep_with_next = True
    if block["caption"].startswith("Table 2."):
        caption.paragraph_format.page_break_before = True

    source_rows = block["rows"]
    is_table_2 = block["caption"].startswith("Table 2.")
    rows = (["Configuration", "Description"], *source_rows) if is_table_2 else source_rows
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    widths = next((v for k, v in TABLE_WIDTHS.items() if block["caption"].startswith(k)), None)
    if widths is None:
        widths = [6.5 / len(rows[0])] * len(rows[0])
    has_header = True
    _set_repeat_table_header(table.rows[0])

    dense = len(rows[0]) >= 6
    table_font = 8.2 if dense else 9.0
    for r_idx, row_data in enumerate(rows):
        _prevent_row_split(table.rows[r_idx])
        for c_idx, value in enumerate(row_data):
            cell = table.cell(r_idx, c_idx)
            cell.text = ""
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            _set_col_width(cell, widths[c_idx])
            _set_cell_margins(cell, top=70, start=75, bottom=70, end=75)
            if has_header and r_idx == 0:
                _set_cell_fill(cell, HEADER_FILL)
            elif r_idx % 2 == 0:
                _set_cell_fill(cell, ALT_FILL)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 or len(value) > 30 else WD_ALIGN_PARAGRAPH.CENTER
            _set_paragraph_spacing(p, after=0, line=1.0)
            run = p.add_run(value)
            _set_run_font(run, table_font, bold=(has_header and r_idx == 0))
    _set_table_borders(table)
    after = doc.add_paragraph()
    _set_paragraph_spacing(after, after=2, line=1.0)


def _add_figure(doc: Document, placeholder: str, caption_text: str) -> None:
    image_path, alt_text = FIGURES[placeholder]
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    inline_shape = run.add_picture(str(image_path), width=Inches(6.35))
    title = placeholder.strip("[]").replace("Insert ", "")
    _set_alt_text(inline_shape, title, alt_text)
    caption = doc.add_paragraph(style="Caption")
    caption.alignment = WD_ALIGN_PARAGRAPH.LEFT
    caption.add_run(caption_text)
    caption.paragraph_format.keep_together = True


def _add_reference(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    fmt = p.paragraph_format
    fmt.left_indent = Inches(0.25)
    fmt.first_line_indent = Inches(-0.25)
    fmt.space_after = Pt(4)
    fmt.line_spacing = 1.0
    match = re.search(r"(https?://\S+)$", text)
    if match:
        prefix = text[: match.start()].rstrip() + " "
        run = p.add_run(prefix)
        _set_run_font(run, 9.5)
        _add_hyperlink(p, _reference_link_label(match.group(1)), match.group(1))
    else:
        run = p.add_run(text)
        _set_run_font(run, 9.5)


def _normalize_docx_package(path: Path) -> None:
    """Rewrite the OOXML ZIP with fixed entry order and timestamps."""
    with zipfile.ZipFile(path, "r") as source_zip:
        entries = [(item, source_zip.read(item.filename)) for item in source_zip.infolist()]
    normalized = path.with_suffix(".normalized.docx")
    with zipfile.ZipFile(normalized, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target_zip:
        for item, payload in sorted(entries, key=lambda pair: pair[0].filename):
            info = zipfile.ZipInfo(item.filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = item.external_attr
            info.create_system = item.create_system
            info.flag_bits = item.flag_bits
            target_zip.writestr(info, payload)
    normalized.replace(path)


def _add_text_paragraph(doc: Document, text: str, kind: str) -> None:
    if kind in {"proposed_replacement", "reference_recommendation"}:
        if doc.paragraphs:
            doc.paragraphs[-1].paragraph_format.keep_with_next = True
        p = doc.add_paragraph(style="Editorial Note")
        p.paragraph_format.keep_with_next = False
        run = p.add_run(text)
        _set_run_font(run, 10, italic=True, color=NOTE_GRAY)
        return
    p = doc.add_paragraph()
    _set_paragraph_spacing(p)
    run = p.add_run(text)
    _set_run_font(run, 11)


def build(output: Path) -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    doc = Document()
    _configure_styles(doc)
    _configure_sections(doc)
    doc.core_properties.title = "Physician review manuscript"
    doc.core_properties.subject = "Oncology information extraction manuscript"
    doc.core_properties.author = ""
    doc.core_properties.last_modified_by = ""
    doc.core_properties.comments = ""

    pending_figure: str | None = None
    blocks = data["blocks"]
    for block in blocks:
        kind = block["kind"]
        text = block.get("text", "")
        if kind == "front_heading":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            _set_run_font(run, 11, bold=True)
        elif kind == "placeholder":
            p = doc.add_paragraph(style="Title" if block["id"] == "B002" else "Normal")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run(text)
            _set_run_font(run, 16 if block["id"] == "B002" else 11)
        elif kind == "section_heading":
            if text == "SUPPLEMENTARY MATERIAL":
                doc.add_page_break()
            p = doc.add_paragraph(style="Heading 1")
            run = p.add_run(text)
            _set_run_font(run, 14, bold=True)
        elif kind == "subheading":
            # The user explicitly requested the open compound "post processing".
            if text == "Verification and postprocessing order":
                text = "Verification and post processing order"
            p = doc.add_paragraph(style="Heading 2")
            run = p.add_run(text)
            _set_run_font(run, 12, bold=True)
        elif kind == "paragraph":
            _add_text_paragraph(doc, text, kind)
        elif kind == "ordered_item":
            # The physician source already contains the item number.  Use a
            # hanging indent rather than adding a second automatic number.
            p = doc.add_paragraph(style="Normal")
            _set_paragraph_spacing(p, after=3)
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.first_line_indent = Inches(-0.25)
            run = p.add_run(text)
            _set_run_font(run, 11)
        elif kind in {"proposed_replacement", "reference_recommendation"}:
            _add_text_paragraph(doc, text, kind)
        elif kind == "reference":
            _add_reference(doc, text)
        elif kind == "table":
            _add_table(doc, block)
        elif kind == "figure_placeholder":
            if text not in FIGURES:
                raise ValueError(f"Unknown figure placeholder: {text}")
            pending_figure = text
        elif kind == "figure_legend":
            if pending_figure is None:
                raise ValueError(f"Figure legend without placeholder: {block['id']}")
            _add_figure(doc, pending_figure, text)
            pending_figure = None
        else:
            raise ValueError(f"Unknown block kind: {kind}")
    if pending_figure is not None:
        raise ValueError(f"Unmatched figure placeholder: {pending_figure}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="workshop-docx-", dir=str(ROOT)) as tmp_dir:
        raw = Path(tmp_dir) / "raw.docx"
        doc.save(raw)
        scrubber = SKILL_ROOT / "scripts/privacy_scrub.py"
        env = os.environ.copy()
        subprocess.run(
            [os.sys.executable, str(scrubber), str(raw), "--out", str(output)],
            check=True,
            env=env,
        )
        _normalize_docx_package(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
