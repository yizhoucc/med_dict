#!/usr/bin/env python3
"""Build the submission-ready abstract-only Word document."""

from __future__ import annotations

import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "WORKSHOP_PAPER_EDITABLE.txt"
OUTPUT = ROOT / "WORKSHOP_PAPER_ABSTRACT.docx"


def extract_abstract() -> str:
    text = SOURCE.read_text(encoding="utf-8-sig")
    matches = re.findall(
        r"(?ms)^ABSTRACT\s*\n(.*?)\n\s*INTRODUCTION\s*$",
        text,
    )
    if len(matches) != 1:
        raise ValueError("Expected exactly one ABSTRACT block before INTRODUCTION")
    abstract = matches[0].strip()
    if not abstract or "\n" in abstract:
        raise ValueError("The source ABSTRACT must be one nonempty paragraph")
    if "[TODO]" in abstract or "\u2013" in abstract or "\u2014" in abstract:
        raise ValueError("The source ABSTRACT contains a forbidden placeholder or dash")
    if len(abstract.split()) != 298:
        raise ValueError("The source ABSTRACT must contain 298 whitespace-delimited words")
    return abstract


def set_times_new_roman(run) -> None:
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().get_or_add_rFonts()
    run._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run._element.rPr.rFonts.set(qn("w:cs"), "Times New Roman")


def build() -> str:
    abstract = extract_abstract()
    document = Document()

    section = document.sections[0]
    section.orientation = WD_ORIENT.PORTRAIT
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal._element.rPr.rFonts.set(qn("w:cs"), "Times New Roman")

    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
    paragraph.paragraph_format.left_indent = Inches(0)
    paragraph.paragraph_format.right_indent = Inches(0)
    paragraph.paragraph_format.first_line_indent = Inches(0)
    paragraph.paragraph_format.space_before = Pt(0)
    paragraph.paragraph_format.space_after = Pt(0)
    paragraph.paragraph_format.line_spacing = 1.15
    run = paragraph.add_run(abstract)
    set_times_new_roman(run)
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0, 0, 0)
    run.bold = False
    run.italic = False
    run.underline = False

    fixed_time = datetime(2000, 1, 1, tzinfo=timezone.utc)
    core = document.core_properties
    core.author = ""
    core.last_modified_by = ""
    core.title = ""
    core.subject = ""
    core.comments = ""
    core.keywords = ""
    core.created = fixed_time
    core.modified = fixed_time

    document.save(OUTPUT)
    return abstract


def validate(expected: str) -> None:
    document = Document(OUTPUT)
    nonempty = [p for p in document.paragraphs if p.text.strip()]
    if len(document.paragraphs) != 1 or len(nonempty) != 1:
        raise AssertionError("The DOCX must contain exactly one body paragraph")
    if nonempty[0].text != expected:
        raise AssertionError("The DOCX paragraph does not match the source ABSTRACT")
    if len(nonempty[0].text.split()) != 298:
        raise AssertionError("The DOCX ABSTRACT word count is not 298")
    if "[TODO]" in nonempty[0].text or "\u2013" in nonempty[0].text or "\u2014" in nonempty[0].text:
        raise AssertionError("The DOCX contains a forbidden placeholder or dash")
    if document.tables or document.inline_shapes:
        raise AssertionError("The DOCX contains a table or inline image")
    if len(document.sections) != 1:
        raise AssertionError("The DOCX must contain exactly one section")

    section = document.sections[0]
    expected_twips = {
        "page_width": 12240,
        "page_height": 15840,
        "top_margin": 1440,
        "bottom_margin": 1440,
        "left_margin": 1440,
        "right_margin": 1440,
    }
    for attr, expected_value in expected_twips.items():
        if getattr(section, attr).twips != expected_value:
            raise AssertionError(f"Unexpected section setting: {attr}")

    paragraph = nonempty[0]
    if paragraph.alignment != WD_ALIGN_PARAGRAPH.LEFT:
        raise AssertionError("The ABSTRACT must be left aligned")
    if paragraph.paragraph_format.first_line_indent.twips != 0:
        raise AssertionError("The ABSTRACT must not have a first-line indent")
    if len(paragraph.runs) != 1:
        raise AssertionError("The ABSTRACT must be stored in one run")
    run = paragraph.runs[0]
    if run.font.name != "Times New Roman" or run.font.size != Pt(12):
        raise AssertionError("The ABSTRACT must use 12 pt Times New Roman")
    if run.underline:
        raise AssertionError("The ABSTRACT must not contain underlining")

    with zipfile.ZipFile(OUTPUT) as archive:
        names = set(archive.namelist())
        if any(name.startswith("word/media/") for name in names):
            raise AssertionError("The DOCX package contains image media")
        if any(name.startswith("word/header") or name.startswith("word/footer") for name in names):
            raise AssertionError("The DOCX package contains a header or footer")
        if "word/comments.xml" in names:
            raise AssertionError("The DOCX package contains comments")
        document_xml = archive.read("word/document.xml")
        for marker in (b"w:commentRangeStart", b"w:ins", b"w:del", b"<w:tbl", b"<w:drawing"):
            if marker in document_xml:
                raise AssertionError(f"Unexpected OOXML marker: {marker.decode()}")


if __name__ == "__main__":
    source_abstract = build()
    validate(source_abstract)
    print(
        f"ABSTRACT_DOCX_OK path={OUTPUT} paragraphs=1 words=298 tables=0 images=0"
    )
