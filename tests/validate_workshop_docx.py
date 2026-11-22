#!/usr/bin/env python3
"""Structural validation for the physician-first Word review manuscript."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from docx import Document
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "manuscript/physician_text/2026-11-21/WORKSHOP_PAPER_WORD_SOURCE.json"
EDITABLE_SOURCE = ROOT / "WORKSHOP_PAPER_EDITABLE.txt"
DOCX = ROOT / "WORKSHOP_PAPER_PHYSICIAN_REVIEW.docx"
FIGURE_DIR = ROOT / "results/extraction_comparison/clinician_ratings/draft_figures"
FIGURE_FILES = [
    "figure1_development_pathway_print.png",
    "figure2_evaluator_distribution_print.png",
    "figure3_interrater_matrix_print.png",
    "figure4_effect_profile_print.png",
    "supplementary_figure_s1_note_margins_print.png",
    "supplementary_figure_s2_staged_ablation_print.png",
]
ALT_DESCRIPTIONS = {
    "Figure 1 development and adaptation workflow with four panels.",
    "Figure 2 physician preference distributions overall and by cancer type.",
    "Figure 3 pairwise exact agreement and Cohen kappa among four physicians.",
    "Figure 4 physician preference by core oncology field.",
    "Supplementary Figure S1 note level preference margins by cancer type.",
    "Supplementary Figure S2 cumulative ablation configurations and adjacent comparisons.",
}
SOURCE_SHA256 = "bcb114f6a0a510783177a836bc796c58ad48cc82b9113f5c25d278586389f090"
EXPECTED_ABSTRACT = (
    "Many clinical uses of large language models (LLMs) depend on reliable extraction from unstructured oncology notes, yet model outputs may omit facts, confuse temporal relationships, or include unsupported information. "
    "We compared physician preferences for outputs from an inference harness and a single prompt baseline using the same frozen, locally deployed LLM. "
    "Using Qwen2.5-32B-Instruct, we developed the breast cancer harness with physician review and adapted it to pancreatic ductal adenocarcinoma (PDAC) through LLM review and regression testing, without physician review of PDAC development outputs. "
    "The harness used separate extraction for each field, verification, clinical rules, deterministic processing, and source attribution when available. "
    "Four physicians compared the systems without disclosure of system identity on an untouched holdout set of 40 CORAL notes, 20 per cancer type. "
    "The primary outcome was the probability of harness preference among directional judgments, estimated with a population-averaged logistic generalized estimating equation that excluded ties, adjusted for physician and cancer type, and clustered repeated judgments by note. "
    "Across 2,160 judgments, the harness was preferred 699 times, the baseline 171 times, and 1,290 comparisons were ties. "
    "Among 870 directional judgments, the adjusted probability of harness preference was 79.98% (95% CI, 75.75%-83.63%; adjusted odds ratio, 3.995; P = 2.67 × 10⁻²⁸). "
    "The estimates were 81.66% for breast cancer (95% CI, 76.02%-86.21%) and 78.15% for PDAC (95% CI, 71.47%-83.62%). "
    "In an exploratory staged ablation evaluated by one blinded nonphysician LLM, verified extraction was preferred over decomposed extraction in 24 comparisons versus 2, with 14 ties (P = 1.05 × 10⁻⁵). "
    "Neither of the other adjacent comparisons showed a significant directional preference. "
    "In this pilot evaluation, physicians preferred the complete harness output package across both cancer types. "
    "Further studies should evaluate extraction accuracy against expert reference standards and include larger datasets, more physicians, other cancer types, and other open-weight models."
)
EXPECTED_ABSTRACT_WORD_COUNT = 298


def reference_link_label(url: str) -> str:
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


def fail(message: str) -> None:
    raise AssertionError(message)


def main() -> None:
    if not DOCX.is_file():
        fail(f"Missing DOCX: {DOCX}")
    size = DOCX.stat().st_size
    if not 500_000 <= size <= 15_000_000:
        fail(f"Unexpected DOCX size: {size}")

    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA256:
        fail("Authoritative source SHA256 changed")
    expected_counts = {
        "blocks": 123,
        "physician_content_blocks": 119,
        "direct_copyedit_occurrences": 18,
        "editorial_annotations": 4,
        "tables": 8,
        "figure_placeholders": 6,
        "figure_legends": 6,
        "references": 15,
        "front_matter_todos": 3,
    }
    if source.get("counts") != expected_counts:
        fail(f"Unexpected source counts: {source.get('counts')}")
    if not source.get("humanizer_review", {}).get("draft_audit_final_completed"):
        fail("Proposed replacement humanizer review is incomplete")
    editable_lines = EDITABLE_SOURCE.read_text(encoding="utf-8-sig").splitlines()
    try:
        abstract_start = editable_lines.index("ABSTRACT") + 1
        abstract_end = editable_lines.index("INTRODUCTION", abstract_start)
    except ValueError as error:
        fail(f"Editable source Abstract boundaries are invalid: {error}")
    editable_abstracts = [line.strip() for line in editable_lines[abstract_start:abstract_end] if line.strip()]
    if editable_abstracts != [EXPECTED_ABSTRACT]:
        fail("Editable source Abstract does not exactly match the approved text")
    if len(EXPECTED_ABSTRACT.split()) != EXPECTED_ABSTRACT_WORD_COUNT:
        fail("Approved Abstract word count constant is stale")
    if "\u2013" in EXPECTED_ABSTRACT or "\u2014" in EXPECTED_ABSTRACT:
        fail("Approved Abstract contains an en dash or em dash")
    document = Document(DOCX)
    if len(document.tables) != 8:
        fail(f"Expected 8 tables, found {len(document.tables)}")
    if len(document.inline_shapes) != 6:
        fail(f"Expected 6 inline figures, found {len(document.inline_shapes)}")
    drawing_paragraphs = [i for i, p in enumerate(document.paragraphs) if p._p.xpath(".//w:drawing")]
    if len(drawing_paragraphs) != 6:
        fail(f"Expected 6 drawing paragraphs, found {len(drawing_paragraphs)}")
    for index in drawing_paragraphs:
        if index + 1 >= len(document.paragraphs) or document.paragraphs[index + 1].style.name != "Caption":
            fail("Each figure must be followed immediately by its Caption paragraph")

    paragraphs = document.paragraphs
    paragraph_texts = [p.text for p in paragraphs]
    all_paragraph_text = "\n".join(paragraph_texts)
    all_cell_text = "\n".join(cell.text for table in document.tables for row in table.rows for cell in row.cells)
    all_text = all_paragraph_text + "\n" + all_cell_text
    ordered_items = [p for p in paragraphs if re.match(r"^[1-5]\. Gate [1-5] ", p.text)]
    if len(ordered_items) != 5 or any(p.style.name != "Normal" for p in ordered_items):
        fail("Supplement gate list must preserve five source numbers without automatic numbering")

    if all_text.count("[TODO]") != 2:
        fail(f"Expected 2 remaining TODO placeholders, found {all_text.count('[TODO]')}")
    if paragraph_texts.count("ABSTRACT") != 1:
        fail(f"Expected one ABSTRACT heading, found {paragraph_texts.count('ABSTRACT')}")
    if paragraph_texts.count(EXPECTED_ABSTRACT) != 1:
        fail("Approved Abstract is missing or duplicated in the DOCX")
    abstract_heading_index = paragraph_texts.index("ABSTRACT")
    if abstract_heading_index + 1 >= len(paragraph_texts) or paragraph_texts[abstract_heading_index + 1] != EXPECTED_ABSTRACT:
        fail("Approved Abstract must immediately follow the ABSTRACT heading")
    if len(paragraph_texts[abstract_heading_index + 1].split()) != EXPECTED_ABSTRACT_WORD_COUNT:
        fail("DOCX Abstract word count mismatch")
    if "\u2013" in paragraph_texts[abstract_heading_index + 1] or "\u2014" in paragraph_texts[abstract_heading_index + 1]:
        fail("DOCX Abstract contains an en dash or em dash")
    if "Verification and post processing order" not in all_text:
        fail("Open compound 'post processing' heading missing")
    for forbidden in ("post-processing", "post-proccessing", "deterministric", "sinlge", "have showed"):
        if forbidden in all_text:
            fail(f"Forbidden stale wording remains: {forbidden}")
    if "Verification and postprocessing order" in all_text:
        fail("Closed compound postprocessing remains in heading")

    annotations = source["editorial_annotations"]
    for annotation in annotations:
        if paragraph_texts.count(annotation["text"]) != 1:
            fail(f"Annotation missing or duplicated: {annotation['id']}")
    editorial_notes = [p for p in paragraphs if p.style.name == "Editorial Note"]
    if len(editorial_notes) != 4:
        fail(f"Expected 4 Editorial Note paragraphs, found {len(editorial_notes)}")
    for p in editorial_notes:
        if not p.text.startswith("(Proposed replacement:") and not p.text.startswith("(Reference recommendation:"):
            fail(f"Unexpected editorial note: {p.text[:80]}")
        note_index = paragraphs.index(p)
        if note_index == 0 or paragraphs[note_index - 1].paragraph_format.keep_with_next is not True:
            fail(f"Editorial note is not kept with its preceding paragraph: {p.text[:80]}")

    source_tables = [b for b in source["blocks"] if b["kind"] == "table"]
    for src, table in zip(source_tables, document.tables, strict=True):
        found = [[cell.text for cell in row.cells] for row in table.rows]
        expected = src["rows"]
        if src["caption"].startswith("Table 2."):
            expected = [["Configuration", "Description"], *expected]
        if found != expected:
            fail(f"Table content mismatch: {src['caption']}")

    for block in source["blocks"]:
        kind = block["kind"]
        text = block.get("text", "")
        if block["id"] == "B006":
            continue
        if kind in {"figure_placeholder", "table"}:
            continue
        if kind == "subheading" and text == "Verification and postprocessing order":
            text = "Verification and post processing order"
        if kind == "figure_legend":
            if paragraph_texts.count(text) != 1:
                fail(f"Figure legend missing or duplicated: {block['id']}")
        elif kind == "reference":
            # Hyperlink text is not always included in python-docx Paragraph.text.
            prefix = re.sub(r"\s+https?://\S+$", "", text)
            if not any(p.startswith(prefix) for p in paragraph_texts):
                fail(f"Reference missing: {block['id']}")
        elif kind not in {"proposed_replacement", "reference_recommendation"} and text:
            if text not in all_text:
                fail(f"Source text missing from DOCX: {block['id']} {text[:80]}")

    with zipfile.ZipFile(DOCX) as archive:
        names = set(archive.namelist())
        document_xml = archive.read("word/document.xml").decode("utf-8")
        styles_xml = archive.read("word/styles.xml").decode("utf-8")
        core_xml = archive.read("docProps/core.xml").decode("utf-8")
        rels_xml = archive.read("word/_rels/document.xml.rels").decode("utf-8")
        media = sorted(name for name in names if name.startswith("word/media/") and name.endswith(".png"))
        if len(media) != 6:
            fail(f"Expected 6 embedded PNG files, found {len(media)}")
        if document_xml.count("<wp:inline") != 6 or "<wp:anchor" in document_xml:
            fail("Figures must be six inline drawings and no floating anchors")
        if document_xml.count("descr=") < 6:
            fail("All figures must have alt descriptions")
        found_alt = set(re.findall(r'descr="([^"]+)"', document_xml))
        if found_alt != ALT_DESCRIPTIONS:
            fail(f"Figure alt text mismatch: {found_alt}")
        if (
            "w:commentRangeStart" in document_xml
            or re.search(r"<w:ins(?:\s|>)", document_xml)
            or re.search(r"<w:del(?:\s|>)", document_xml)
        ):
            fail("Unexpected Word comments or tracked changes")
        if "comments.xml" in "\n".join(names):
            fail("Unexpected Word comments part")
        if "docProps/custom.xml" in names:
            fail("Unexpected custom document properties remain")
        if re.search(r"<dc:creator>[^<]+", core_xml) or re.search(r"<cp:lastModifiedBy>[^<]+", core_xml):
            fail("Personal author metadata remains")
        if rels_xml.count("relationships/hyperlink") != 15:
            fail(f"Expected 15 reference hyperlinks, found {rels_xml.count('relationships/hyperlink')}")
        rel_root = ET.fromstring(rels_xml)
        rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
        rel_targets = {
            rel.attrib["Id"]: rel.attrib["Target"]
            for rel in rel_root.findall(f"{{{rel_ns}}}Relationship")
            if rel.attrib.get("Type", "").endswith("/hyperlink")
        }
        doc_root = ET.fromstring(document_xml)
        w_ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        r_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        actual_links = Counter()
        for hyperlink in doc_root.findall(f".//{{{w_ns}}}hyperlink"):
            relationship_id = hyperlink.attrib.get(f"{{{r_ns}}}id")
            if relationship_id not in rel_targets:
                continue
            visible = "".join(node.text or "" for node in hyperlink.findall(f".//{{{w_ns}}}t"))
            actual_links[(rel_targets[relationship_id], visible)] += 1
        source_urls = [
            re.search(r"(https?://\S+)$", block["text"]).group(1)
            for block in source["blocks"]
            if block["kind"] == "reference"
        ]
        expected_links = Counter((url, reference_link_label(url)) for url in source_urls)
        if actual_links != expected_links:
            fail(f"Reference hyperlink target/label mismatch: {actual_links}")
        if document_xml.count("w:tblHeader") != 8:
            fail(f"Expected 8 repeating table header rows, found {document_xml.count('w:tblHeader')}")
        expected_rows = sum(len(table["rows"]) for table in source_tables) + 1
        if document_xml.count("w:cantSplit") != expected_rows:
            fail(f"Expected {expected_rows} non-splitting table rows, found {document_xml.count('w:cantSplit')}")
        title_style = re.search(r'<w:style[^>]+w:styleId="Title".*?</w:style>', styles_xml, re.S)
        if title_style is None or "<w:pBdr" in title_style.group(0):
            fail("Title style must exist without a decorative paragraph border")

        embedded_hashes = {hashlib.sha256(archive.read(name)).hexdigest() for name in media}

    manifest = {}
    for line in (FIGURE_DIR / "MANIFEST.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, filename = line.split(None, 1)
        manifest[filename.strip()] = digest
    for filename in FIGURE_FILES:
        path = FIGURE_DIR / filename
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if manifest.get(filename) != digest:
            fail(f"Figure manifest mismatch: {filename}")
        with Image.open(path) as image:
            if image.width != 2250:
                fail(f"Approved figure width is not 2250 px: {filename}={image.size}")
            dpi = image.info.get("dpi")
            if dpi is None or any(abs(float(value) - 300.0) > 1.0 for value in dpi[:2]):
                fail(f"Approved figure DPI is not 300: {filename}={dpi}")
    approved_hashes = {manifest[name] for name in FIGURE_FILES}
    if embedded_hashes != approved_hashes:
        fail("Embedded figures do not exactly match the six approved PNG assets")

    print(
        "DOCX_VALIDATION_OK "
        f"bytes={size} paragraphs={len(document.paragraphs)} tables={len(document.tables)} "
        f"figures={len(document.inline_shapes)} notes={len(editorial_notes)} refs=15 "
        f"abstract_words={EXPECTED_ABSTRACT_WORD_COUNT}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"DOCX_VALIDATION_FAIL: {exc}", file=sys.stderr)
        raise
