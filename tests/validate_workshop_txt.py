#!/usr/bin/env python3
"""Validate the deterministic Word-friendly manuscript text export."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import export_workshop_txt as exporter  # noqa: E402
import render_workshop_draft as renderer  # noqa: E402


def main() -> None:
    before = exporter.OUTPUT_TXT.read_bytes()
    subprocess.run([sys.executable, "export_workshop_txt.py"], cwd=ROOT, check=True)
    after = exporter.OUTPUT_TXT.read_bytes()
    assert before == after
    assert after.startswith(b"\xef\xbb\xbf")
    assert after.count(b"\xef\xbb\xbf") == 1
    assert 30_000 < len(after) < 200_000

    text = after.decode("utf-8-sig")
    lower = text.lower()
    assert not re.search(r"<[a-z/][^>]*>", text, flags=re.I)
    for forbidden in [
        "<script", "<style", "<svg", "base64", "author-review-tooltip",
        "为什么需要复核", "建议改为", "Original:", "Version history",
        "English working manuscript", "Current change:", "Outstanding physician comments",
        "Author literature-review workspace", "Draft relevance:", "technical insert",
    ]:
        assert forbidden.lower() not in lower, forbidden

    expected_sections = [
        "TITLE", "AUTHORS", "ABSTRACT", "INTRODUCTION", "METHODS", "RESULTS",
        "DISCUSSION", "REFERENCES", "SUPPLEMENTARY MATERIAL", "Supplementary Methods",
    ]
    positions = [re.search(rf"(?m)^{re.escape(section)}$", text).start() for section in expected_sections]
    assert positions == sorted(positions)
    assert "TITLE\n[TODO]" in text and "AUTHORS\n[TODO]" in text and "ABSTRACT\n[TODO]" in text

    body, references_and_supplement = text.split("\n\nREFERENCES\n\n", 1)
    references, supplement = references_and_supplement.split("\n\nSUPPLEMENTARY MATERIAL\n\n", 1)
    citation_groups = re.findall(r"\[(\d+(?:,\d+)*)\]", body + "\n" + supplement)
    citation_numbers = {int(number) for group in citation_groups for number in group.split(",")}
    assert citation_numbers == set(range(1, 16))
    assert not re.search(r"\]\s*[,;]\s*\[", body + supplement)
    assert not re.search(r"\[(?:0|1[6-9]|[2-9]\d)(?:,|\])", body + supplement)

    reference_lines = [line for line in references.splitlines() if line.strip()]
    assert len(reference_lines) == 15
    for number, key in enumerate(renderer.FORMAL_REFERENCE_ORDER, 1):
        _, url, citation = renderer.EXTERNAL_CITATIONS[key]
        assert reference_lines[number - 1] == f"{number}. {citation} {url}"
        assert citation_numbers.__contains__(number)
    assert "Bhattarai K" not in references

    figure_labels = ["Figure 1", "Figure 2", "Figure 3", "Figure 4", "Supplementary Figure S1", "Supplementary Figure S2"]
    parser = exporter.TreeParser()
    parser.feed(exporter.generated_html())
    for label in figure_labels:
        assert text.count(f"[Insert {label} here]") == 1
        assert len(re.findall(rf"(?m)^{re.escape(label)}\.", text)) == 1
    for figure_id in ["figure-1", "figure-2", "figure-3", "figure-4", "figure-s1", "figure-s2"]:
        expected = exporter.figure_block(exporter.find_by_id(parser.root, figure_id))
        assert text.count(expected) == 1

    table_labels = ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5", "Supplementary Table S1", "Supplementary Table S2", "Supplementary Table S3"]
    for label in table_labels:
        assert len(re.findall(rf"(?m)^{re.escape(label)}\.", text)) == 1
    for table_id in ["table-1", "table-2", "table-3", "table-4", "table-5", "supplementary-table-s1", "supplementary-table-s2", "supplementary-table-s3"]:
        expected = exporter.table_block(exporter.find_by_id(parser.root, table_id))
        assert text.count(expected) == 1
    tab_lines = [line for line in text.splitlines() if "\t" in line]
    assert len(tab_lines) >= 35
    assert all("\n" not in cell and "\r" not in cell for line in tab_lines for cell in line.split("\t"))

    for required in [
        "2,160", "79.98% (75.75%-83.63%)", "3.995 (3.124-5.110)",
        "2.67e-28", "1.05 × 10⁻⁵", "Cohen’s κ", "40 holdout note texts",
        "Huang et al. [5] demonstrated", "Hein et al. [6] showed", "P = .736",
        "Supplementary Table S3. Sensitivity analysis", "[Insert Supplementary Figure S2 here]",
    ]:
        assert required in text, required
    assert "P =.736" not in text and "and.736" not in text

    print(
        "TXT_VALIDATION_OK bom=utf8 refs=15 cited=15 figures=6 tables=8 "
        f"tabs={len(tab_lines)} bytes={len(after)}"
    )


if __name__ == "__main__":
    main()
