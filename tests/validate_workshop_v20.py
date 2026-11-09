#!/usr/bin/env python3
"""Deterministic v2.0 manuscript and physician-source preservation checks."""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import render_workshop_draft as renderer  # noqa: E402


def main() -> None:
    paths = [renderer.OUTPUT_HTML, renderer.SOURCE_MD, renderer.PRESERVATION_LEDGER]
    before = {path: path.read_bytes() for path in paths}
    subprocess.run([sys.executable, "render_workshop_draft.py"], cwd=ROOT, check=True)
    after = {path: path.read_bytes() for path in paths}
    assert before == after

    document = before[renderer.OUTPUT_HTML].decode("utf-8")
    markdown = before[renderer.SOURCE_MD].decode("utf-8")
    ledger = before[renderer.PRESERVATION_LEDGER].decode("utf-8")
    data = renderer.load_canonical()
    renderer.validate_v2(document, markdown, ledger, data)

    assert data["counts"] == {
        "body_paragraphs": 122,
        "nonempty_body_paragraphs": 65,
        "native_tables": 4,
        "images": 5,
        "comments": 9,
        "tracked_changes": 0,
        "visible_words": 4243,
    }
    assert sum(len(row) for table in data["tables"] for row in table["rows"]) == 107
    assert 'class="focus-highlight"' not in document
    underlines = re.findall(
        r'<span class="author-review-underline" id="([^"]+)"[^>]*>(.*?)</span>',
        document,
        flags=re.S,
    )
    actual = [(anchor, html.unescape(re.sub(r"<[^>]+>", "", body))) for anchor, body in underlines]
    expected = [
        (anchor, phrase)
        for specs in renderer.AUTHOR_REVIEW_SPECS.values()
        for anchor, phrase, _ in specs
    ]
    assert actual == expected
    assert len(actual) == 16
    assert document.count('href="#author-review-list"') == 17
    review_list_tag = re.search(r'<details[^>]+id="author-review-list"[^>]*>', document)
    assert review_list_tag and not re.search(r'\sopen(?:\s|>)', review_list_tag.group())

    parser = renderer.PreservationParser()
    parser.feed(document)
    assert parser.main_figures == ["1", "2", "3", "4"]
    assert parser.supplementary_figures == ["S1", "S2"]
    assert set(parser.technical_table_containers) == {
        "table-1", "table-2", "table-3", "table-4", "table-5",
        "supplementary-table-s1", "supplementary-table-s2", "supplementary-table-s3",
    }
    assert not parser.visible_comment_ids
    assert 'id="version-2-0"' in document
    assert 'id="version-1-8-1"' in document
    assert "Version 2.0 · November 8, 2026" in document
    assert "difference from 50%" not in parser.visible_text.lower()
    assert "fixed position mapping" not in parser.visible_text.lower()
    assert "randomized placement" not in parser.visible_text.lower()
    assert "doctor Word draft is now the prose authority" in document
    assert "Underlined phrases require author review" in document

    for path in [*renderer.MAIN_FIGURES.values(), *renderer.SUPPLEMENTARY_FIGURES.values()]:
        assert path.exists() and path.suffix == ".svg"
    assert document.count('class="figure-screen"') == 6
    assert document.count('class="figure-print"') == 6
    assert 'data-source-block="P080"' in document
    assert '<p class="physician-paragraph" id="P080"' not in document
    print(
        "V2_VALIDATION_OK "
        "paragraphs=122 cells=107 comments=9 figures=6 tables=8 underlines=16"
    )


if __name__ == "__main__":
    main()
