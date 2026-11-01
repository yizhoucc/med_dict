#!/usr/bin/env python3
"""Deterministic v1.6.4 manuscript, section-revert, and manual-focus checks."""

from __future__ import annotations

import hashlib
import html
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import render_workshop_draft as renderer  # noqa: E402


def focus_spans(document: str) -> list[tuple[str, str, str]]:
    matches = re.findall(
        r'<span (?P<attrs>[^>]*class="[^"]*\bfocus-highlight\b[^"]*"[^>]*)>(?P<body>.*?)</span>',
        document,
        flags=re.S,
    )
    rows = []
    for attrs, body in matches:
        block = re.search(r'data-review-for="([^"]+)"', attrs)
        assert block
        text = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        rows.append((html.unescape(block.group(1)), text, body))
    return rows


def main() -> None:
    html_path = ROOT / "WORKSHOP_PAPER_DRAFT.html"
    md_path = ROOT / "WORKSHOP_PAPER_DRAFT.md"
    before_html = html_path.read_bytes()
    before_md = md_path.read_bytes()
    subprocess.run([sys.executable, "render_workshop_draft.py"], cwd=ROOT, check=True)
    assert html_path.read_bytes() == before_html
    assert md_path.read_bytes() == before_md

    document = before_html.decode("utf-8")
    data = renderer.load_canonical()
    assert data["counts"]["body_paragraphs"] == 127
    assert sum(len(row) for table in data["tables"] for row in table["rows"]) == 35
    assert data["counts"]["comments"] == 12

    state = renderer.review_redline_state(document)
    assert state["changed"] == state["highlighted"]
    assert len(state["changed"]) == 67
    assert state["section_counts"] == {
        "discussion": 2,
        "introduction": 5,
        "methods": 29,
        "results": 6,
        "supplementary-material": 4,
        "supplementary-methods": 21,
    }

    rows = focus_spans(document)
    assert len(rows) == 33
    assert len({row[0] for row in rows}) == 13
    assert max(len(row[1]) for row in rows) == 79
    assert not [row for row in rows if "external-citation" in row[2] or "comment-anchor" in row[2]]
    assert not re.findall(
        r'<(?:p|h3|h4|li|td|th|figcaption)[^>]*class="[^"]*\bfocus-highlight\b',
        document,
    )
    assert not re.search(r"revision-(?:legend|swatch|light|medium|strong)|revision-highlight", document)

    expected_focus_blocks = set(renderer.MANUAL_FOCUS_PHRASES) | {
        "supplement-data-access",
        "supplement-evaluation-interface",
    }
    assert {row[0] for row in rows} == expected_focus_blocks
    focus_text = {row[1] for row in rows}
    assert {
        "focuses on that extraction task",
        "outputs can vary",
        "checking every field in every output can approach the cost of manual extraction",
        "privacy and governance concerns",
        "local models can still produce unstable or incorrect extractions",
        "without physician review",
        "No physician reviewed pancreatic cancer outputs",
        "System identities",
        "judge model and settings",
        "not provided to the models",
        "not used during harness development",
        "not used to calculate the primary physician preference endpoint",
        "A displayed the inference harness output",
        "B displayed the one prompt baseline",
        "blinded to system identities",
        "Source attribution was available only with the harness output",
        "complete output packages",
    }.issubset(focus_text)

    for paragraph_id in ["P004", "P008", "P010", "P012"]:
        match = re.search(rf'<p[^>]*id="{paragraph_id}"[^>]*>(.*?)</p>', document, flags=re.S)
        assert match
        plain = html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip()
        focus_chars = sum(len(text) for block, text, _ in rows if block == paragraph_id)
        assert focus_chars < len(plain) * 0.45
    p006 = re.search(r'<p[^>]*id="P006"[^>]*>(.*?)</p>', document, flags=re.S)
    assert p006 and "focus-highlight" not in p006.group(1)

    blocks = renderer.collect_review_blocks(document)
    for section, next_section in [("introduction", "methods"), ("methods", "results")]:
        assert hashlib.sha256(renderer.section_markup(document, section, next_section).encode()).hexdigest() == renderer.F7_SECTION_BASELINE[section]["markup_sha256"]
        assert hashlib.sha256(renderer.section_block_text(document, section).encode()).hexdigest() == renderer.F7_SECTION_BASELINE[section]["text_sha256"]
    assert not [
        row for row in rows
        if re.fullmatch(r"P\d{3}", row[0]) and 84 <= int(row[0][1:]) <= 119
    ]
    for block_id, focused_text, _ in rows:
        sentences = {
            sentence.strip().rstrip(".!?").strip()
            for sentence in re.split(r"(?<=[.!?])\s+", blocks[block_id].text)
            if sentence.strip()
        }
        assert focused_text.strip().rstrip(".!?").strip() not in sentences
    assert renderer.sentence_sized_focus_runs(document) == []
    synthetic_adjacent_sentence = (
        '<p data-review-block-id="synthetic">'
        '<span class="focus-highlight">System identities</span> '
        '<span class="focus-highlight">were not disclosed</span>.</p>'
    )
    assert renderer.sentence_sized_focus_runs(synthetic_adjacent_sentence) == ["synthetic"]
    assert len(blocks) == 355
    results_text = "\n".join(
        f"{block_id}\t{block.text}" for block_id, block in blocks.items() if block.section == "results"
    )
    assert sum(block.section == "results" for block in blocks.values()) == 121
    assert hashlib.sha256(results_text.encode("utf-8")).hexdigest() == renderer.EXPECTED_V164_RESULTS_SHA256
    assert 'id="loo-results"' in document and 'href="#supplementary-table-s3"' in document
    assert renderer.rendered_physician_prose(document, "P085").endswith(
        "(note cluster robust 95% CI 76.14% to 83.96%)."
    )
    assert renderer.rendered_physician_prose(document, "P096") == "Performance across Clinical Extraction Fields"

    assert document.count('data-comment-text-id=') == 12
    assert document.count('data-literature-entry=') == 30
    assert "Version 1.6.4 · November 1, 2026" in document
    assert "draft v1.6.4" in document

    print(
        "V164_OK "
        f"canonical=127+35+12 visible_blocks={len(blocks)} changed={len(state['changed'])} "
        f"focus_spans={len(rows)} focus_blocks={len(expected_focus_blocks)} results_sha256="
        f"{hashlib.sha256(results_text.encode('utf-8')).hexdigest()}"
    )


if __name__ == "__main__":
    main()
