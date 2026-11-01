#!/usr/bin/env python3
"""Deterministic v1.6.9 manuscript and manual focus checks."""

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
    assert len(state["changed"]) == 68
    assert state["section_counts"] == {
        "discussion": 3,
        "introduction": 5,
        "methods": 29,
        "results": 6,
        "supplementary-material": 4,
        "supplementary-methods": 21,
    }

    rows = focus_spans(document)
    assert len(rows) == 45
    assert len({row[0] for row in rows}) == 15
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
        "distinguish evidence recorded before and after a regimen change",
        "complete note together with structured context",
        "semantic and source support checks",
        "independent contribution difficult to isolate",
        "support the complete workflow",
        "temporal controls alone",
        "complete harness output as presented",
        "harness was always displayed as A",
        "source attribution appeared only with the harness",
        "both the direction and magnitude",
        "cannot separate the effects",
        "size of the preference difference",
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
    results_discussion_rows = [
        row for row in rows
        if re.fullmatch(r"P\d{3}", row[0]) and 84 <= int(row[0][1:]) <= 119
    ]
    assert [(row[0], row[1]) for row in results_discussion_rows] == (
        [("P115", phrase) for phrase in renderer.MANUAL_FOCUS_PHRASES["P115"]]
        + [("P119", phrase) for phrase in renderer.MANUAL_FOCUS_PHRASES["P119"]]
    )
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
    assert "limitations" not in blocks
    assert 'href="#limitations"' not in document
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
    document_without_comment_ui = re.sub(
        r'<sup class="comment-anchor".*?</sup>', "", document, flags=re.S
    )
    blocks_without_comment_ui = renderer.collect_review_blocks(document_without_comment_ui)
    visible_text_without_comment_ui = "\n".join(
        f"{block_id}\t{block.text}"
        for block_id, block in blocks_without_comment_ui.items()
        if block_id not in {"limitations", "P115", "P119"}
    )
    assert (
        hashlib.sha256(visible_text_without_comment_ui.encode("utf-8")).hexdigest()
        == renderer.EXPECTED_V169_VISIBLE_BLOCK_TEXT_EXCLUDING_COMMENT_UI_P115_AND_P119_SHA256
    )
    p115_text = renderer.rendered_physician_prose(document, "P115")
    assert hashlib.sha256(p115_text.encode("utf-8")).hexdigest() == renderer.EXPECTED_V169_P115_TEXT_SHA256
    assert all(phrase in p115_text for phrase in [
        "distinguish evidence recorded before and after a regimen change",
        "complete note together with structured context on current medications and recent treatment changes",
        "semantic and source support checks",
        "independent contribution difficult to isolate",
        "support the complete workflow but do not determine how much of the observed preference came from the temporal controls alone",
    ])
    assert "G5" not in p115_text and "temporal gate" not in p115_text.lower()
    p119_text = renderer.rendered_physician_prose(document, "P119")
    assert hashlib.sha256(p119_text.encode("utf-8")).hexdigest() == renderer.EXPECTED_V168_P119_TEXT_SHA256
    assert all(phrase in p119_text for phrase in [
        "primary outcome was physician preference rather than absolute extraction accuracy",
        "complete harness output as presented in both cancer types and sensitivity analyses",
        "harness was always displayed as A and the baseline as B",
        "source attribution appeared only with the harness",
        "affected both the direction and magnitude of physician preference",
        "cannot show whether the direction would remain the same with randomized placement or equal source attribution",
        "cannot separate the effects of extraction quality, the surrounding workflow, source attribution, and display position",
        "size of the preference difference and the contribution of each factor therefore remain uncertain",
        "same base model family",
        "generalization across model families was not assessed",
    ])
    assert renderer.EXTERNAL_CITATIONS["williams"][1] not in re.search(
        r'id="P119"[^>]*>(.*?)</p>', document, flags=re.S
    ).group(1)

    assert document.count('data-comment-text-id=') == 12
    assert document.count('data-comment-id=') == 0
    assert 'data-comment-id="13"' not in document and 'comment-anchor-13' not in document
    assert 'data-comment-id="14"' not in document and 'comment-anchor-14' not in document
    assert (
        'data-comment-text-id="13" data-status="resolved" '
        f'data-resolution-basis="{renderer.COMMENT_13_RESOLUTION_BASIS}"'
    ) in document
    assert (
        'data-comment-text-id="14" data-status="resolved" '
        f'data-resolution-basis="{renderer.COMMENT_14_RESOLUTION_BASIS}"'
    ) in document
    assert "Outstanding physician comments (0)" in document
    assert document.count('data-literature-entry=') == 30
    assert "Version 1.6.9 · November 1, 2026" in document
    assert "draft v1.6.9" in document

    print(
        "V169_OK "
        f"canonical=127+35+12 visible_blocks={len(blocks)} changed={len(state['changed'])} "
        f"focus_spans={len(rows)} focus_blocks={len(expected_focus_blocks)} results_sha256="
        f"{hashlib.sha256(results_text.encode('utf-8')).hexdigest()}"
    )


if __name__ == "__main__":
    main()
