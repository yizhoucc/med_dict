#!/usr/bin/env python3
"""Deterministic v1.8.1 manuscript, physician Results, and focus checks."""

from __future__ import annotations

if __name__ == "__main__":
    from validate_workshop_v20 import main as _v20_main
    _v20_main()
    raise SystemExit(0)

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
    assert len(state["changed"]) == 71
    assert state["section_counts"] == {
        "discussion": 3,
        "introduction": 5,
        "methods": 29,
        "results": 9,
        "supplementary-material": 4,
        "supplementary-methods": 21,
    }

    rows = focus_spans(document)
    assert len(rows) == 57
    assert len({row[0] for row in rows}) == 20
    assert max(len(row[1]) for row in rows) == 97
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
        "four physicians evaluated both cancer types",
        "required judgments for each note and field",
        "adjusted probabilities of harness preference ranged from 77.08% to 83.53%",
        "across the seven prespecified clinical extraction fields",
        "not driven by a small number of clinical notes",
        "17 preferences for each, 6 ties, and P = 1.000",
        "24 preferences for verified extraction, 2 for decomposed extraction, 14 ties, and P = 1.05 × 10⁻⁵",
        "19 preferences for verified extraction, 16 for the full inference harness, 5 ties, and P = .736",
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
        assert hashlib.sha256(renderer.section_markup(document, section, next_section).encode()).hexdigest() == renderer.EXPECTED_V180_SECTION_LOCKS[section]["markup_sha256"]
        assert hashlib.sha256(renderer.section_block_text(document, section).encode()).hexdigest() == renderer.EXPECTED_V180_SECTION_LOCKS[section]["text_sha256"]
    results_discussion_rows = [
        row for row in rows
        if re.fullmatch(r"P\d{3}", row[0]) and 84 <= int(row[0][1:]) <= 119
    ]
    assert [(row[0], row[1]) for row in results_discussion_rows] == (
        [("P085", phrase) for phrase in renderer.MANUAL_FOCUS_PHRASES["P085"]]
        + [("P095", phrase) for phrase in renderer.MANUAL_FOCUS_PHRASES["P095"]]
        + [("P097", phrase) for phrase in renderer.MANUAL_FOCUS_PHRASES["P097"]]
        + [("P109", phrase) for phrase in renderer.MANUAL_FOCUS_PHRASES["P109"]]
        + [("P115", phrase) for phrase in renderer.MANUAL_FOCUS_PHRASES["P115"]]
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
    assert len(blocks) == 358
    assert "limitations" not in blocks
    assert 'href="#limitations"' not in document
    results_text = "\n".join(
        f"{block_id}\t{block.text}" for block_id, block in blocks.items() if block.section == "results"
    )
    assert sum(block.section == "results" for block in blocks.values()) == 124
    assert hashlib.sha256(results_text.encode("utf-8")).hexdigest() == renderer.EXPECTED_V180_RESULTS_SHA256
    assert 'id="loo-results"' not in document
    assert 'id="P095"' in document and 'href="#supplementary-table-s3"' in re.search(r'id="P095"[^>]*>(.*?)</p>', document, flags=re.S).group(1)
    assert renderer.rendered_physician_prose(document, "P085").endswith(
        "(note cluster robust 95% CI 76.14% to 83.96%)."
    )
    assert renderer.rendered_physician_prose(document, "P096") == "Performance across clinical extraction fields"
    document_without_comment_ui = re.sub(
        r'<sup class="comment-anchor".*?</sup>', "", document, flags=re.S
    )
    blocks_without_comment_ui = renderer.collect_review_blocks(document_without_comment_ui)
    visible_manuscript_text = "\n".join(
        f"{block_id}\t{block.text}" for block_id, block in blocks_without_comment_ui.items()
    )
    assert hashlib.sha256(visible_manuscript_text.encode("utf-8")).hexdigest() == renderer.EXPECTED_V181_VISIBLE_MANUSCRIPT_TEXT_SHA256
    assert hashlib.sha256(blocks_without_comment_ui["figure-s2::caption"].text.encode("utf-8")).hexdigest() == renderer.EXPECTED_V170_S2_CAPTION_SHA256
    results_displays = "\n".join(
        f"{block_id}\t{block.text}"
        for block_id, block in blocks_without_comment_ui.items()
        if block.section == "results" and (block_id.startswith("table-") or block_id.startswith("figure-"))
    )
    assert hashlib.sha256(results_displays.encode()).hexdigest() == renderer.EXPECTED_V180_RESULTS_DISPLAY_SHA256
    discussion_except_p109 = "\n".join(
        f"{block_id}\t{block.text}"
        for block_id, block in blocks_without_comment_ui.items()
        if block.section == "discussion" and block_id != "P109"
    )
    assert hashlib.sha256(discussion_except_p109.encode()).hexdigest() == renderer.EXPECTED_V180_DISCUSSION_EXCLUDING_P109_SHA256
    supplement_text = "\n".join(
        f"{block_id}\t{block.text}"
        for block_id, block in blocks_without_comment_ui.items()
        if block.section in {"supplementary-material", "supplementary-methods"}
    )
    assert hashlib.sha256(supplement_text.encode()).hexdigest() == renderer.EXPECTED_V180_SUPPLEMENT_SHA256
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

    p085_text = renderer.rendered_physician_prose(document, "P085")
    assert p085_text.count("2,160") == 1 and "2160" not in p085_text
    assert all(value in p085_text for value in ["699", "1,290", "171", "32.36%", "59.72%", "7.92%", "870", "80.34%"])
    p095_text = renderer.rendered_physician_prose(document, "P095")
    assert all(value in p095_text for value in ["+24.4", "39 of 40", "77.08%", "83.53%"])
    assert re.search(r'data-source-block="P100"', document) and not re.search(r'(?<![\w-])id="P100"', document)
    p109_text = renderer.rendered_physician_prose(document, "P109")
    assert hashlib.sha256(p109_text.encode()).hexdigest() == renderer.EXPECTED_V181_P109_TEXT_SHA256
    assert "The note level results also indicate" in p109_text
    assert "The note level findings also suggest" not in p109_text
    assert "not driven by a small number of clinical notes" in p109_text
    for heading_id, heading in [
        ("overall-physician-preference", "Overall physician preference"),
        ("adjusted-preference-estimates", "Adjusted preference estimates"),
        ("sensitivity-and-physician-agreement", "Sensitivity and physician agreement"),
        ("performance-across-clinical-extraction-fields", "Performance across clinical extraction fields"),
        ("technical-ablation", "Technical ablation"),
    ]:
        match = re.search(rf'<h3[^>]*id="{heading_id}"[^>]*>(.*?)</h3>', document, flags=re.S)
        assert match and html.unescape(re.sub(r"<[^>]+>", "", match.group(1))) == heading
        assert "focus-highlight" not in match.group(1)
    ablation = re.search(r'id="technical-ablation-results"[^>]*>(.*?)</p>', document, flags=re.S)
    assert ablation
    ablation_text = html.unescape(re.sub(r"<[^>]+>", "", ablation.group(1)))
    assert 110 <= len(ablation_text.split()) <= 140
    assert all(value in ablation_text for value in [
        "120 adjacent comparisons", "single nonphysician LLM judge", "Table 2",
        "17 preferences for each, 6 ties, and P = 1.000",
        "24 preferences for verified extraction, 2 for decomposed extraction, 14 ties, and P = 1.05 × 10⁻⁵",
        "19 preferences for verified extraction, 16 for the full inference harness, 5 ties, and P = .736",
        "unadjusted two sided exact sign tests excluded ties", "Supplementary Figure S2",
    ])
    focus_attrs = re.findall(r'<span (?P<attrs>[^>]*class="[^"]*\bfocus-highlight\b[^"]*"[^>]*)>', document)
    intensities = [re.search(r'data-review-intensity="([^"]+)"', attrs).group(1) for attrs in focus_attrs]
    assert intensities.count("light") == 4 and intensities.count("strong") == 3
    assert set(intensities) == {"light", "medium", "strong"}
    strong_text = {
        text for block, text, _ in rows
        if block == "technical-ablation-results" and text.startswith(("17 preferences", "24 preferences", "19 preferences"))
    }
    assert len(strong_text) == 3
    assert all('title="Added in v1.8.0"' in attrs for attrs in focus_attrs if 'data-review-for="technical-ablation-results"' in attrs)

    preservation = (ROOT / "manuscript/results_source_preservation_v1_8_0.md").read_text()
    assert "13 sentences are verbatim" in preservation
    assert "11 received light edits" in preservation
    assert "1 moved to Discussion" in preservation
    assert "3 were removed" in preservation
    assert hashlib.sha256((ROOT / "manuscript/doctor_drafts/202601023_Manuscript_JAMA.canonical.json").read_bytes()).hexdigest() == renderer.EXPECTED_CANONICAL_SHA256

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
    assert "Version 1.8.1 · November 1, 2026" in document
    assert "draft v1.8.1" in document

    print(
        "V181_OK "
        f"canonical=127+35+12 visible_blocks={len(blocks)} changed={len(state['changed'])} "
        f"focus_spans={len(rows)} focus_blocks={len(expected_focus_blocks)} results_sha256="
        f"{hashlib.sha256(results_text.encode('utf-8')).hexdigest()}"
    )


if __name__ == "__main__":
    main()
