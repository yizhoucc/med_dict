#!/usr/bin/env python3
"""Deterministic v2.0.1 manuscript and physician-source preservation checks."""

from __future__ import annotations

import html
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import render_workshop_draft as renderer  # noqa: E402


EXPECTED_REVIEW_METADATA = [
    ("author-review-1", "a broader multistep workflow developed from physician feedback can be used to improve note extraction", "研究缺口可以更具体地对应跨癌种适应过程。", "We therefore examined whether a multistep workflow refined with physician feedback on breast cancer notes could be adapted to longitudinal PDAC notes without physician review during PDAC development."),
    ("author-review-2a", "one prompt baseline", "系统名称与全文其他位置不一致。", "single-prompt baseline"),
    ("author-review-2b", "single-prompt baseline", "这里已经使用建议的系统名称，其他三处可与此统一。", "Keep “single-prompt baseline” here and use the same term throughout."),
    ("author-review-2c", "one prompt baseline", "系统名称与全文其他位置不一致。", "single-prompt baseline"),
    ("author-review-3", "population-averaged logistic generalized estimating equation (GEE) with an exchangeable working correlation structure at the note level", "当前句子重复说明 note-level correlation 和 clustering。", "population-averaged logistic GEE with an exchangeable working correlation"),
    ("author-review-4", "to account for clustering of repeated field-level judgments within the same note", "聚类单位是 note，表述可以更直接。", "to account for repeated field-level judgments within each note"),
    ("author-review-5a", "p < 0.05", "Table 5 已提供 overall estimate 的精确 P 值。", "P = 2.67 × 10⁻²⁸"),
    ("author-review-5b", "p < 0.05", "Table 5 已提供 breast cancer estimate 的精确 P 值。", "P = 7.04 × 10⁻¹⁸"),
    ("author-review-5c", "p < 0.05", "Table 5 已提供 pancreatic cancer estimate 的精确 P 值。", "P = 2.30 × 10⁻¹²"),
    ("author-review-6", "P < 0.05", "Supplementary Figure S2 已报告 exact sign test 的精确结果。", "P = 1.05 × 10⁻⁵"),
    ("author-review-7", "significantly affected the LLM judge’s preference", "累计配置比较支持偏好差异，但不能隔离单一步骤的因果效应。", "In the B versus C comparison, the LLM judge preferred verified extraction over decomposed extraction."),
    ("author-review-2d", "one prompt baseline", "系统名称与全文其他位置不一致。", "single-prompt baseline"),
    ("author-review-8", "improve the clinical quality of extracted information", "主要 endpoint 是 physician preference，不是 absolute clinical quality。", "These findings indicate that physicians more often preferred inference harness outputs than same-model baseline outputs."),
    ("author-review-9", "LLM-as-a-judge", "既往研究中的 agreement 会随 rubric、模型和应用场景变化。", "Prior studies have examined LLM-as-a-judge evaluation in clinical generative AI, with agreement varying by rubric and setting (Gershon et al., 2026; Williams et al., 2026)."),
    ("author-review-10", "the clinical rules developed from breast cancer did not provide further improvement for pancreatic cancer", "C versus D 是累计配置比较，不能把结果单独归因于 breast cancer clinical rules。", "The C versus D comparison did not provide evidence of a directional preference between verified extraction and the full inference harness (19 vs 16 preferences; 5 ties; P = .736)."),
    ("author-review-11", "some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status", "实现中确有 receptor-specific prompts 和 checks，但 ablation 没有单独验证这一机制。", "For example, the breast cancer workflow included receptor-specific prompts and post-processing checks for estrogen receptor (ER), progesterone receptor (PR), and HER2 status."),
]

EXPECTED_PAIRED = {
    "author-review-3", "author-review-4", "author-review-5a", "author-review-5b",
    "author-review-5c", "author-review-10", "author-review-11",
}


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
        r'<span class="author-review-underline" id="([^"]+)"([^>]*)>(.*?)</span>',
        document,
        flags=re.S,
    )
    actual = []
    for anchor, attributes, body in underlines:
        reason = re.search(r'\bdata-review-reason="([^"]*)"', attributes)
        suggestion = re.search(r'\bdata-review-suggestion="([^"]*)"', attributes)
        actual.append((
            anchor,
            html.unescape(re.sub(r"<[^>]+>", "", body)),
            html.unescape(reason.group(1)) if reason else "",
            html.unescape(suggestion.group(1)) if suggestion else "",
        ))
        assert 'tabindex="0"' in attributes
        assert 'aria-describedby="author-review-tooltip"' in attributes
        assert ('data-review-paired="true"' in attributes) == (anchor in EXPECTED_PAIRED)
    assert actual == EXPECTED_REVIEW_METADATA
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
    assert 'id="version-2-0-1"' in document
    assert 'id="version-2-0"' in document
    assert 'id="version-1-8-1"' in document
    assert "Version 2.0.1 · November 9, 2026" in document
    assert "difference from 50%" not in parser.visible_text.lower()
    assert "fixed position mapping" not in parser.visible_text.lower()
    assert "randomized placement" not in parser.visible_text.lower()
    assert "doctor Word draft is now the prose authority" in document
    assert "Underlined phrases require author review" in document
    assert "Underlined phrases now show reasons and suggested revisions on hover/focus." in document
    assert "tooltip.id='author-review-tooltip'" in document

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
