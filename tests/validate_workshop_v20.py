#!/usr/bin/env python3
"""Deterministic v2.3.0 manuscript and physician-source preservation checks."""

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
    ("author-review-12", "about 80% of healthcare data are unstructured", "Kong 支持的是 healthcare data 的约 80% 估计，不是更窄的 EHR clinical data 精确比例。", 'Original: "80% of clinical data in EHR exist in unstructured form." Revised: "about 80% of healthcare data are unstructured."'),
    ("author-review-13a", "Reviewing every extracted field may be impractical in routine use, which motivated the verification steps in this workflow.", "原句把逐字段核查成本与人工抽取成本相比，现有研究没有提供这一成本比较。", 'Original: "checking every field in every output can approach the cost of manual extraction." Revised: "Reviewing every extracted field may be impractical in routine use, which motivated the verification steps in this workflow."'),
    ("author-review-13b", "In the CORAL benchmark, commercial models outperformed the open model evaluated in that study", "CORAL 支持该研究内部的模型比较，不支持笼统声称 commercial models 在临床抽取中普遍更好。", 'Original: "Commercial models perform better on some clinical extraction tasks." Revised: "In the CORAL benchmark, commercial models outperformed the open model evaluated in that study."'),
    ("author-review-13c", "raises privacy and governance concerns", "现有引用没有直接建立 HIPAA-specific compliance 结论，因此只删除该限定语。", 'Original: "raises privacy and governance concerns, including HIPAA compliance." Revised: "raises privacy and governance concerns."'),
    ("author-review-13d", "Local deployment can keep PHI within the institution, but it does not by itself ensure accurate extraction.", "Wiest 支持本地处理的 privacy 逻辑；accuracy 仍需要单独验证，不能由部署位置推定。", 'Original: "Local deployment avoids external transfer of PHI, but local models can still produce unstable or incorrect extractions." Revised: "Local deployment can keep PHI within the institution, but it does not by itself ensure accurate extraction."'),
    ("author-review-1", "a broader multistep workflow developed from physician feedback can be used to improve note extraction", "研究缺口可以更具体地对应跨癌种适应过程。", "We therefore examined whether a multistep workflow refined with physician feedback on breast cancer notes could be adapted to longitudinal PDAC notes without physician review during PDAC development."),
    ("author-review-2a", "one prompt baseline", "系统名称与全文其他位置不一致。", "single-prompt baseline"),
    ("author-review-2b", "single-prompt baseline", "这里已经使用建议的系统名称，其他三处可与此统一。", "Keep “single-prompt baseline” here and use the same term throughout."),
    ("author-review-2c", "one prompt baseline", "系统名称与全文其他位置不一致。", "single-prompt baseline"),
    ("author-review-14", "with AWQ quantization, served locally through vLLM", "Qwen technical report 支持模型家族；AWQ quantization 和 vLLM 是本研究的实现配置。", 'Original: "Qwen2.5-32B-Instruct-AWQ served locally through vLLM (Qwen et al., 2025)." Revised: "Qwen2.5-32B-Instruct (Qwen et al., 2025) with AWQ quantization, served locally through vLLM."'),
    ("author-review-3", "population-averaged logistic generalized estimating equation (GEE) with an exchangeable working correlation structure at the note level", "当前句子重复说明 note-level correlation 和 clustering。", "population-averaged logistic GEE with an exchangeable working correlation"),
    ("author-review-4", "to account for clustering of repeated field-level judgments within the same note", "聚类单位是 note，表述可以更直接。", "to account for repeated field-level judgments within each note"),
    ("author-review-5a", "p < 0.05", "Table 5 已提供 overall estimate 的精确 P 值。", "P = 2.67 × 10⁻²⁸"),
    ("author-review-5b", "p < 0.05", "Table 5 已提供 breast cancer estimate 的精确 P 值。", "P = 7.04 × 10⁻¹⁸"),
    ("author-review-5c", "p < 0.05", "Table 5 已提供 pancreatic cancer estimate 的精确 P 值。", "P = 2.30 × 10⁻¹²"),
    ("author-review-2d", "one prompt baseline", "系统名称与全文其他位置不一致。", "single-prompt baseline"),
    ("author-review-8", "improve the clinical quality of extracted information", "主要 endpoint 是 physician preference，不是 absolute clinical quality。", "These findings indicate that physicians more often preferred inference harness outputs than same-model baseline outputs."),
    ("author-review-9", "Prior studies have examined LLM-as-a-judge evaluation in clinical generative AI, with agreement varying by rubric and setting", "既往研究中的 agreement 会随 rubric、模型和应用场景变化，不能概括为共同显示 high concordance。", 'Original: "LLM-as-a-judge has been reported in prior studies, which showed high concordance on text-comparison rubrics such as faithfulness and completeness." Revised: "Prior studies have examined LLM-as-a-judge evaluation in clinical generative AI, with agreement varying by rubric and setting."'),
    ("author-review-10", "the clinical rules developed from breast cancer did not provide further improvement for pancreatic cancer", "C versus D 是累计配置比较，不能把结果单独归因于 breast cancer clinical rules。", "The C versus D comparison did not provide evidence of a directional preference between verified extraction and the full inference harness (19 vs 16 preferences; 5 ties; P = .736)."),
    ("author-review-11", "some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status", "实现中确有 receptor-specific prompts 和 checks，但 ablation 没有单独验证这一机制。", "For example, the breast cancer workflow included receptor-specific prompts and post-processing checks for estrogen receptor (ER), progesterone receptor (PR), and HER2 status."),
    ("author-review-15a", "follow reporting guidance for model and evaluation procedures", "TRIPOD-LLM 支持 reporting guidance，不直接支持扩大数据集或跨模型 generalizability。", 'Original: "TRIPOD-LLM was cited after all future-study recommendations." Revised: "follow reporting guidance for model and evaluation procedures."'),
    ("author-review-15b", "use structured approaches to human evaluation", "QUEST 支持 human evaluation 的设计与实施，不直接支持 expert-reference accuracy 结论。", 'Original: "QUEST was cited after all future-study recommendations." Revised: "use structured approaches to human evaluation."'),
    ("author-review-15c", "evaluate extraction accuracy against expert reference standards", "VALID 支持将模型提取与 expert reference standard 比较。", 'Original: "evaluate extraction accuracy against expert reference standards" with three citations at the end of the sentence. Revised: retain the clause and cite VALID directly.'),
    ("author-review-15d", "Larger and more diverse clinical datasets are also needed to assess whether the harness generalizes across open-weight models.", "现有三篇 framework 文献不直接证明本研究需要的样本规模或跨模型 generalizability，因此该建议不附这些引用。", 'Original: "include larger and more diverse clinical datasets, and assess the generalizability of the harness across different open-weight models" with three citations. Revised: "Larger and more diverse clinical datasets are also needed to assess whether the harness generalizes across open-weight models."'),
]

EXPECTED_PAIRED = {
    "author-review-3", "author-review-4", "author-review-5a", "author-review-5b",
    "author-review-5c", "author-review-10", "author-review-11",
    "author-review-15a", "author-review-15b", "author-review-15c", "author-review-15d",
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
    assert len(actual) == 24
    assert document.count('href="#author-review-list"') == 25
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
    assert 'id="version-2-3-0"' in document
    assert 'id="version-2-2-1"' in document
    assert 'id="version-2-2-0"' in document
    assert 'id="version-2-1-1"' in document
    assert 'id="version-2-1-0"' in document
    assert 'id="version-2-0-1"' in document
    assert 'id="version-2-0"' in document
    assert 'id="version-1-8-1"' in document
    assert "Version 2.3.0 · November 18, 2026" in document
    assert "Unified the palette across Figures 1 to 4, changed Figure 3 to a blue kappa scale, and completed the formal References and Supplement citation audit." in document
    assert ".figure{max-width:100%;margin-left:0;margin-right:0;overflow-x:auto" in document
    assert ".figure .figure-screen{width:700px;max-width:none}" in document
    assert ".figure{max-width:none;margin:14px 0 18px;overflow:visible}" in document
    assert "difference from 50%" not in parser.visible_text.lower()
    assert "fixed position mapping" not in parser.visible_text.lower()
    assert "randomized placement" not in parser.visible_text.lower()
    assert "doctor Word draft is now the prose authority" in document
    assert "Underlined phrases require author review" in document
    assert "Underlined phrases now show reasons and suggested revisions on hover/focus." in document
    assert "tooltip.id='author-review-tooltip'" in document

    formal_references = re.findall(
        r'<li id="reference-(\d+)" data-reference-key="([^"]+)">',
        document,
    )
    assert formal_references == [
        (str(index), key)
        for index, key in enumerate(renderer.FORMAL_REFERENCE_ORDER, 1)
    ]
    citation_targets = re.findall(r'<a class="citation" href="#reference-(\d+)"', document)
    assert set(citation_targets) == {
        str(index) for index in range(1, len(renderer.FORMAL_REFERENCE_ORDER) + 1)
    }
    assert not re.search(r'<a[^>]+class="[^"]*\bexternal-citation\b', document)
    assert document.find('id="section-discussion"') < document.find('id="references"') < document.find('id="supplementary-material"')
    assert '<section class="fold references custom-fold is-collapsed" id="references" data-origin="technical-insert">' in document
    assert '<button class="fold-toggle" type="button" aria-expanded="false" aria-controls="references-body">References</button>' in document
    assert "Prespecified core fields used in the exploratory field analysis" in document
    assert "The blue scale encodes kappa" in document
    assert "The green scale encodes kappa" not in document
    p006 = renderer.rendered_physician_prose(document, "P006")
    p008 = renderer.rendered_physician_prose(document, "P008")
    p027 = re.search(r'id="P027"[^>]*>(.*?)</p>', document, flags=re.S).group(1)
    p108 = renderer.rendered_physician_prose(document, "P108")
    p114 = renderer.rendered_physician_prose(document, "P114")
    assert p006.startswith("It is estimated that about 80% of healthcare data are unstructured")
    assert "including HIPAA compliance" not in p008
    assert "approach the cost of manual extraction" not in p008
    assert "In the CORAL benchmark, commercial models outperformed the open model evaluated in that study" in p008
    assert "raises privacy and governance concerns" in p008
    assert re.search(r'Qwen2\.5-32B-Instruct \(<a class="citation" href="#reference-8".*?</a>\)', p027, flags=re.S)
    assert "high concordance" not in p108
    assert "agreement varying by rubric and setting" in p108
    assert all(phrase in p114 for phrase in [
        "follow reporting guidance for model and evaluation procedures",
        "use structured approaches to human evaluation",
        "evaluate extraction accuracy against expert reference standards",
    ])
    assert "Fourteen entries in this workspace are cited in the manuscript" in document
    assert "Findings of the Association for Computational Linguistics: EMNLP 2025" in document
    assert "Proceedings of the 23rd Workshop on Biomedical Natural Language Processing" in document
    assert parser.literature_entries == [
        *[str(index) for index in sorted(renderer.USED_LITERATURE_ENTRY_NUMBERS)],
        *[
            str(index) for index in range(1, len(renderer.LITERATURE_ENTRIES) + 1)
            if index not in renderer.USED_LITERATURE_ENTRY_NUMBERS
        ],
    ]

    for path in [*renderer.MAIN_FIGURES.values(), *renderer.SUPPLEMENTARY_FIGURES.values()]:
        assert path.exists() and path.suffix == ".svg"
    assert document.count('class="figure-screen"') == 6
    assert document.count('class="figure-print"') == 6
    assert 'data-source-block="P080"' in document
    assert '<p class="physician-paragraph" id="P080"' not in document
    print(
        "V2_VALIDATION_OK "
        "paragraphs=122 cells=107 comments=9 figures=6 tables=8 underlines=24"
    )


if __name__ == "__main__":
    main()
