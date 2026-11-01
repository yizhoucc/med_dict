#!/usr/bin/env python3
"""Deterministic v1.5.2 manuscript and token-level redline checks."""

from __future__ import annotations

import ast
import difflib
import hashlib
import html
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import render_workshop_draft as renderer  # noqa: E402


def committed_renderer_paragraphs(commit: str) -> dict[str, str]:
    source = subprocess.check_output(
        ["git", "show", f"{commit}:render_workshop_draft.py"],
        cwd=ROOT,
        text=True,
    )
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "REVISED_PARAGRAPHS" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"REVISED_PARAGRAPHS is missing at {commit}")


def introduction_diff_metrics() -> dict[str, int]:
    old = committed_renderer_paragraphs("17e09d5d")
    citation = re.compile(r"\s*\[\[cite:[a-z_]+\]\]")
    token = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
    old_text = " ".join(citation.sub("", old[block_id]) for block_id in renderer.INTRODUCTION_IDS)
    new_text = " ".join(citation.sub("", renderer.REVISED_PARAGRAPHS[block_id]) for block_id in renderer.INTRODUCTION_IDS)
    old_words = [word.lower() for word in token.findall(old_text)]
    new_words = [word.lower() for word in token.findall(new_text)]
    opcodes = difflib.SequenceMatcher(None, old_words, new_words).get_opcodes()
    return {
        "old_words": len(old_words),
        "new_words": len(new_words),
        "inserted_words": sum(j2 - j1 for kind, _, _, j1, j2 in opcodes if kind == "insert"),
        "replacement_words": sum(j2 - j1 for kind, _, _, j1, j2 in opcodes if kind == "replace"),
        "deleted_words": sum(i2 - i1 for kind, i1, i2, _, _ in opcodes if kind == "delete"),
        "replaced_old_words": sum(i2 - i1 for kind, i1, i2, _, _ in opcodes if kind == "replace"),
    }


def result_hash(document: str) -> tuple[int, str]:
    blocks = renderer.collect_review_blocks(document)
    text = "\n".join(f"{block_id}\t{block.text}" for block_id, block in blocks.items() if block.section == "results")
    return sum(block.section == "results" for block in blocks.values()), hashlib.sha256(text.encode("utf-8")).hexdigest()


def revision_span_metrics(document: str) -> dict[str, object]:
    matches = re.findall(
        r'<span (?P<attrs>[^>]*class="[^"]*\brevision-highlight\b[^"]*"[^>]*)>(?P<body>.*?)</span>',
        document,
        flags=re.S,
    )
    rows = []
    for attrs, body in matches:
        block = re.search(r'data-review-for="([^"]+)"', attrs)
        assert block
        text = html.unescape(re.sub(r"<[^>]+>", "", body)).strip()
        change = re.search(r'data-review-change="([^"]+)"', attrs)
        rows.append((html.unescape(block.group(1)), text, change.group(1) if change else "", body))
    return {
        "rows": rows,
        "count": len(rows),
        "blocks": {row[0] for row in rows},
        "max_length": max(len(row[1]) for row in rows),
    }


def main() -> None:
    document = (ROOT / "WORKSHOP_PAPER_DRAFT.html").read_text(encoding="utf-8")
    state = renderer.review_redline_state(document)
    v15_document = subprocess.check_output(
        ["git", "show", "744986d9:WORKSHOP_PAPER_DRAFT.html"],
        cwd=ROOT,
        text=True,
    )
    v15_blocks = renderer.collect_review_blocks(v15_document)
    current_blocks = renderer.collect_review_blocks(document)
    assert {block_id: block.text for block_id, block in current_blocks.items()} == {
        block_id: block.text for block_id, block in v15_blocks.items()
    }
    assert state["changed"] == state["highlighted"]
    assert not [block_id for block_id in state["highlighted"] if block_id not in state["scoped_current"]]
    assert not [block_id for block_id in state["changed"] if state["current"][block_id].section == "results"]
    assert len(state["scoped_current"]) == 342
    assert len(state["changed"]) == len(state["highlighted"]) == 44
    expected_importance = {level: set(block_ids) for level, block_ids in renderer.REVIEW_IMPORTANCE.items()}
    assert state["importance"] == expected_importance
    assert {level: len(block_ids) for level, block_ids in state["importance"].items()} == {
        "light": 20,
        "medium": 12,
        "strong": 12,
    }
    assert set().union(*state["importance"].values()) == state["changed"]
    assert sum(len(block_ids) for block_ids in state["importance"].values()) == len(state["changed"])
    assert not [block_id for block_id, block in state["current"].items() if block.importance and block_id not in state["changed"]]
    assert state["section_counts"] == {
        "discussion": 2,
        "introduction": 5,
        "methods": 15,
        "supplementary-material": 1,
        "supplementary-methods": 21,
    }
    assert "P117" in state["changed"] == state["highlighted"]

    spans = revision_span_metrics(document)
    assert spans["count"] == 242
    assert len(spans["blocks"]) == 66
    assert spans["max_length"] == 79
    assert not [row for row in spans["rows"] if "external-citation" in row[3] or "comment-anchor" in row[3]]
    assert not re.findall(
        r'<(?:p|h3|h4|li|td|th|figcaption)[^>]*class="[^"]*\brevision-highlight\b',
        document,
    )
    for paragraph_id in ["P004", "P006", "P008", "P010", "P012"]:
        opening = re.search(rf'<p[^>]*id="{paragraph_id}"[^>]*>', document)
        assert opening and "revision-highlight" not in opening.group(0)
        assert paragraph_id in spans["blocks"]
    assert "figure-s1::caption" not in spans["blocks"]
    assert "figure-s2::caption" in spans["blocks"]
    required_focus = {
        "not used for development",
        "not provided to the model",
        "primary physician-preference endpoint",
        "A displayed the inference-harness output",
        "B displayed the single-prompt baseline",
        "not shown these identities",
        "Source attribution was available only with the harness output",
        "complete system outputs",
        "same base-model family",
        "did not evaluate generalization across model families",
    }
    assert required_focus.issubset({row[1] for row in spans["rows"]})

    metrics = introduction_diff_metrics()
    inserted_or_replaced = metrics["inserted_words"] + metrics["replacement_words"]
    # The approved final text is immutable in this check. Under the documented
    # token metric it contributes 71 inserted or replacement-side words, close
    # to the editorial estimate of about 60 words.
    assert 65 <= inserted_or_replaced <= 75, metrics

    old_document = subprocess.check_output(
        ["git", "show", "17e09d5d:WORKSHOP_PAPER_DRAFT.html"],
        cwd=ROOT,
        text=True,
    )
    old_result_count, old_result_hash = result_hash(old_document)
    new_result_count, new_result_hash = result_hash(document)
    assert (old_result_count, old_result_hash) == (new_result_count, new_result_hash)

    print(
        "V152_REVIEW_OK "
        f"baseline_blocks={state['baseline']['block_count']} "
        f"current_blocks={len(state['current'])} "
        f"changed_blocks={len(state['changed'])} token_spans={spans['count']} span_blocks={len(spans['blocks'])} "
        f"deleted={len(state['deleted'])} importance="
        f"{dict(sorted((level, len(block_ids)) for level, block_ids in state['importance'].items()))} "
        f"sections={state['section_counts']}"
    )
    print(f"V152_WORDING_OK visible_blocks={len(current_blocks)} unchanged_from=744986d9")
    print(
        "V152_INTRO_OK "
        f"old_words={metrics['old_words']} new_words={metrics['new_words']} "
        f"inserted_or_replaced={inserted_or_replaced} metrics={metrics}"
    )
    print(f"V152_RESULTS_OK blocks={new_result_count} sha256={new_result_hash}")


if __name__ == "__main__":
    main()
