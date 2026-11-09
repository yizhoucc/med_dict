#!/usr/bin/env python3
"""Compatibility checks for the current physician manuscript renderer."""

from __future__ import annotations

if __name__ == "__main__":
    from validate_workshop_v20 import main as _v20_main
    _v20_main()
    raise SystemExit(0)

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
    current_blocks = renderer.collect_review_blocks(document)
    assert state["changed"] == state["highlighted"]
    assert not [block_id for block_id in state["highlighted"] if block_id not in state["scoped_current"]]
    assert len(state["changed"]) == len(state["highlighted"]) == 71
    assert state["section_counts"] == {
        "discussion": 3,
        "introduction": 5,
        "methods": 29,
        "results": 9,
        "supplementary-material": 4,
        "supplementary-methods": 21,
    }
    assert "P117" in state["changed"] == state["highlighted"]

    assert "revision-highlight" not in document
    focus_spans = re.findall(r'<span [^>]*class="[^"]*\bfocus-highlight\b[^"]*"[^>]*>', document)
    assert len(focus_spans) == 57
    assert all(re.search(r'data-review-intensity="(?:light|medium|strong)"', span) for span in focus_spans)
    assert not re.findall(r'<(?:p|h3|h4|li|td|th|figcaption)[^>]*class="[^"]*\bfocus-highlight\b', document)

    new_result_count, new_result_hash = result_hash(document)
    assert (new_result_count, new_result_hash) == (124, renderer.EXPECTED_V180_RESULTS_SHA256)
    for section, next_section in [("introduction", "methods"), ("methods", "results")]:
        assert hashlib.sha256(renderer.section_markup(document, section, next_section).encode()).hexdigest() == renderer.EXPECTED_V180_SECTION_LOCKS[section]["markup_sha256"]
        assert hashlib.sha256(renderer.section_block_text(document, section).encode()).hexdigest() == renderer.EXPECTED_V180_SECTION_LOCKS[section]["text_sha256"]

    print(
        "CURRENT_REVIEW_OK "
        f"baseline_blocks={state['baseline']['block_count']} "
        f"current_blocks={len(state['current'])} "
        f"changed_blocks={len(state['changed'])} focus_spans={len(focus_spans)} "
        f"deleted={len(state['deleted'])} "
        f"sections={state['section_counts']}"
    )
    print(f"CURRENT_WORDING_OK visible_blocks={len(current_blocks)}")
    print("V180_INTRO_METHODS_LOCK_OK")
    print(f"V180_RESULTS_OK blocks={new_result_count} sha256={new_result_hash}")


if __name__ == "__main__":
    main()
