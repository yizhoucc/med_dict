#!/usr/bin/env python3
"""Render WORKSHOP_PAPER_DRAFT.md as a self-contained, readable HTML page.

Run from the repository root:
    python3 render_workshop_draft.py

The renderer uses the locally installed CommonMark ``cmark`` command, adds GFM-like
table handling, and embeds available draft SVG plots as data URIs. No network
access is required.
"""

from __future__ import annotations

import base64
import html
import re
import shutil
import subprocess
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "WORKSHOP_PAPER_DRAFT.md"
OUTPUT = ROOT / "WORKSHOP_PAPER_DRAFT.html"
FIGURE_DIR = ROOT / "results/extraction_comparison/clinician_ratings/draft_figures"

FIGURES = {
    "1": (
        FIGURE_DIR / "figure1_development_pathway_rough.svg",
        "Physician-guided development, model-assisted adaptation, and extraction evaluation",
    ),
    "2": (FIGURE_DIR / "figure2_evaluator_distribution_rough.svg", "Physician preference by evaluation"),
    "3": (
        FIGURE_DIR / "figure3_interrater_matrix_rough.svg",
        "Pairwise physician agreement in breast and pancreatic cancer",
    ),
    "4": (
        FIGURE_DIR / "figure4_effect_profile.svg",
        "Effect profile across fields, cancers, and notes",
    ),
    "5": (
        FIGURE_DIR / "figure5_staged_ablation.svg",
        "Blinded staged technical ablation overall and by cancer",
    ),
    "6": (
        FIGURE_DIR / "figure6_clinical_handoff.svg",
        "Conceptual physician-model complementarity across extraction tasks",
    ),
}

SUPPLEMENTARY_FIGURES = {
    "S1": (
        FIGURE_DIR / "supplementary_figure_s1_note_margins.svg",
        "Detailed note-level preference margins pooled across four physicians",
    ),
}


CSS = r"""
:root {
  --ink: #17212b;
  --muted: #5f6b76;
  --navy: #123b5d;
  --blue: #2b8cbe;
  --blue-soft: #eaf5fb;
  --orange: #d95f0e;
  --paper: #ffffff;
  --page: #edf1f5;
  --line: #d8e0e7;
  --violet: #6750a4;
  --violet-soft: #f4f0fb;
  --warning: #8a5a00;
  --warning-soft: #fff7df;
  --confirm: #a23b3b;
  --confirm-soft: #fff1f0;
  --suggest: #8a5a00;
  --suggest-soft: #fff8e6;
  --verified: #24613b;
  --verified-soft: #edf8f0;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: var(--page);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  line-height: 1.62;
}
.layout {
  display: grid;
  grid-template-columns: 270px minmax(0, 920px);
  gap: 28px;
  width: min(1240px, calc(100% - 36px));
  margin: 24px auto 70px;
  align-items: start;
}
.review-checklist {
  margin: 20px 0 30px;
  padding: 18px 20px;
  border: 1px solid #d7c8ee;
  border-left: 5px solid var(--violet);
  border-radius: 10px;
  background: var(--violet-soft);
}
.review-checklist h2 {
  margin: 0 0 10px;
  color: #4f3b7d;
  font-size: 1.08rem;
}
.review-checklist ul { margin: 0; padding-left: 1.4rem; }
.review-checklist li { margin: 0.35rem 0; }
.review-checklist .done { color: #24613b; }
.review-checklist .open { color: #825600; }
.chinese-version .author-confirmation-index {
  margin: 24px 0 34px;
  padding: 20px 22px;
  border: 1px solid #c9d6e0;
  border-left: 6px solid var(--navy);
  border-radius: 11px;
  background: #f7fafc;
}
.chinese-version .author-confirmation-index h2,
.chinese-version .author-confirmation-index .author-confirmation-title {
  margin: 0 0 8px;
  padding: 0;
  border: 0;
  font-size: 1.3rem;
  font-weight: 750;
}
.chinese-version .author-confirmation-index > p {
  margin: 6px 0 14px;
  color: var(--muted);
}
.chinese-version .author-confirmation-index ol { margin: 0; padding-left: 1.6rem; }
.chinese-version .author-confirmation-index li { margin: 0.48rem 0; }
.chinese-version .author-confirmation-index a {
  color: var(--navy);
  text-decoration-thickness: 1px;
  text-underline-offset: 2px;
}
.chinese-version .status-badge {
  display: inline-block;
  min-width: 5.8em;
  margin-right: 0.45em;
  padding: 1px 7px;
  border: 1px solid currentColor;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 750;
  line-height: 1.55;
  text-align: center;
  vertical-align: 0.08em;
}
.chinese-version .badge-critical { color: var(--confirm); background: var(--confirm-soft); }
.chinese-version .badge-suggested { color: var(--suggest); background: var(--suggest-soft); }
.chinese-version .badge-verified { color: var(--verified); background: var(--verified-soft); }
.chinese-version .author-note {
  width: 100%;
  margin: 18px 0 24px;
  padding: 15px 17px;
  border: 1px solid #d7dee5;
  border-left: 5px solid var(--navy);
  border-radius: 9px;
  background: #fbfcfd;
  scroll-margin-top: 18px;
  break-inside: avoid;
}
.chinese-version .author-note.severity-critical {
  border-left-color: var(--confirm);
  background: var(--confirm-soft);
}
.chinese-version .author-note.severity-suggested {
  border-left-color: var(--suggest);
  background: var(--suggest-soft);
}
.chinese-version .author-note.severity-verified {
  border-left-color: var(--verified);
  background: var(--verified-soft);
}
.chinese-version .author-note:target {
  outline: 3px solid rgba(43, 140, 190, 0.28);
  outline-offset: 3px;
}
.chinese-version .author-note h4 {
  margin: 7px 0 8px;
  color: #20394d;
  font-size: 1rem;
}
.chinese-version .author-note p { margin: 6px 0; }
.chinese-version .author-note .back-link {
  display: inline-block;
  margin-top: 5px;
  color: #4a6071;
  font-size: 0.8rem;
}
.toc {
  position: sticky;
  top: 18px;
  max-height: calc(100vh - 36px);
  overflow: auto;
  padding: 18px 16px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 4px 18px rgba(20, 43, 64, 0.06);
}
.toc-title { margin: 0 0 10px; color: var(--navy); font-size: 15px; font-weight: 750; }
.toc a { display: block; color: #34495a; text-decoration: none; padding: 4px 6px; border-radius: 6px; font-size: 12.5px; }
.toc a:hover { color: var(--navy); background: var(--blue-soft); }
.toc a.level-3 { padding-left: 18px; color: var(--muted); font-size: 11.5px; }
.toc-actions { display: flex; gap: 8px; margin: 0 0 14px; }
.toc button {
  border: 1px solid #aab8c4;
  border-radius: 7px;
  background: white;
  color: var(--navy);
  padding: 6px 10px;
  cursor: pointer;
  font-weight: 650;
}
article {
  min-width: 0;
  padding: 46px 54px 70px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: var(--paper);
  box-shadow: 0 12px 35px rgba(20, 43, 64, 0.08);
}
h1 {
  margin: 0 0 12px;
  color: var(--navy);
  font-family: Georgia, "Times New Roman", serif;
  font-size: 35px;
  line-height: 1.18;
  letter-spacing: -0.025em;
}
h2 {
  margin: 40px 0 14px;
  padding-bottom: 7px;
  border-bottom: 2px solid #dce8f0;
  color: var(--navy);
  font-family: Georgia, "Times New Roman", serif;
  font-size: 25px;
  line-height: 1.25;
}
h3 {
  margin: 28px 0 10px;
  color: #214f70;
  font-size: 18px;
  line-height: 1.3;
}
p { margin: 10px 0; }
ul, ol { margin: 9px 0 14px; padding-left: 26px; }
li { margin: 5px 0; }
strong { color: #102f47; }
em { color: #34495a; }
code {
  padding: 1px 5px;
  border: 1px solid #dce4ea;
  border-radius: 5px;
  background: #f4f7f9;
  color: #23465e;
  font-family: "SFMono-Regular", Consolas, monospace;
  font-size: 0.88em;
}
a { color: #176a9a; }
.citation-link {
  color: #125d89;
  font-weight: 700;
  text-decoration: none;
}
.citation-link:hover { text-decoration: underline; }
.draft-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 16px 0 24px;
}
.chip {
  padding: 5px 10px;
  border-radius: 999px;
  background: #e8f2f8;
  color: var(--navy);
  font-size: 12px;
  font-weight: 700;
}
.render-note {
  margin: 18px 0 26px;
  padding: 12px 15px;
  border-left: 4px solid var(--violet);
  border-radius: 7px;
  background: var(--violet-soft);
  color: #433767;
  font-size: 13px;
}
.language-nav {
  display: flex;
  gap: 10px;
  margin: 0 0 22px;
}
.language-nav a {
  padding: 5px 11px;
  border: 1px solid #b8cad6;
  border-radius: 999px;
  background: #f7fbfd;
  color: var(--navy);
  font-size: 12px;
  font-weight: 700;
  text-decoration: none;
}
.language-break {
  margin: 70px -20px 42px;
  border-top: 4px solid var(--navy);
  padding-top: 34px;
}
.abstract-section {
  margin: 26px -18px 32px;
  border: 1px solid #cfe0ec;
  border-left: 5px solid var(--blue);
  border-radius: 10px;
  background: #f7fbfd;
}
.top-level-section {
  margin: 34px 0 0;
  scroll-margin-top: 18px;
}
.top-level-section > summary {
  padding: 9px 4px 10px;
  color: var(--navy);
  cursor: pointer;
}
.top-level-section > summary::marker { color: #6a7f8f; }
.top-level-section > summary:hover h2 { color: #176a9a; }
.top-level-section > summary:focus-visible {
  border-radius: 6px;
  outline: 3px solid rgba(43, 140, 190, 0.32);
  outline-offset: 3px;
}
.top-level-section > summary h2 {
  display: inline;
  margin: 0;
  padding: 0;
  border: 0;
  color: inherit;
  font-size: 25px;
}
.section-state {
  float: right;
  margin: 7px 2px 0 14px;
  color: #6b7882;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 0.72rem;
  font-weight: 650;
}
.top-level-section:not([open]) > summary .section-state-open,
.top-level-section[open] > summary .section-state-closed { display: none; }
.main-section {
  border-top: 1px solid #dce5eb;
}
.main-section[open] > summary {
  border-bottom: 2px solid #dce8f0;
}
.main-section > .section-body { padding-top: 4px; }
.abstract-section > summary { padding: 14px 18px 10px; }
.abstract-section[open] > summary { border-bottom: 0; }
.abstract-section > .section-body { padding: 0 18px 18px; }
.appendix-section,
.references-section,
.auxiliary-section {
  border: 1px solid #d5dee5;
  border-radius: 9px;
  background: #fafcfd;
}
.appendix-section > summary,
.references-section > summary,
.auxiliary-section > summary { padding: 12px 15px; }
.appendix-section[open] > summary,
.references-section[open] > summary,
.auxiliary-section[open] > summary { border-bottom: 1px solid #dde5ea; }
.appendix-section > .section-body,
.references-section > .section-body,
.auxiliary-section > .section-body { padding: 3px 16px 18px; }
.references-section { color: #44515c; font-size: 13px; }
.references-section > summary h2 { font-size: 22px; }
.judge-appendix { padding-top: 0; }
.chinese-review-meta {
  margin: 18px 0 32px;
  border: 1px solid #c9d6e0;
  border-radius: 10px;
  background: #f8fbfd;
}
.chinese-review-meta > summary {
  padding: 14px 17px;
  color: var(--navy);
  cursor: pointer;
  font-weight: 750;
}
.chinese-review-meta[open] > summary { border-bottom: 1px solid #d5e0e8; }
.chinese-review-meta:not([open]) .details-label-open,
.chinese-review-meta[open] .details-label-closed { display: none; }
.chinese-review-meta-body { padding: 4px 18px 18px; }
.chinese-review-meta .author-confirmation-index {
  margin: 18px 0 0;
  background: white;
}
.table-wrap {
  width: 100%;
  margin: 18px 0 24px;
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 9px;
}
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 9px 11px; border-bottom: 1px solid #e4eaef; text-align: left; vertical-align: top; }
th { background: #eef5f9; color: var(--navy); font-weight: 750; }
tbody tr:nth-child(even) { background: #fafcfd; }
tbody tr:last-child td { border-bottom: 0; }
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.manuscript-figure {
  margin: 25px 0 8px;
  padding: 14px;
  border: 1px solid #cad7df;
  border-radius: 10px;
  background: #fbfdfe;
  text-align: center;
}
.manuscript-figure img { display: block; width: 100%; height: auto; margin: 0 auto; }
.supplementary-figure { background: #f8fafb; }
.judge-metadata {
  margin: 14px 0 18px;
  padding: 13px 15px;
  border: 1px solid #d7c8ee;
  border-left: 5px solid var(--violet);
  border-radius: 8px;
  background: var(--violet-soft);
}
.judge-metadata p { margin: 0; }
.prompt-stage {
  margin: 10px 0;
  border: 1px solid #d8e0e7;
  border-radius: 8px;
  background: #fbfcfd;
}
.prompt-stage > summary {
  padding: 11px 14px;
  color: #314a5d;
  cursor: pointer;
  font-size: 0.92rem;
  font-weight: 720;
}
.prompt-stage[open] > summary { border-bottom: 1px solid #e1e7eb; }
.prompt-stage .prompt-body { padding: 0 13px 13px; }
.prompt-stage pre {
  max-width: 100%;
  margin: 12px 0 0;
  padding: 13px 14px;
  overflow-x: hidden;
  border: 1px solid #e0e5e9;
  border-radius: 7px;
  background: white;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  font-size: 0.78rem;
  line-height: 1.48;
}
.prompt-stage pre code {
  padding: 0;
  border: 0;
  background: transparent;
  white-space: inherit;
}
.audit-artifacts {
  margin: 16px 0 0;
  padding: 10px 13px;
  border-radius: 7px;
  background: #f3f6f8;
  font-size: 0.86rem;
}
.todo {
  padding: 11px 14px;
  border: 1px solid #eed48a;
  border-radius: 8px;
  background: var(--warning-soft);
  color: var(--warning);
  font-weight: 650;
}
.references-section li {
  margin: 0 0 13px;
  padding-left: 3px;
  scroll-margin-top: 20px;
}
.references-section li:target {
  padding: 8px 10px 8px 3px;
  border-radius: 7px;
  background: var(--blue-soft);
}
.reference-note {
  display: block;
  margin-top: 4px;
  color: #5b6670;
  line-height: 1.5;
}
.english-manuscript-details {
  margin: 0 0 44px;
  border: 1px solid #b9c9d6;
  border-radius: 10px;
  background: #f8fbfd;
}
.english-manuscript-details > summary {
  padding: 15px 18px;
  color: var(--navy);
  cursor: pointer;
  font-size: 1rem;
  font-weight: 750;
}
.english-manuscript-details[open] > summary {
  border-bottom: 1px solid #d5e0e8;
}
.english-manuscript-details:not([open]) .details-label-open,
.english-manuscript-details[open] .details-label-closed {
  display: none;
}
.english-manuscript-details > .english-version {
  padding: 0 18px 20px;
  background: white;
}
@media (max-width: 980px) {
  .layout { display: block; width: min(920px, calc(100% - 22px)); margin-top: 11px; }
  .toc { position: relative; top: 0; max-height: none; margin-bottom: 12px; }
  .toc a { display: none; }
  article { padding: 30px 24px 55px; }
  h1 { font-size: 29px; }
  .top-level-section > summary h2 { font-size: 22px; }
  .section-state { margin-top: 5px; }
  .abstract-section { margin-left: -8px; margin-right: -8px; }
  .chinese-version .author-confirmation-index,
  .chinese-version .author-note { width: 100%; }
}
@media print {
  @page { size: letter; margin: 16mm; }
  body { background: white; font-size: 10.5pt; }
  .layout { display: block; width: auto; margin: 0; }
  .toc { display: none; }
  article { border: 0; box-shadow: none; padding: 0; }
  h1 { font-size: 22pt; }
  h2 { page-break-after: avoid; font-size: 16pt; }
  h3 { page-break-after: avoid; font-size: 13pt; }
  .manuscript-figure, .table-wrap { break-inside: avoid; }
  .manuscript-figure img { max-height: 650px; object-fit: contain; }
  .english-manuscript-details { margin: 0; border: 0; background: white; }
  .english-manuscript-details > summary { display: none; }
  .english-manuscript-details:not([open]) > .english-version {
    display: block !important;
  }
  .top-level-section { margin: 22px 0 0; border: 0; background: white; }
  .top-level-section > summary {
    display: block;
    padding: 0 0 7px;
    border-bottom: 2px solid #dce8f0;
    cursor: default;
    list-style: none;
    page-break-after: avoid;
    break-after: avoid-page;
  }
  .top-level-section > summary::marker { content: ""; }
  .top-level-section > summary::-webkit-details-marker { display: none; }
  .top-level-section > summary h2 {
    display: block;
    margin: 0;
    color: var(--navy);
    font-size: 16pt;
  }
  .top-level-section > summary .section-state { display: none !important; }
  .top-level-section:not([open]) > .section-body {
    display: block !important;
  }
  .chinese-review-meta { margin: 0; border: 0; background: white; }
  .chinese-review-meta > summary,
  .prompt-stage > summary { display: none; }
  .chinese-review-meta:not([open]) > .chinese-review-meta-body,
  .prompt-stage:not([open]) > .prompt-body { display: block !important; }
  .prompt-stage { border: 0; break-inside: auto; }
  .prompt-stage .prompt-body { padding: 0; }
  .chinese-version .author-confirmation-index,
  .chinese-version .author-note { width: 100%; break-inside: avoid; }
}
"""


def inline_markup(value: str) -> str:
    """Render the small inline Markdown subset used inside tables."""
    escaped = html.escape(value.strip())
    code_tokens: list[str] = []

    def protect_code(match: re.Match[str]) -> str:
        code_tokens.append(f"<code>{match.group(1)}</code>")
        return f"@@CODE{len(code_tokens) - 1}@@"

    escaped = re.sub(r"`([^`]+)`", protect_code, escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        escaped,
    )
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", escaped)
    for index, token in enumerate(code_tokens):
        escaped = escaped.replace(f"@@CODE{index}@@", token)
    return escaped


def split_table_row(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def numeric_cell(value: str) -> bool:
    stripped = value.strip().replace("**", "").replace("`", "")
    return bool(re.fullmatch(r"[+\-−]?\d[\d,]*(?:\.\d+)?%?", stripped))


def is_table_separator(line: str) -> bool:
    cells = split_table_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells)


def convert_tables(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    index = 0
    while index < len(lines):
        if (
            index + 1 < len(lines)
            and "|" in lines[index]
            and is_table_separator(lines[index + 1])
        ):
            headers = split_table_row(lines[index])
            index += 2
            rows: list[list[str]] = []
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(split_table_row(lines[index]))
                index += 1
            output.append('<div class="table-wrap"><table><thead><tr>')
            output.extend(f"<th>{inline_markup(cell)}</th>" for cell in headers)
            output.append("</tr></thead><tbody>")
            for row in rows:
                padded = row + [""] * max(0, len(headers) - len(row))
                output.append("<tr>")
                output.extend(
                    f'<td class="num">{inline_markup(cell)}</td>'
                    if numeric_cell(cell)
                    else f"<td>{inline_markup(cell)}</td>"
                    for cell in padded[: len(headers)]
                )
                output.append("</tr>")
            output.append("</tbody></table></div>")
            continue
        output.append(lines[index])
        index += 1
    return "\n".join(output) + "\n"


def run_cmark(markdown_text: str) -> str:
    command = shutil.which("cmark")
    if command is None:
        fallback = Path("/Users/yizhoucc/Library/Python/3.9/bin/cmark")
        if fallback.exists():
            command = str(fallback)
    if command is None:
        raise RuntimeError("The CommonMark cmark command is required to render the draft.")
    completed = subprocess.run(
        [command],
        input=markdown_text,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout


def format_prompt_appendices(body: str) -> str:
    """Turn the general judge protocol into a compact native disclosure.

    This runs before heading IDs and TOC construction, so prompt labels do not
    appear as manuscript sections in the navigation. The ``pre`` contents are
    moved without editing their text.
    """

    appendix_pattern = re.compile(
        r"(<h2>(?:Appendix S1\. General blinded Codex judge protocol|"
        r"附录 S1\. General blinded Codex judge protocol)</h2>)"
        r"(.*?)"
        r"(?=<h2>(?:Appendix S2\.[^<]*|附录 S2\.[^<]*|References|参考文献)</h2>)",
        flags=re.S,
    )

    def format_appendix(match: re.Match[str]) -> str:
        heading, content = match.groups()
        first_prompt = content.find("<h3>")
        if first_prompt < 0:
            raise RuntimeError("Appendix S1 prompt headings were not found.")
        metadata = content[:first_prompt].strip()
        prompts_and_links = content[first_prompt:]
        prompts_and_links = re.sub(
            r"<h3>(.*?)</h3>\s*(<pre>.*?</pre>)",
            lambda prompt: (
                '<details class="prompt-stage"><summary>'
                + prompt.group(1)
                + '</summary><div class="prompt-body">'
                + prompt.group(2)
                + "</div></details>"
            ),
            prompts_and_links,
            flags=re.S,
        )
        prompts_and_links = re.sub(
            r"<p>(Audit artifacts?:|审计 artifacts?：)(.*?)</p>",
            r'<p class="audit-artifacts"><strong>\1</strong>\2</p>',
            prompts_and_links,
            flags=re.S,
        )
        prompt_count = prompts_and_links.count('<details class="prompt-stage">')
        if prompt_count != 1:
            raise RuntimeError(
                f"Expected one folded judge protocol in Appendix S1, found {prompt_count}."
            )
        return (
            heading
            + '<div class="judge-metadata">'
            + metadata
            + "</div>"
            + prompts_and_links
        )

    formatted, count = appendix_pattern.subn(format_appendix, body)
    if count != 2:
        raise RuntimeError(f"Expected English and Chinese Appendix S1 sections, found {count}.")
    return formatted


def slugify(text: str) -> str:
    plain = html.unescape(re.sub(r"<[^>]+>", "", text)).lower()
    plain = re.sub(r"[_\W]+", "-", plain, flags=re.UNICODE).strip("-")
    return plain or "section"


def add_heading_ids(body: str) -> tuple[str, str]:
    headings: list[tuple[int, str, str]] = []
    seen: Counter[str] = Counter()

    def replace(match: re.Match[str]) -> str:
        level = int(match.group(1))
        label = match.group(2)
        slug = slugify(label)
        seen[slug] += 1
        if seen[slug] > 1:
            slug = f"{slug}-{seen[slug]}"
        headings.append((level, slug, re.sub(r"<[^>]+>", "", label)))
        return f'<h{level} id="{slug}">{label}</h{level}>'

    body = re.sub(r"<h([23])>(.*?)</h\1>", replace, body, flags=re.S)
    toc_lines = [
        '<nav class="toc">',
        '<div class="toc-title">Workshop draft</div>',
        '<div class="toc-actions">Use the browser print command to save a PDF.</div>',
    ]
    for level, slug, label in headings:
        toc_lines.append(f'<a class="level-{level}" href="#{slug}">{html.escape(label)}</a>')
    toc_lines.append("</nav>")
    return body, "\n".join(toc_lines)


def embed_svg(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def inject_figures(body: str) -> str:
    for key, (path, title) in FIGURES.items():
        marker = f'<div data-rough-figure="{key}"></div>'
        if marker not in body or not path.exists():
            continue
        figure = f'''<figure class="manuscript-figure">
<img src="{embed_svg(path)}" alt="{html.escape(title)}">
</figure>'''
        body = body.replace(marker, figure)
    for key, (path, title) in SUPPLEMENTARY_FIGURES.items():
        marker = f'<div data-supplementary-figure="{key}"></div>'
        if marker not in body or not path.exists():
            continue
        figure = f'''<figure class="manuscript-figure supplementary-figure">
<img src="{embed_svg(path)}" alt="{html.escape(title)}">
</figure>'''
        body = body.replace(marker, figure)
    return body


def link_citation_markers(text: str, reference_prefix: str) -> str:
    """Link numeric citation groups such as [2] and [3,8] to reference entries."""

    def replace(match: re.Match[str]) -> str:
        numbers = [value.strip() for value in match.group(1).split(",")]
        links = [
            f'<a class="citation-link" href="#{reference_prefix}-{number}">{number}</a>'
            for number in numbers
        ]
        return "[" + ",".join(links) + "]"

    return re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", replace, text)


def add_reference_ids(reference_section: str, reference_prefix: str) -> tuple[str, int]:
    """Assign stable anchors to the first ordered list in a reference section."""

    list_match = re.search(r"<ol>(.*?)</ol>", reference_section, flags=re.S)
    if list_match is None:
        raise RuntimeError(f"Reference list not found for {reference_prefix}.")

    count = 0

    def replace_item(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return f'<li id="{reference_prefix}-{count}">{match.group(1)}</li>'

    numbered_list = re.sub(r"<li>(.*?)</li>", replace_item, list_match.group(0), flags=re.S)
    return (
        reference_section[: list_match.start()]
        + numbered_list
        + reference_section[list_match.end() :],
        count,
    )


def link_citations_and_references(body: str) -> tuple[str, tuple[int, int]]:
    """Add language-specific reference anchors and citation hyperlinks."""

    english_heading = '<h2 id="references">References</h2>'
    chinese_marker = '<div class="language-break" id="chinese-version"></div>'
    chinese_heading = '<h2 id="参考文献">参考文献</h2>'

    english_reference_start = body.index(english_heading)
    chinese_start = body.index(chinese_marker)
    chinese_reference_start = body.index(chinese_heading, chinese_start)

    english_body = link_citation_markers(body[:english_reference_start], "ref-en")
    english_references, english_count = add_reference_ids(
        body[english_reference_start:chinese_start], "ref-en"
    )
    chinese_body = link_citation_markers(
        body[chinese_start:chinese_reference_start], "ref-zh"
    )
    chinese_references, chinese_count = add_reference_ids(
        body[chinese_reference_start:], "ref-zh"
    )
    chinese_references = link_citation_markers(chinese_references, "ref-zh")

    return (
        english_body + english_references + chinese_body + chinese_references,
        (english_count, chinese_count),
    )


def style_sections(body: str) -> str:
    body = re.sub(
        r"<p>(\[(?:FINAL|TODO|最终|投稿前)[^<]+\])</p>",
        r'<p class="todo">\1</p>',
        body,
    )
    return body


def fold_top_level_sections(body: str) -> str:
    """Wrap every manuscript h2 section in an accessible native disclosure."""

    marker = '<div class="language-break" id="chinese-version"></div>'
    english_body, chinese_body = body.split(marker, 1)

    def fold_language(language_body: str, language: str) -> str:
        headings = list(
            re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', language_body, flags=re.S)
        )
        if not headings:
            raise RuntimeError(f"No top-level headings found for {language}.")

        output = [language_body[: headings[0].start()]]
        for index, heading in enumerate(headings):
            end = headings[index + 1].start() if index + 1 < len(headings) else len(language_body)
            section_id, heading_html = heading.groups()
            label = html.unescape(re.sub(r"<[^>]+>", "", heading_html)).strip()
            content = language_body[heading.end() : end]

            is_main = label in {"Abstract", "摘要"} or bool(re.match(r"^\d+\.", label))
            is_appendix = label.startswith("Appendix ") or label.startswith("附录 ")
            is_references = label in {"References", "参考文献"}
            is_version_history = label in {"Version history", "版本历史"}

            if is_main:
                kind = "main"
                classes = ["top-level-section", "main-section"]
                if label in {"Abstract", "摘要"}:
                    classes.append("abstract-section")
            elif is_appendix:
                kind = "appendix"
                classes = ["top-level-section", "appendix-section"]
                if re.match(r"^(?:Appendix|附录) S1\.", label):
                    classes.append("judge-appendix")
            elif is_references:
                kind = "references"
                classes = ["top-level-section", "references-section"]
            elif is_version_history:
                kind = "auxiliary"
                classes = [
                    "top-level-section",
                    "auxiliary-section",
                    "version-history-section",
                ]
            else:
                kind = "auxiliary"
                classes = ["top-level-section", "auxiliary-section"]

            open_attribute = " open" if is_main else ""
            closed_label, open_label = (
                ("Show", "Hide") if language == "en" else ("展开", "收起")
            )
            output.append(
                f'<details class="{" ".join(classes)}" '
                f'data-section-kind="{kind}" data-section-key="{section_id}"{open_attribute}>'
                '<summary>'
                f'<h2 id="{section_id}">{heading_html}'
                f'<span class="section-state section-state-closed" aria-hidden="true">{closed_label}</span>'
                f'<span class="section-state section-state-open" aria-hidden="true">{open_label}</span>'
                '</h2></summary><div class="section-body">'
                + content
                + "</div></details>"
            )
        return "".join(output)

    return fold_language(english_body, "en") + marker + fold_language(chinese_body, "zh")


ANCHOR_SCRIPT = r"""
<script>
(function () {
  "use strict";

  function targetFromHash(hash) {
    if (!hash || hash.length < 2) return null;
    var identifier = hash.slice(1);
    try { identifier = decodeURIComponent(identifier); } catch (error) { /* use raw id */ }
    return document.getElementById(identifier);
  }

  function openAncestorDetails(target) {
    var ancestors = [];
    for (var node = target.parentElement; node; node = node.parentElement) {
      if (node.tagName === "DETAILS") ancestors.push(node);
    }
    ancestors.reverse().forEach(function (details) { details.open = true; });
  }

  function revealTarget(target, moveFocus) {
    if (!target) return;
    openAncestorDetails(target);
    if (moveFocus) {
      if (!target.hasAttribute("tabindex")) target.setAttribute("tabindex", "-1");
      target.focus({ preventScroll: true });
    }
    window.requestAnimationFrame(function () {
      target.scrollIntoView({ block: "start" });
    });
  }

  function revealCurrentHash(moveFocus) {
    revealTarget(targetFromHash(window.location.hash), moveFocus);
  }

  document.addEventListener("click", function (event) {
    var link = event.target.closest('a[href^="#"]');
    if (!link) return;
    revealTarget(targetFromHash(link.getAttribute("href")), true);
  }, true);
  window.addEventListener("hashchange", function () { revealCurrentHash(true); });
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () { revealCurrentHash(true); });
  } else {
    revealCurrentHash(true);
  }
}());
</script>
"""


def wrap_language_sections(body: str) -> str:
    """Collapse English and the Chinese review metadata independently."""
    marker = '<div class="language-break" id="chinese-version"></div>'
    english_body, chinese_body = body.split(marker, 1)
    title_end = chinese_body.index("</h1>") + len("</h1>")
    abstract_start = chinese_body.index(
        '<details class="top-level-section main-section abstract-section"',
        title_end,
    )
    review_metadata = chinese_body[title_end:abstract_start]
    chinese_body = (
        chinese_body[:title_end]
        + '<details class="chinese-review-meta"><summary>'
        + '<span class="details-label-closed">草稿信息与作者确认（点击展开）</span>'
        + '<span class="details-label-open">草稿信息与作者确认（点击收起）</span>'
        + '</summary><div class="chinese-review-meta-body">'
        + review_metadata
        + "</div></details>"
        + chinese_body[abstract_start:]
    )
    return (
        '<details class="english-manuscript-details">'
        '<summary>'
        '<span class="details-label-closed">English manuscript (click to expand) / 英文稿（点击展开）</span>'
        '<span class="details-label-open">English manuscript (click to collapse) / 英文稿（点击收起）</span>'
        '</summary>'
        '<section class="manuscript-language english-version" lang="en">'
        + english_body
        + '</section></details>'
        + '<section class="manuscript-language chinese-version language-break" '
        + 'id="chinese-version" lang="zh-CN">'
        + chinese_body
        + '</section>'
    )


def render() -> str:
    markdown_text = SOURCE.read_text(encoding="utf-8")
    chinese_marker = '<div class="language-break" id="chinese-version"></div>'
    if chinese_marker not in markdown_text:
        raise RuntimeError("The bilingual draft is missing the Chinese-version marker.")
    english_text, chinese_text = markdown_text.split(chinese_marker, 1)
    paired_counts = {
        "main figure markers": (
            english_text.count('data-rough-figure="'),
            chinese_text.count('data-rough-figure="'),
        ),
        "supplementary figure markers": (
            english_text.count('data-supplementary-figure="'),
            chinese_text.count('data-supplementary-figure="'),
        ),
    }
    mismatched = {
        label: counts for label, counts in paired_counts.items() if counts[0] != counts[1]
    }
    if mismatched:
        raise RuntimeError(f"English and Chinese structures do not match: {mismatched}")
    version_match = re.search(r"Version ([^\n]+)", markdown_text)
    version = version_match.group(1) if version_match else "draft"
    prepared = convert_tables(markdown_text)
    body = run_cmark(prepared)
    body = format_prompt_appendices(body)
    body, toc = add_heading_ids(body)
    body = inject_figures(body)
    body, reference_counts = link_citations_and_references(body)
    if reference_counts[0] != reference_counts[1]:
        raise RuntimeError(
            f"English and Chinese reference counts differ: {reference_counts}"
        )
    body = style_sections(body)
    body = fold_top_level_sections(body)
    body = body.replace(
        "</h1>",
        f'''</h1>
<div class="draft-meta">
  <span class="chip">Version {html.escape(version)}</span>
  <span class="chip">4 physicians</span>
  <span class="chip">2,160 required note-field judgments</span>
  <span class="chip">120 blinded ablation pairs</span>
  <span class="chip">HTML review copy</span>
</div>
<div class="render-note">The English text is kept close to submission prose. The Chinese section may retain collaborator notes. Figures are embedded as reproducible vector graphics.</div>''',
        1,
    )
    body = body.replace(
        '<div id="english-version"></div>',
        '<div id="english-version"></div><div class="language-nav"><a href="#english-version">English</a><a href="#chinese-version">中文</a></div>',
        1,
    )
    body = wrap_language_sections(body)
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Workshop Paper Draft</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="layout">
    {toc}
    <article>{body}</article>
  </div>
{ANCHOR_SCRIPT}
</body>
</html>
'''


class BalancedHTMLParser(HTMLParser):
    """Minimal structural check for the renderer's explicit HTML tags."""

    VOID_TAGS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in self.VOID_TAGS:
            self.stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        return

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            current = self.stack[-1] if self.stack else "none"
            raise RuntimeError(f"Unbalanced HTML: closing {tag} while {current} is open.")
        self.stack.pop()


def main() -> None:
    output = render()
    OUTPUT.write_text(output, encoding="utf-8")
    required = [
        '<figure class="manuscript-figure">',
        "table-wrap",
        "data:image/svg+xml;base64",
        "Cohen's kappa",
        "2,160 required note-field judgments",
        "Version 0.25, October 2026",
        "79.98%",
        "leave-one-physician-out analysis",
        "Supplementary Table S1",
        "tie-inclusive net preference margin",
        "120 blinded ablation pairs",
        'id="chinese-version"',
        'class="manuscript-language english-version" lang="en"',
        'class="manuscript-language chinese-version language-break"',
        '<details class="english-manuscript-details">',
        '<details class="chinese-review-meta">',
        '<details class="prompt-stage">',
        '<details class="top-level-section main-section"',
        'data-section-kind="appendix"',
        'data-section-kind="references"',
        'data-section-kind="auxiliary"',
        'English manuscript (click to expand)',
        'English manuscript (click to collapse)',
        '草稿信息与作者确认（点击展开）',
        'id="author-confirmation-index"',
        "参考文献",
    ]
    missing = [value for value in required if value not in output]
    if missing:
        raise RuntimeError(f"Rendered HTML is missing required content: {missing}")
    if output.count('<figure class="manuscript-figure">') != 12:
        raise RuntimeError("Expected six main figures in each language version.")
    if output.count('<figure class="manuscript-figure supplementary-figure">') != 2:
        raise RuntimeError("Expected one supplementary figure in each language version.")
    if output.count('<details class="prompt-stage">') != 2:
        raise RuntimeError("Expected one folded judge protocol in each language version.")
    rendered_prompts = re.findall(
        r'<details class="prompt-stage"><summary>.*?</summary>'
        r'<div class="prompt-body"><pre><code[^>]*>(.*?)</code></pre></div></details>',
        output,
        flags=re.S,
    )
    if len(rendered_prompts) != 2 or rendered_prompts[0] != rendered_prompts[1]:
        raise RuntimeError("Appendix S1 must preserve one identical judge prompt per language.")
    for appendix_key in ("s1", "s2", "s3"):
        appendix_headings = len(
            re.findall(
                rf'<h2 id="(?:appendix|附录)-{appendix_key}-[^"]+">',
                output,
            )
        )
        if appendix_headings != 2:
            raise RuntimeError(
                f"Expected independent English and Chinese Appendix {appendix_key.upper()} sections."
            )
    if re.search(r"\boncologists?\b", output, flags=re.I):
        raise RuntimeError("Rendered study terminology must not use oncologist(s).")
    clinician_count = len(re.findall(r"\bclinicians?\b", output, flags=re.I))
    if clinician_count != 2:
        raise RuntimeError(
            "Clinician must occur only in the unchanged Corso reference title, once per language."
        )
    obsolete_judge_labels = [
        "Initial 12-pair calibration instruction",
        "Remaining 108-pair instruction",
        "Minor progress-file correction",
        "Final recalibration instruction",
    ]
    stale_labels = [label for label in obsolete_judge_labels if label in output]
    if stale_labels:
        raise RuntimeError(f"Rendered HTML retains obsolete judge-stage labels: {stale_labels}")
    if output.count('<details class="chinese-review-meta">') != 1:
        raise RuntimeError("Expected one folded Chinese draft-information block.")
    if output.count("Clinical accuracy 与医学措辞") != 1:
        raise RuntimeError("Expected one final physician-review reminder.")
    if "供临床合作者审阅的问题" in output:
        raise RuntimeError("The duplicate yellow clinical-review block must be removed.")
    if '<aside class="author-note ' in output:
        raise RuntimeError("The duplicate conclusion author note must be removed.")
    details_start = output.index('<details class="english-manuscript-details">')
    english_start = output.index('class="manuscript-language english-version"')
    details_end = output.index('</section></details>', english_start) + len('</section></details>')
    chinese_start = output.index('id="chinese-version"')
    if not (details_start < english_start < details_end < chinese_start):
        raise RuntimeError("English must be inside details and Chinese must remain outside it.")
    details_tag = output[details_start : output.index('>', details_start) + 1]
    if " open" in details_tag:
        raise RuntimeError("The English manuscript details element must be closed by default.")
    print_css = CSS.split("@media print {", 1)[1]
    print_requirements = [
        '.english-manuscript-details > summary { display: none; }',
        '.top-level-section > summary {',
        'display: block;',
        'cursor: default;',
        'list-style: none;',
        'page-break-after: avoid;',
        'break-after: avoid-page;',
        '.top-level-section > summary::marker { content: ""; }',
        '.top-level-section > summary::-webkit-details-marker { display: none; }',
        '.top-level-section > summary h2 {',
        '.top-level-section > summary .section-state { display: none !important; }',
        '.top-level-section:not([open]) > .section-body',
        '.chinese-review-meta > summary,',
        '.prompt-stage > summary { display: none; }',
        '.prompt-stage:not([open]) > .prompt-body { display: block !important; }',
    ]
    missing_print_rules = [rule for rule in print_requirements if rule not in print_css]
    if missing_print_rules:
        raise RuntimeError(f"Print CSS is incomplete: {missing_print_rules}")
    if '.top-level-section > summary { display: none; }' in print_css:
        raise RuntimeError("Print CSS must retain top-level summaries as static headings.")
    if 'display: block !important;' not in output:
        raise RuntimeError("Print CSS must expose closed disclosure content.")
    if "onclick=" in output or output.count("<script>") != 1:
        raise RuntimeError("The review copy must use one unobtrusive anchor script and no inline handlers.")
    script_requirements = [
        "openAncestorDetails",
        'window.addEventListener("hashchange"',
        'a[href^="#"]',
        "target.scrollIntoView",
    ]
    missing_script_features = [value for value in script_requirements if value not in output]
    if missing_script_features:
        raise RuntimeError(f"Anchor disclosure script is incomplete: {missing_script_features}")

    top_level_tags = re.findall(
        r'<details class="([^"]*\btop-level-section\b[^"]*)" '
        r'data-section-kind="([^"]+)" data-section-key="([^"]+)"( open)?>',
        output,
    )
    if len(top_level_tags) != 24:
        raise RuntimeError(
            f"Expected 24 bilingual top-level disclosures, found {len(top_level_tags)}."
        )
    kind_counts = Counter(kind for _, kind, _, _ in top_level_tags)
    if kind_counts != Counter({"main": 12, "appendix": 8, "references": 2, "auxiliary": 2}):
        raise RuntimeError(f"Unexpected top-level disclosure classes: {kind_counts}")
    for classes, kind, section_id, open_attribute in top_level_tags:
        should_open = kind == "main"
        if bool(open_attribute) != should_open:
            raise RuntimeError(
                f"Top-level section {section_id} has the wrong default disclosure state."
            )
        if kind == "references" and "appendix-section" in classes:
            raise RuntimeError("References must remain independent from appendices.")
    if output.count('<h2 id="') != len(top_level_tags):
        raise RuntimeError("Every h2 must be the heading of one top-level disclosure.")

    parser = BalancedHTMLParser()
    parser.feed(output)
    parser.close()
    if parser.stack:
        raise RuntimeError(f"Rendered HTML has unclosed tags: {parser.stack}")
    element_ids = re.findall(r'\bid="([^"]+)"', output)
    duplicate_ids = sorted({value for value in element_ids if element_ids.count(value) > 1})
    if duplicate_ids:
        raise RuntimeError(f"Rendered HTML contains duplicate IDs: {duplicate_ids}")
    hrefs = re.findall(r'\bhref="([^"]+)"', output)
    missing_anchors = sorted(
        {href for href in hrefs if href.startswith("#") and href[1:] not in element_ids}
    )
    if missing_anchors:
        raise RuntimeError(f"Rendered HTML contains missing anchors: {missing_anchors}")
    local_targets = {
        href
        for href in hrefs
        if not href.startswith("#") and "://" not in href and not href.startswith("mailto:")
    }
    missing_local_targets = sorted(
        href for href in local_targets if not (ROOT / href).exists()
    )
    if missing_local_targets:
        raise RuntimeError(
            f"Rendered HTML contains missing local resources: {missing_local_targets}"
        )
    print(f"Wrote {OUTPUT.name} ({len(output):,} characters)")


if __name__ == "__main__":
    main()
