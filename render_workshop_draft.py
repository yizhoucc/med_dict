#!/usr/bin/env python3
"""Render WORKSHOP_PAPER_DRAFT.md as a self-contained, readable HTML page.

Run from the repository root:
    python3 render_workshop_draft.py

The renderer uses the locally installed CommonMark ``cmark`` command, adds GFM-like
table handling, styles figure placeholders as design cards, and embeds available
rough SVG plots as data URIs. No network access is required.
"""

from __future__ import annotations

import base64
import html
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "WORKSHOP_PAPER_DRAFT.md"
OUTPUT = ROOT / "WORKSHOP_PAPER_DRAFT.html"
FIGURE_DIR = ROOT / "results/extraction_comparison/clinician_ratings/draft_figures"

ROUGH_FIGURES = {
    "2": (FIGURE_DIR / "figure2_evaluator_distribution_rough.svg", "Clinician preference by evaluation"),
    "3": (FIGURE_DIR / "figure3_interrater_matrix_rough.svg", "Pairwise breast-cancer agreement"),
    "4": (FIGURE_DIR / "figure4_core_fields_rough.svg", "Clinician preference by core category"),
    "5": (FIGURE_DIR / "figure5_note_margins_rough.svg", "Per-note normalized preference margins"),
    "7": (FIGURE_DIR / "figure7_letter_differences_rough.svg", "Exploratory patient-letter score differences"),
    "S1": (
        FIGURE_DIR / "supplementary_figure1_technical_vs_clinician_rough.svg",
        "Technical-audit and clinician net preference rates",
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
  padding: 8px 18px 22px;
  border: 1px solid #cfe0ec;
  border-left: 5px solid var(--blue);
  border-radius: 10px;
  background: #f7fbfd;
}
.abstract-section h2 { margin-top: 14px; border-bottom: 0; }
.review-section {
  margin: 24px 0 30px;
  padding: 2px 18px 12px;
  border: 1px solid #eadfc3;
  border-radius: 10px;
  background: #fffaf0;
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
.figure-placeholder {
  position: relative;
  margin: 22px 0 28px;
  padding: 20px 20px 14px;
  border: 2px dashed #9a87c7;
  border-radius: 11px;
  background: var(--violet-soft);
  color: #342a50;
}
.figure-placeholder::before {
  content: "FIGURE SPECIFICATION";
  display: inline-block;
  margin-bottom: 8px;
  padding: 3px 8px;
  border-radius: 999px;
  background: var(--violet);
  color: white;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.08em;
}
.figure-placeholder p { margin: 8px 0; }
.figure-placeholder p:first-of-type { font-size: 16px; }
.rough-figure {
  margin: 25px 0 8px;
  padding: 14px;
  border: 1px solid #cad7df;
  border-radius: 10px;
  background: #fbfdfe;
  text-align: center;
}
.rough-figure img { display: block; width: 100%; height: auto; margin: 0 auto; }
.rough-figure figcaption { margin-top: 9px; color: var(--muted); font-size: 12px; }
.rough-label {
  display: inline-block;
  margin-bottom: 8px;
  padding: 3px 8px;
  border-radius: 999px;
  background: #e6eef3;
  color: #3c5262;
  font-size: 10px;
  font-weight: 800;
  letter-spacing: 0.06em;
}
.todo {
  padding: 11px 14px;
  border: 1px solid #eed48a;
  border-radius: 8px;
  background: var(--warning-soft);
  color: var(--warning);
  font-weight: 650;
}
.references-section { color: #44515c; font-size: 13px; }
.references-section h2 { font-size: 22px; }
@media (max-width: 980px) {
  .layout { display: block; width: min(920px, calc(100% - 22px)); margin-top: 11px; }
  .toc { position: relative; top: 0; max-height: none; margin-bottom: 12px; }
  .toc a { display: none; }
  article { padding: 30px 24px 55px; }
  h1 { font-size: 29px; }
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
  .figure-placeholder, .rough-figure, .table-wrap { break-inside: avoid; }
  .rough-figure img { max-height: 650px; object-fit: contain; }
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


def add_rough_markers(markdown_text: str) -> str:
    output: list[str] = []
    for line in markdown_text.splitlines():
        english_match = re.match(
            r"> \*\*(Figure [23457]|Supplementary Figure S1) placeholder:", line
        )
        chinese_match = re.match(r"> \*\*(图 [23457]|补充图 S1) 占位：", line)
        match = english_match or chinese_match
        if match:
            key = (
                match.group(1)
                .replace("Figure ", "")
                .replace("Supplementary ", "")
                .replace("补充图 ", "")
                .replace("图 ", "")
            )
            output.append(f'<div data-rough-figure="{key}"></div>')
            output.append("")
        output.append(line)
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
    toc_lines = ['<nav class="toc">', '<div class="toc-title">Workshop draft</div>', '<div class="toc-actions"><button onclick="window.print()">Print / Save PDF</button></div>']
    for level, slug, label in headings:
        toc_lines.append(f'<a class="level-{level}" href="#{slug}">{html.escape(label)}</a>')
    toc_lines.append("</nav>")
    return body, "\n".join(toc_lines)


def embed_svg(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def inject_rough_figures(body: str) -> str:
    for key, (path, title) in ROUGH_FIGURES.items():
        marker = f'<div data-rough-figure="{key}"></div>'
        if marker not in body or not path.exists():
            continue
        figure = f'''<figure class="rough-figure">
<div class="rough-label">INTERNAL ROUGH PLOT</div>
<img src="{embed_svg(path)}" alt="{html.escape(title)}">
<figcaption>{html.escape(title)}. Trend-check rendering only; replace with publication artwork.</figcaption>
</figure>'''
        body = body.replace(marker, figure)
    return body


def style_sections(body: str) -> str:
    body = re.sub(
        r"<blockquote>\s*(?=<p><strong>(?:Figure|Supplementary Figure|图|补充图).*?(?:placeholder:|占位：))(.*?)</blockquote>",
        r'<aside class="figure-placeholder">\1</aside>',
        body,
        flags=re.S,
    )
    body = re.sub(
        r"(<h2 id=\"questions-for-the-clinical-collaborator\">.*?)(?=<h2 id=\"abstract\">)",
        r'<section class="review-section">\1</section>',
        body,
        flags=re.S,
    )
    body = re.sub(
        r"(<h2 id=\"abstract\">.*?)(?=<h2 id=\"1-introduction\">)",
        r'<section class="abstract-section">\1</section>',
        body,
        flags=re.S,
    )
    body = re.sub(
        r"(<h2 id=\"references\">.*?)(?=<div class=\"language-break\" id=\"chinese-version\"></div>)",
        r'<section class="references-section">\1</section>',
        body,
        flags=re.S,
    )
    body = re.sub(
        r"(<h2 id=\"供临床合作者审阅的问题\">.*?)(?=<h2 id=\"摘要\">)",
        r'<section class="review-section">\1</section>',
        body,
        flags=re.S,
    )
    body = re.sub(
        r"(<h2 id=\"摘要\">.*?)(?=<h2 id=\"1-引言\">)",
        r'<section class="abstract-section">\1</section>',
        body,
        flags=re.S,
    )
    body = re.sub(
        r"(<h2 id=\"参考文献\">.*)$",
        r'<section class="references-section">\1</section>',
        body,
        flags=re.S,
    )
    body = re.sub(
        r"<p>(\[(?:FINAL|TODO|最终|投稿前)[^<]+\])</p>",
        r'<p class="todo">\1</p>',
        body,
    )
    return body


def render() -> str:
    markdown_text = SOURCE.read_text(encoding="utf-8")
    chinese_marker = '<div class="language-break" id="chinese-version"></div>'
    if chinese_marker not in markdown_text:
        raise RuntimeError("The bilingual draft is missing the Chinese-version marker.")
    english_text, chinese_text = markdown_text.split(chinese_marker, 1)
    paired_counts = {
        "level-2 headings": (english_text.count("\n## "), chinese_text.count("\n## ")),
        "level-3 headings": (english_text.count("\n### "), chinese_text.count("\n### ")),
        "tables": (english_text.count("|---"), chinese_text.count("|---")),
        "figure placeholders": (english_text.count("placeholder:"), chinese_text.count("占位：")),
    }
    mismatched = {
        label: counts for label, counts in paired_counts.items() if counts[0] != counts[1]
    }
    if mismatched:
        raise RuntimeError(f"English and Chinese structures do not match: {mismatched}")
    version_match = re.search(r"Version ([^\n]+)", markdown_text)
    version = version_match.group(1) if version_match else "draft"
    prepared = add_rough_markers(convert_tables(markdown_text))
    body = run_cmark(prepared)
    body, toc = add_heading_ids(body)
    body = inject_rough_figures(body)
    body = style_sections(body)
    body = body.replace(
        "</h1>",
        f'''</h1>
<div class="draft-meta">
  <span class="chip">Version {html.escape(version)}</span>
  <span class="chip">3 oncologists</span>
  <span class="chip">1,359 required-field judgments</span>
  <span class="chip">HTML review copy</span>
</div>
<div class="render-note">Figure specification cards describe the intended publication graphics. Embedded SVGs are internal trend checks and are not final artwork.</div>''',
        1,
    )
    body = body.replace(
        '<div id="english-version"></div>',
        '<div id="english-version"></div><div class="language-nav"><a href="#english-version">English</a><a href="#chinese-version">中文</a></div>',
        1,
    )
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
</body>
</html>
'''


def main() -> None:
    output = render()
    OUTPUT.write_text(output, encoding="utf-8")
    required = [
        "figure-placeholder",
        "table-wrap",
        "data:image/svg+xml;base64",
        "Cohen's kappa from 0.511 to 0.646",
        "1,359 required-field judgments",
        'id="chinese-version"',
        "供临床合作者审阅的问题",
        "参考文献",
    ]
    missing = [value for value in required if value not in output]
    if missing:
        raise RuntimeError(f"Rendered HTML is missing required content: {missing}")
    print(f"Wrote {OUTPUT.name} ({len(output):,} characters)")


if __name__ == "__main__":
    main()
