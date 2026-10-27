#!/usr/bin/env python3
"""Build the English-only physician-authored manuscript HTML (version 1.0).

The canonical Word extraction is the sole prose source. Technical tables,
figures, captions, navigation, and supplementary analyses are inserted around
the physician blocks and are explicitly marked with data-origin attributes.
"""

from __future__ import annotations

import base64
import csv
import html
import importlib.util
import json
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_DOCX = ROOT / "manuscript/doctor_drafts/202601023_Manuscript_JAMA.docx"
CANONICAL_JSON = ROOT / "manuscript/doctor_drafts/202601023_Manuscript_JAMA.canonical.json"
EXTRACTOR = ROOT / "manuscript/doctor_drafts/extract_physician_source.py"
SOURCE_MD = ROOT / "WORKSHOP_PAPER_DRAFT.md"
OUTPUT_HTML = ROOT / "WORKSHOP_PAPER_DRAFT.html"
RATINGS_DIR = ROOT / "results/extraction_comparison/clinician_ratings"
FIGURE_DIR = RATINGS_DIR / "draft_figures"
EXPECTED_DOCX_SHA256 = "6a1ffc6809e66ad3861284274bdf001d110b4997bbdb053cdf666d4ef27e7066"
EXPECTED_VISIBLE_BLOCK_SHA256 = "5adf3d6434992babecc631e584ff5ba925983bd5036a28995e90b48474c577e3"

MAIN_FIGURES = {
    "1": FIGURE_DIR / "figure1_development_pathway_rough.svg",
    "2": FIGURE_DIR / "figure2_evaluator_distribution_rough.svg",
    "3": FIGURE_DIR / "figure3_interrater_matrix_rough.svg",
    "4": FIGURE_DIR / "figure4_effect_profile.svg",
}
SUPPLEMENTARY_FIGURES = {
    "S1": FIGURE_DIR / "supplementary_figure_s1_note_margins.svg",
    "S2": FIGURE_DIR / "supplementary_figure_s2_staged_ablation.svg",
}

SECTION_HEADINGS = {
    "P002": ("introduction", "Introduction"),
    "P014": ("methods", "Methods"),
    "P084": ("results", "Results"),
    "P107": ("discussion", "Discussion"),
}
SUBHEADINGS = {
    "Study Design",
    "Dataset",
    "Baseline model and model development",
    "Inference Harness",
    "Clinician-in-the-loop breast-cancer refinement",
    "Model-assist pancreatic-cancer refinement",
    "Clinical evaluation data collection and scoring rubric",
    "Staged Technical ablation",
    "Statistical analysis",
    "Performance across Clinical Extraction Fields",
}
PLACEHOLDER_TEXTS = {
    "Table 1",
    "Table 2",
    "Table 3 ",
    "Figure 2 ",
    "Figure 3",
    "Figure 4 删掉4b,4c放在Supplement.",
}


CSS = r"""
:root{--ink:#1e2832;--muted:#65717c;--navy:#173f5f;--blue:#0072b2;--orange:#d55e00;--tie:#b8b8b8;--green:#009e73;--violet:#7a5fa8;--paper:#fff;--page:#eef2f5;--line:#d5dde4;--soft:#f6f8fa;--note:#fff8e6}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--page);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;line-height:1.62}
.layout{display:grid;grid-template-columns:270px minmax(0,940px);gap:28px;width:min(1260px,calc(100% - 36px));margin:24px auto 70px;align-items:start}.toc{position:sticky;top:18px;max-height:calc(100vh - 36px);overflow:auto;padding:18px 15px;border:1px solid var(--line);border-radius:12px;background:#fffffff5}.toc strong{display:block;margin-bottom:9px;color:var(--navy)}.toc a{display:block;padding:4px 6px;border-radius:5px;color:#3e5160;text-decoration:none;font-size:12.5px}.toc a:hover{background:#eaf4fa}.toc .sub{padding-left:18px;font-size:11.5px;color:var(--muted)}
article{min-width:0;padding:45px 54px 68px;border:1px solid var(--line);border-radius:14px;background:var(--paper);box-shadow:0 12px 34px #142b4012}h1{margin:0;color:var(--navy);font-family:Georgia,"Times New Roman",serif;font-size:36px;line-height:1.18}h2,h3{color:var(--navy)}h3{margin:27px 0 9px;font-size:19px}.subtitle{margin:8px 0 20px;color:var(--muted)}
details{border:0}summary{cursor:pointer}.metadata{margin:22px 0 28px;border:1px solid var(--line);border-radius:10px;background:var(--soft)}.metadata>summary,.fold>summary{padding:13px 16px;color:var(--navy);font-weight:750;list-style:none}.metadata>summary::-webkit-details-marker,.fold>summary::-webkit-details-marker{display:none}.metadata>summary::before,.fold>summary::before{content:"▸";display:inline-block;width:1.2em}.metadata[open]>summary::before,.fold[open]>summary::before{content:"▾"}.metadata-body{padding:0 18px 16px}.metadata-grid{display:grid;grid-template-columns:180px 1fr;gap:5px 14px;margin:10px 0}.technical-key{margin:14px 0 0;padding:10px 12px;border-left:4px solid var(--violet);background:#f4f0fa}.badge{display:inline-block;margin-right:6px;padding:1px 7px;border:1px solid var(--violet);border-radius:99px;color:#594477;background:#f4f0fa;font-size:11px;font-weight:750}
.current-change{margin:14px 0 20px;padding:10px 13px;border-left:4px solid var(--blue);background:#eef7fb;color:#264b61;font-size:13px}.author-queries{margin:18px 0 28px;border:1px solid #d8ccea;border-radius:10px;background:#faf8fd}.author-queries>summary{padding:13px 16px;color:#563f78;font-weight:750;list-style:none}.author-queries>summary::-webkit-details-marker,.version-entry>summary::-webkit-details-marker{display:none}.author-queries>summary::before,.version-entry>summary::before{content:"▸";display:inline-block;width:1.2em}.author-queries[open]>summary::before,.version-entry[open]>summary::before{content:"▾"}.query-body{padding:0 18px 16px}.query-list{margin:4px 0 0;padding-left:1.5rem}.query-item{margin:12px 0;scroll-margin-top:16px}.query-item strong{color:#4f3b70}.query-links{display:block;margin-top:3px;font-size:12px}.query-links a{margin-right:9px}.review-marker{margin:10px 0;padding:8px 11px;border-left:3px solid #a080c5;background:#faf7fd;color:#56466d;font-size:12px;scroll-margin-top:16px}.review-marker a{font-weight:700}.review-marker-label{margin-right:6px;color:#6e4b98;font-weight:750}.version-entry{margin:8px 0;border:1px solid var(--line);border-radius:7px;background:#fafbfc}.version-entry>summary{padding:9px 11px;color:#30485a;font-weight:700;list-style:none}.version-entry-body{padding:0 13px 10px;color:#4b5964;font-size:13px}.version-entry-body p{margin:5px 0}
.fold{margin:22px 0;border-top:1px solid var(--line)}.fold>summary{margin:0;padding:20px 0 10px;font-family:Georgia,"Times New Roman",serif;font-size:25px}.fold-body{padding:0 0 8px}.source-empty{height:0}.physician-paragraph{margin:11px 0}.physician-subheading{scroll-margin-top:15px}.source-highlight{padding:0 .06em;background:#fff0a6;color:inherit}.physician-editorial{margin:12px 0;padding:10px 13px;border-left:4px solid #d29b25;background:var(--note);color:#5c491d}.physician-editorial-label{display:block;margin-bottom:3px;color:#7b5a0a;font-size:11px;font-weight:750;text-transform:uppercase;letter-spacing:.04em}.technical-note{margin:12px 0;padding:9px 12px;border-left:4px solid var(--violet);background:#f6f2fb;color:#493c62;font-size:13px}.technical-note .badge{vertical-align:.08em}
.table-wrap{margin:20px 0 28px;overflow-x:auto}.table-caption,.figure-caption{margin:0 0 8px;color:#314452;font-size:13px}.table-caption strong,.figure-caption strong{color:var(--navy)}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 9px;border:1px solid #d6dde3;vertical-align:top}th{background:#e9f1f6;color:#173f5f;text-align:left}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tbody tr:nth-child(even){background:#fafbfc}.physician-native-table th{background:#eaf4ef;color:#18533f}
.figure{margin:24px 0 30px}.figure img{display:block;width:100%;height:auto;border:1px solid #e0e5e9;background:white}.figure-caption{margin-top:9px}.comment{margin:13px 0;padding:12px 14px;border-left:4px solid var(--green);background:#f0f8f5}.comment-meta{font-size:12px;color:var(--muted)}.comment-text{margin:7px 0;white-space:pre-wrap}.comment-context{margin-top:7px;color:#4e5c67;font-size:12px}.references li{margin:8px 0}.version-list li{margin:6px 0}.source-hash{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;overflow-wrap:anywhere}
@media(max-width:900px){.layout{display:block;width:min(100% - 18px,940px);margin:10px auto 40px}.toc{position:relative;top:auto;max-height:none;margin-bottom:12px}.toc a.sub{display:none}article{padding:30px 24px 48px}h1{font-size:30px}.metadata-grid{grid-template-columns:1fr}.table-wrap{margin-left:-8px;margin-right:-8px}.figure{margin-left:-8px;margin-right:-8px}}
@media(max-width:440px){article{padding:24px 15px 40px}h1{font-size:27px}.fold>summary{font-size:22px}table{font-size:11px}th,td{padding:6px 5px}.figure-caption,.table-caption{padding:0 3px}}
@media print{body{background:#fff}.layout{display:block;width:auto;margin:0}.toc,.author-queries>summary,.fold>summary::before,.version-entry>summary::before{display:none!important}article{padding:0;border:0;box-shadow:none}.author-queries:not([open])>.query-body,.fold:not([open])>.fold-body,.version-entry:not([open])>.version-entry-body{display:block!important}.fold>summary,.version-entry>summary{display:block!important;padding:20px 0 8px;list-style:none}.figure,.table-wrap{break-inside:avoid}.physician-editorial,.technical-note,.review-marker{break-inside:avoid}}
"""

SCRIPT = r"""
function revealTarget(target){if(!target)return;if(target instanceof HTMLDetailsElement||target.tagName==='DETAILS')target.open=true;let p=target.parentElement;const chain=[];while(p){if(p.tagName==='DETAILS')chain.push(p);p=p.parentElement;}chain.reverse().forEach(d=>d.open=true);setTimeout(()=>{if(!target.hasAttribute('tabindex'))target.setAttribute('tabindex','-1');target.focus({preventScroll:true});target.scrollIntoView({block:'start'});},0)}
document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(!a)return;const id=decodeURIComponent(a.getAttribute('href').slice(1));revealTarget(document.getElementById(id))});
window.addEventListener('hashchange',()=>revealTarget(document.getElementById(decodeURIComponent(location.hash.slice(1)))));window.addEventListener('DOMContentLoaded',()=>{if(location.hash)revealTarget(document.getElementById(decodeURIComponent(location.hash.slice(1))))});
"""


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def markdown_source_text(value: str) -> str:
    """Keep source trailing spaces visible without Git whitespace errors."""

    trailing = len(value) - len(value.rstrip(" "))
    return value.rstrip(" ") + "&#32;" * trailing


def load_canonical() -> dict[str, object]:
    subprocess.run(
        [sys.executable, str(EXTRACTOR), str(SOURCE_DOCX), "--output", str(CANONICAL_JSON), "--check"],
        cwd=ROOT,
        check=True,
    )
    data = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))
    assert data["source_docx_sha256"] == EXPECTED_DOCX_SHA256
    assert data["canonical_visible_block_sha256"] == EXPECTED_VISIBLE_BLOCK_SHA256
    assert data["counts"] == {
        "body_paragraphs": 127,
        "nonempty_body_paragraphs": 62,
        "native_tables": 2,
        "images": 5,
        "comments": 12,
        "tracked_changes": 0,
        "visible_words": 3335,
    }
    return data


def svg_data_uri(path: Path) -> str:
    data = path.read_bytes()
    if len(data) >= 209_715_200:
        raise ValueError(f"Figure exceeds 200 MiB limit: {path}")
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode("ascii")


def technical_note(text: str, *, anchor: str | None = None) -> str:
    anchor_attr = f' id="{esc(anchor)}"' if anchor else ""
    return (
        f'<p class="technical-note" data-origin="technical-insert"{anchor_attr}>'
        f'<span class="badge">Technical insert</span>{esc(text)}</p>'
    )


def render_physician_paragraph(paragraph: dict[str, object]) -> str:
    block_id = str(paragraph["id"])
    text = str(paragraph["raw_visible_text"])
    attrs = f'id="{esc(block_id)}" data-origin="physician" data-source-block="{esc(block_id)}"'
    rich_text = "".join(
        f'<mark class="source-highlight">{esc(segment["text"])}</mark>'
        if segment["highlighted"]
        else esc(segment["text"])
        for segment in paragraph.get("segments", [{"text": text, "highlighted": False}])
    )
    if not text:
        return f'<div class="source-empty" {attrs}></div>'
    if text in SUBHEADINGS:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return f'<h3 class="physician-subheading" id="{slug}" data-origin="physician" data-source-block="{esc(block_id)}">{rich_text}</h3>'
    if text in PLACEHOLDER_TEXTS or bool(paragraph.get("editorial_note")):
        return (
            '<aside class="physician-editorial">'
            '<span class="physician-editorial-label">Physician source note</span>'
            f'<span {attrs}>{rich_text}</span></aside>'
        )
    return f'<p class="physician-paragraph" {attrs}>{rich_text}</p>'


def render_native_table(table: dict[str, object], label: str, title: str) -> str:
    rows = table["rows"]
    parts = [
        f'<div class="table-wrap" data-origin="technical-insert" id="{esc(label.lower().replace(" ", "-"))}">',
        f'<p class="table-caption"><span class="badge">Technical insert</span><strong>{esc(label)}.</strong> {esc(title)}</p>',
        '<table class="physician-native-table">',
    ]
    for row_index, row in enumerate(rows):
        parts.append("<thead><tr>" if row_index == 0 else "<tbody><tr>")
        tag = "th" if row_index == 0 else "td"
        for cell in row:
            parts.append(
                f'<{tag} data-origin="physician" data-source-block="{esc(cell["id"])}">'
                f'{esc(cell["raw_visible_text"])}</{tag}>'
            )
        parts.append("</tr></thead>" if row_index == 0 else "</tr></tbody>")
    parts.extend(["</table>", "</div>"])
    return "".join(parts)


def result_table(label: str, title: str, headers: list[str], rows: list[list[str]], *, table_id: str) -> str:
    parts = [
        f'<div class="table-wrap" data-origin="technical-insert" id="{table_id}">',
        f'<p class="table-caption"><span class="badge">Technical insert</span><strong>{esc(label)}.</strong> {esc(title)}</p>',
        "<table><thead><tr>",
    ]
    for index, header in enumerate(headers):
        parts.append(f'<th class="{"num" if index else ""}">{esc(header)}</th>')
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for index, value in enumerate(row):
            parts.append(f'<td class="{"num" if index else ""}">{esc(value)}</td>')
        parts.append("</tr>")
    parts.extend(["</tbody></table></div>"])
    return "".join(parts)


def figure(number: str, title: str, caption: str, *, supplementary: bool = False) -> str:
    path = SUPPLEMENTARY_FIGURES[number] if supplementary else MAIN_FIGURES[number]
    label = f"Supplementary Figure {number}" if supplementary else f"Figure {number}"
    attr = "data-supplementary-figure" if supplementary else "data-main-figure"
    return (
        f'<figure class="figure" id="figure-{number.lower()}" data-origin="technical-insert" {attr}="{number}">'
        f'<img src="{svg_data_uri(path)}" alt="{esc(title)}">'
        f'<figcaption class="figure-caption"><span class="badge">Technical insert</span>'
        f'<strong>{esc(label)}.</strong> {esc(caption)}</figcaption></figure>'
    )


def table_1() -> str:
    return result_table(
        "Table 1", "Physician preferences for breast-cancer extraction",
        ["Physician", "Required judgments", "Harness", "Baseline", "Tie", "Harness share among directional judgments"],
        [["01", "280", "84", "22", "174", "79.2%"], ["02", "280", "79", "8", "193", "90.8%"], ["03", "280", "119", "24", "137", "83.2%"], ["04", "280", "81", "26", "173", "75.7%"], ["All", "1,120", "363", "80", "677", "81.9%"]],
        table_id="table-1",
    )


def table_2() -> str:
    return result_table(
        "Table 2", "Physician preferences for PDAC extraction",
        ["Physician", "Required judgments", "Harness", "Baseline", "Tie", "Harness share among directional judgments"],
        [["01", "260", "75", "14", "171", "84.3%"], ["02", "260", "86", "9", "165", "90.5%"], ["03", "260", "103", "30", "127", "77.4%"], ["04", "260", "72", "38", "150", "65.5%"], ["All", "1,040", "336", "91", "613", "78.7%"]],
        table_id="table-2",
    )


def table_3() -> str:
    return result_table(
        "Table 3", "Adjusted probability of inference-harness preference among directional judgments",
        ["Scope", "Directional judgments", "Harness", "Baseline", "Note clusters", "Adjusted probability (95% CI)", "Adjusted odds ratio (95% CI)", "P value"],
        [["Overall", "870", "699", "171", "40", "79.98% (75.75%-83.63%)", "3.995 (3.124-5.110)", "2.67e-28"], ["Breast cancer", "443", "363", "80", "20", "81.66% (76.02%-86.21%)", "4.452 (3.169-6.253)", "7.04e-18"], ["PDAC", "427", "336", "91", "20", "78.15% (71.47%-83.62%)", "3.576 (2.505-5.105)", "2.30e-12"]],
        table_id="table-3",
    )


def load_agreement_rows() -> list[list[str]]:
    module_path = RATINGS_DIR / "plot_manuscript_figures.py"
    spec = importlib.util.spec_from_file_location("manuscript_plots", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raters = module.load_raters()
    names = list(raters)
    rows: list[list[str]] = []
    for cancer, prefix, fields in [("Breast cancer", "b", module.BREAST_FIELDS), ("PDAC", "p", module.PDAC_FIELDS)]:
        for left in range(4):
            for right in range(left + 1, 4):
                agreement, kappa, count = module.pairwise_agreement(raters[names[left]], raters[names[right]], prefix, fields)
                rows.append([cancer, f"Physician {left + 1} vs Physician {right + 1}", str(count), f"{agreement:.1%}", f"{kappa:.3f}"])
    return rows


def supplementary_table_1() -> str:
    return result_table(
        "Supplementary Table S1", "Core fields used in the primary clinical comparison",
        ["Core field", "Applicability", "Rating field", "Evaluation definition"],
        [
            ["Active anticancer medications", "Breast cancer and PDAC", "Current_Medications.current_meds", "Anticancer drugs currently being taken or administered; supportive or home medications, completed or stopped drugs, and planned-only drugs are excluded."],
            ["Stage", "Breast cancer and PDAC", "Cancer_Diagnosis.Stage_of_Cancer", "Cancer stage, TNM, or resectability status, retaining original and current time context when both are documented."],
            ["Distant metastasis", "Breast cancer and PDAC", "Cancer_Diagnosis.Distant Metastasis", "Presence, absence, sites, and certainty of distant spread; regional nodes are not counted as distant metastasis."],
            ["Regional or overall metastasis", "Breast cancer and PDAC", "Cancer_Diagnosis.Metastasis", "Regional nodal and distant involvement, with the site and certainty of each supported finding."],
            ["Treatment response", "Breast cancer and PDAC", "Response_Assessment.response_assessment", "Response to the current or most recently completed anticancer treatment, linked to the applicable regimen and supporting evidence."],
            ["Breast cancer type and receptors", "Breast cancer only", "Cancer_Diagnosis.Type_of_Cancer", "Breast histology and the individual ER, PR, and HER2 results when documented."],
            ["Completed molecular or genetic results", "Breast cancer and PDAC", "Genetic_Testing_Results.genetic_testing_results", "Completed germline, somatic, or other molecular results; pending or merely ordered tests are excluded."],
        ],
        table_id="supplementary-table-s1",
    )


def supplementary_table_2() -> str:
    return result_table(
        "Supplementary Table S2", "Pairwise exact agreement and Cohen's kappa",
        ["Cancer", "Physician pair", "Shared judgments", "Exact agreement", "Cohen's kappa"],
        load_agreement_rows(), table_id="supplementary-table-s2",
    )


def supplementary_table_3() -> str:
    rows: list[list[str]] = []
    with (RATINGS_DIR / "leave_one_physician_out_gee.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append([row["excluded_physician"], row["directional_judgments"], row["clusters"], f'{float(row["adjusted_probability"]):.2%} ({float(row["probability_ci_low"]):.2%}-{float(row["probability_ci_high"]):.2%})', f'{float(row["odds_ratio"]):.3f} ({float(row["ci_low"]):.3f}-{float(row["ci_high"]):.3f})', f'{float(row["p_value"]):.3g}'])
    return result_table(
        "Supplementary Table S3", "Leave-one-physician-out sensitivity analysis of the overall directional GEE",
        ["Excluded physician", "Directional judgments", "Notes", "Adjusted probability (95% CI)", "Adjusted odds ratio (95% CI)", "P value"],
        rows, table_id="supplementary-table-s3",
    )


def render_comments(data: dict[str, object]) -> str:
    parts = []
    for comment in data["comments"]:
        comment_id = str(comment["id"])
        parts.append(
            f'<div class="comment" id="physician-comment-{esc(comment_id)}" data-origin="physician-comment" data-comment-id="{esc(comment_id)}">'
            f'<div class="comment-meta">Comment {esc(comment_id)} · {esc(comment["author"])} · {esc(comment["date"])}</div>'
            f'<div class="comment-text" data-origin="physician" data-comment-text-id="{esc(comment_id)}">{esc(comment["raw_visible_text"])}</div>'
            f'<div class="comment-context"><strong>Anchor:</strong> <a href="#{esc(comment["anchor_paragraph_id"])}">{esc(comment["anchor_paragraph_id"])}</a><br>'
            f'<strong>Selected text:</strong> {esc(comment["anchor_selected_text"])}<br>'
            f'<strong>Paragraph context:</strong> {esc(comment["anchor_paragraph_context"])}</div></div>'
        )
    return "".join(parts)


def references() -> str:
    entries = [
        ("Sushil M, Kennedy VE, Mandair D, et al. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. NEJM AI. 2024;1(4).", "https://doi.org/10.1056/AIdbp2300110"),
        ("Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. npj Digital Medicine. 2024;7:257.", "https://doi.org/10.1038/s41746-024-01233-2"),
        ("Huang J, Yang DM, Rong R, et al. A critical assessment of using ChatGPT for extracting structured data from clinical notes. npj Digital Medicine. 2024;7:106.", "https://doi.org/10.1038/s41746-024-01079-8"),
        ("Kong HJ. Managing Unstructured Big Data in Healthcare System. Healthcare Informatics Research. 2019;25(1):1.", "https://doi.org/10.4258/hir.2019.25.1.1"),
        ("Rule A, Bedrick S, Chiang MF, Hribar MR. Length and Redundancy of Outpatient Progress Notes Across a Decade at an Academic Medical Center. JAMA Network Open. 2021;4(7):e2115334.", "https://doi.org/10.1001/jamanetworkopen.2021.15334"),
        ("Hein D, Christie A, Holcomb M, et al. Iterative refinement and goal articulation to optimize large language models for clinical information extraction. npj Digital Medicine. 2025;8:301.", "https://doi.org/10.1038/s41746-025-01686-z"),
    ]
    return '<ol data-origin="technical-insert">' + "".join(f'<li>{esc(citation)} <a href="{esc(url)}">{esc(url)}</a></li>' for citation, url in entries) + "</ol>"


def review_marker(marker_id: str, query_id: str, label: str) -> str:
    return (
        f'<aside class="review-marker" id="{esc(marker_id)}" data-origin="technical-insert" data-review-query="{esc(query_id)}">'
        f'<span class="review-marker-label">Review marker · {esc(query_id.upper())}</span>{esc(label)} '
        f'<a href="#author-query-{esc(query_id)}" data-origin="technical-insert">Back to {esc(query_id.upper())}</a></aside>'
    )


def author_queries() -> str:
    queries = [
        (
            "aq1",
            "Front matter and submission format",
            "Confirm the final manuscript title, author list, affiliations, target venue and article format, and whether the venue requires a structured abstract.",
            [("review-marker-aq1-front-matter", "Front-matter marker")],
        ),
        (
            "aq2",
            "Table numbering and source cross-references",
            "The physician source calls the two Methods tables Table 1 and Table 2 and later refers to a single Results Table1. The current technical layout uses Methods Tables M1/M2 and separate breast and PDAC Results Tables 1/2. Confirm the final numbering and the source cross-references to use in the editable manuscript.",
            [("review-marker-aq2-m1", "Methods Table M1"), ("review-marker-aq2-m2", "Methods Table M2"), ("review-marker-aq2-results", "Results Tables 1/2")],
        ),
        (
            "aq3",
            "Figure numbering and panel placement",
            "The physician source refers to note-level findings as Figure 2 and Figure 4 and assigns core-field results to Figure 3, while the previously approved layout uses Figure 3 for inter-physician agreement, Figure 4 for core-field raw and adjusted results, and Supplementary Figure S1 for the former note-distribution panel. The source also contains an instruction to delete Figure 4B and move Figure 4C to the supplement. Confirm the final numbering after physician review; the current HTML does not alter the source wording.",
            [("review-marker-aq3-figure2", "Figure 2 reference"), ("review-marker-aq3-core", "Core-field reference"), ("review-marker-aq3-notes", "Note-level reference"), ("review-marker-aq3-editorial", "Source figure instruction"), ("review-marker-aq3-ablation", "Supplementary ablation label")],
        ),
        (
            "aq4",
            "Difference-from-50% reporting",
            "The physician Methods text says that the absolute difference from 50% was reported. At physician request, Table 3 currently omits that redundant column, and the Results report adjusted probabilities and odds ratios without the absolute difference. Confirm whether to restore the absolute effect in Results or revise that Methods sentence in the editable source.",
            [("review-marker-aq4-methods", "Methods statement"), ("review-marker-aq4-table3", "Table 3")],
        ),
        (
            "aq5",
            "Citation placeholders and Word comments",
            "The physician source still contains a literal citation placeholder and 12 unresolved Word comments. Confirm which comment-supplied sources should become formal citations and close or answer the remaining comments before submission.",
            [("review-marker-aq5-citation", "Citation placeholder"), ("review-marker-aq5-references", "References"), ("review-marker-aq5-comments", "Physician comments")],
        ),
        (
            "aq6",
            "Final physician review",
            "A physician author should confirm clinical accuracy, clinical terminology, and interpretation after the technical tables, figures, and cross-reference decisions are final.",
            [("review-marker-aq6-clinical", "End-of-manuscript review marker")],
        ),
    ]
    items = []
    for query_id, title, description, links in queries:
        link_html = " ".join(f'<a href="#{esc(target)}" data-origin="technical-insert">{esc(text)}</a>' for target, text in links)
        items.append(
            f'<li class="query-item" id="author-query-{esc(query_id)}" data-origin="technical-insert">'
            f'<strong>{esc(query_id.upper())}. {esc(title)}</strong><br>{esc(description)}'
            f'<span class="query-links">Jump to: {link_html}</span></li>'
        )
    return (
        '<details class="author-queries" id="author-queries" data-origin="technical-insert">'
        '<summary>Review and author queries · 6 open items</summary>'
        '<div class="query-body"><ol class="query-list">' + "".join(items) + "</ol></div></details>"
    )


def technical_inserts_after(block_id: str) -> str:
    inserts = {
        "P006": review_marker("review-marker-aq5-citation", "aq5", "The source contains a literal citation placeholder in this paragraph."),
        "P030": technical_note("The physician source refers to Table 1. In this HTML, the component table is labeled Methods Table M1.") + review_marker("review-marker-aq2-m1", "aq2", "The Methods table number differs from the source cross-reference."),
        "P038": figure("1", "Development, adaptation, and evaluation workflow", "Development, transfer, and evaluation pathway. Breast-cancer physician review informed the inference harness. The failure history and clinical rubric then supported author-controlled PDAC adaptation without target-domain physician review of outputs. Four physicians compared the final harness with the same-model single-prompt baseline on the 40-note holdout set."),
        "P063": technical_note("The evaluated core fields are defined in Supplementary Table S1."),
        "P068": technical_note("The physician source refers to Table 2. In this HTML, the ablation-condition table is labeled Methods Table M2.") + review_marker("review-marker-aq2-m2", "aq2", "The Methods table number differs from the source cross-reference."),
        "P080": review_marker("review-marker-aq4-methods", "aq4", "The source says the absolute difference from 50% was reported."),
        "P082": technical_note("Exploratory core-field analysis used one joint population-averaged logistic GEE on 495 directional core judgments. Field, physician, and cancer type were fixed effects; repeated judgments were clustered by note with an exchangeable working correlation. Field-specific probabilities were standardized over the physician and cancer distribution observed for that field. Breast type and receptor status applied only to breast-cancer notes. Ties were excluded. These estimates describe preference rather than accuracy, and no pairwise field tests or multiplicity claims were made."),
        "P085": technical_note("The physician source says 'Table1'. The cancer-specific results are separated into Tables 1 and 2 in this HTML.") + review_marker("review-marker-aq2-results", "aq2", "The source uses one Results table reference; this HTML separates the two cancers.") + table_1() + table_2(),
        "P086": review_marker("review-marker-aq3-figure2", "aq3", "The source cites Figure 2 for breast note-level results; the current Figure 2 is the physician-preference overview."),
        "P087": figure("2", "Physician preference overview", "Physician preferences across eight physician-by-cancer evaluations. Panel A retains harness, baseline, and tie judgments. Panel B shows the harness share among directional judgments."),
        "P092": review_marker("review-marker-aq4-table3", "aq4", "Table 3 omits the difference-from-50% column at physician request.") + table_3(),
        "P095": figure("3", "Inter-physician agreement", "Pairwise exact agreement and Cohen's kappa among four physicians in breast cancer and PDAC. Each breast-cancer pair shared 280 judgments, and each PDAC pair shared 260 judgments."),
        "P097": technical_note("The physician source refers to Figure 3 for the core-field results. In this HTML, Figure 3 presents inter-physician agreement and Figure 4 presents the core-field analysis.") + review_marker("review-marker-aq3-core", "aq3", "The physician source and the approved technical layout assign different content to Figure 3.") + figure("4", "Core-field raw and adjusted preference estimates", "Core-field preference results. Panel A retains the tie-inclusive raw distribution for seven physician-prioritized fields. Panel B shows the requested core-field adjusted harness-preference probabilities with 95% confidence intervals from one joint exploratory GEE fitted to 495 directional core judgments. The prior overall adjusted-probability panel was removed. The model included field, physician, and cancer fixed effects, clustered repeated judgments by note with an exchangeable working correlation, and standardized each field over its observed physician and cancer distribution. Breast type and receptor status applied only to breast-cancer notes. Ties were excluded from Panel B. The estimates report preference, not accuracy. No pairwise field tests or multiplicity claims were made."),
        "P100": technical_note("The note-level distribution previously drafted as Figure 4C is reported in Supplementary Figure S1. The prior detailed note panels are not retained.") + review_marker("review-marker-aq3-notes", "aq3", "The source assigns note-level results to Figure 4; the current layout places the distribution in Supplementary Figure S1."),
        "P105": review_marker("review-marker-aq3-editorial", "aq3", "The source contains an inline instruction about Figure 4 panel placement."),
        "P117": technical_note("The staged ablation is reported as Supplementary Figure S2.") + review_marker("review-marker-aq3-ablation", "aq3", "The source still uses the placeholder 'Supplementary Figure X'."),
        "P119": review_marker("review-marker-aq6-clinical", "aq6", "Final physician review is required for clinical accuracy, terminology, and interpretation."),
    }
    return inserts.get(block_id, "")


def build_scientific_sections(data: dict[str, object]) -> tuple[str, str]:
    paragraphs = {item["id"]: item for item in data["paragraphs"]}
    tables = {item["id"]: item for item in data["tables"]}
    front_blocks: list[str] = []
    sections: list[str] = []
    markdown: list[str] = []
    current_parts: list[str] = []
    current_md: list[str] = []
    current_key = ""
    current_title = ""

    def flush() -> None:
        nonlocal current_parts, current_md
        if not current_key:
            return
        heading_id = next(key for key, value in SECTION_HEADINGS.items() if value[0] == current_key)
        heading = paragraphs[heading_id]
        heading_span = f'<span id="{heading_id}" data-origin="physician" data-source-block="{heading_id}">{esc(heading["raw_visible_text"])}</span>'
        sections.append(f'<details class="fold main-section" id="section-{current_key}" open><summary>{heading_span}</summary><div class="fold-body">{"".join(current_parts)}</div></details>')
        markdown.extend([f"## {current_title}", "", f'<!-- physician:{heading_id} -->', current_title, ""])
        markdown.extend(current_md)
        current_parts = []
        current_md = []

    for entry in data["body_sequence"]:
        if entry["kind"] == "paragraph":
            block_id = entry["id"]
            paragraph = paragraphs[block_id]
            if block_id in SECTION_HEADINGS:
                flush()
                current_key, current_title = SECTION_HEADINGS[block_id]
                continue
            rendered = render_physician_paragraph(paragraph)
            if not current_key:
                front_blocks.append(rendered)
                markdown.extend([f'<!-- physician:{block_id} -->', markdown_source_text(str(paragraph["raw_visible_text"])) if paragraph["raw_visible_text"] else f'<!-- {block_id} empty -->', ""])
                continue
            current_parts.append(rendered)
            text = str(paragraph["raw_visible_text"])
            current_md.extend([f'<!-- physician:{block_id} -->', markdown_source_text(text) if text else f'<!-- {block_id} empty -->', ""])
            insert = technical_inserts_after(block_id)
            if insert:
                current_parts.append(insert)
                current_md.extend([f'<!-- technical insert after {block_id} -->', ""])
        elif entry["kind"] == "table":
            table_id = entry["id"]
            if table_id == "T1":
                current_parts.append(render_native_table(tables[table_id], "Methods Table M1", "Inference harness components"))
            elif table_id == "T2":
                current_parts.append(render_native_table(tables[table_id], "Methods Table M2", "Staged technical ablation conditions"))
            current_md.extend([f'<!-- physician native table:{table_id}; cells preserved in canonical JSON -->', ""])
    flush()
    return "".join(front_blocks + sections), "\n".join(markdown)


def build_supplement() -> str:
    return (
        '<details class="fold supplement" id="supplementary-material" data-origin="technical-insert"><summary>Supplementary material</summary><div class="fold-body">'
        + supplementary_table_1()
        + supplementary_table_2()
        + figure("S1", "Note-level preference-margin distribution", "The note-level distribution previously drafted as main-figure Panel C, after pooling all four physicians within each cancer type. The margin is (harness preferences minus baseline preferences) divided by completed judgments for that note. All 20 breast-cancer notes and 19 of 20 PDAC notes had positive margins; one PDAC note had a negative margin. The prior detailed breast and PDAC note panels are not retained. This is a supporting preference sensitivity analysis, not an accuracy estimate.", supplementary=True)
        + figure("S2", "Staged technical ablation", "Blinded staged technical ablation across three cumulative transitions. A was the single-prompt baseline; B added decomposition and orchestration; C added semantic, faithfulness, and temporal verification; D added deterministic inline processing and the complete POST-hook layer. One non-physician Codex judge produced these secondary technical judgments. Exact two-sided sign tests exclude ties.", supplementary=True)
        + supplementary_table_3()
        + "</div></details>"
    )


def version_history() -> str:
    entries = [
        ("v1.1", "2026-10-26", "Removed the draft-information panel, consolidated six open author queries with bidirectional review links, retained only the approved note-distribution panel in Supplementary Figure S1, and made each version-history entry independently collapsible."),
        ("v1.0", "2026-10-26", "Replaced the prior bilingual draft with an English-only rendering of the physician Word manuscript as the sole prose source, added exact source-preservation checks, split the cancer-specific result tables, and rebuilt the main and supplementary figure set."),
        ("v0.25", "2026-10-13", "Added the leave-one-physician-out GEE sensitivity analysis and its supplementary table."),
        ("v0.24", "2026-10-12", "Restored the physician-authored point that fluent outputs can conceal extraction errors."),
        ("v0.23", "2026-10-12", "Made manuscript sections collapsible, added anchor-driven disclosure opening, and preserved complete print output."),
        ("v0.22", "2026-10-11", "Consolidated the main result figures and moved the fully labeled note-level result to the supplement."),
        ("v0.21", "2026-10-11", "Clarified the staged ablation, standardized figure color semantics, and changed the adjusted-effect display to probability scale."),
        ("v0.20", "2026-10-11", "Added probability-scale adjusted effects alongside the odds-ratio results."),
        ("v0.19", "2026-10-11", "Redrew the figure set in a common visual system and consolidated the judge protocol."),
        ("v0.18", "2026-10-10", "Integrated four physicians across two cancer types and revised the study framing around extraction rather than letter generation."),
        ("v0.17", "Date not recorded", "Established the first tracked multi-physician analysis and staged technical ablation draft."),
    ]
    return "".join(
        f'<details class="version-entry" id="version-{esc(version[1:].replace(".", "-"))}" data-origin="technical-insert">'
        f'<summary>{esc(version)} · {esc(date)}</summary><div class="version-entry-body"><p>{esc(note)}</p></div></details>'
        for version, date, note in entries
    )


def toc() -> str:
    links = [("#author-queries", "Review and author queries", False), ("#section-introduction", "Introduction", False), ("#section-methods", "Methods", False)]
    for heading in ["Study Design", "Dataset", "Baseline model and model development", "Inference Harness", "Clinician-in-the-loop breast-cancer refinement", "Model-assist pancreatic-cancer refinement", "Clinical evaluation data collection and scoring rubric", "Staged Technical ablation", "Statistical analysis"]:
        links.append(("#" + re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-"), heading, True))
    links.extend([("#section-results", "Results", False), ("#performance-across-clinical-extraction-fields", "Performance across Clinical Extraction Fields", True), ("#section-discussion", "Discussion", False), ("#supplementary-material", "Supplementary material", False), ("#references", "References", False), ("#physician-comments", "Physician comments", False), ("#version-history", "Version history", False)])
    return '<nav class="toc" data-origin="technical-insert"><strong>Contents</strong>' + "".join(f'<a class="{"sub" if sub else ""}" href="{href}" data-origin="technical-insert">{esc(label)}</a>' for href, label, sub in links) + "</nav>"


def build_html(data: dict[str, object]) -> tuple[str, str]:
    sections, markdown_body = build_scientific_sections(data)
    article = f"""
<article>
  <h1>Physician-authored manuscript draft</h1>
  <p class="subtitle">English working manuscript · Version 1.1 · October 26, 2026</p>
  <p class="current-change" data-origin="technical-insert"><strong>Current change:</strong> consolidated open author queries, corrected the supplementary note-panel placement, and added bidirectional review navigation. Physician-authored text is unchanged.</p>
  {author_queries()}
  {review_marker("review-marker-aq1-front-matter", "aq1", "Final title, authorship, affiliations, venue, format, and abstract requirements remain open.")}
  {sections}
  {build_supplement()}
  <details class="fold references" id="references" data-origin="technical-insert"><summary>References</summary><div class="fold-body">{review_marker("review-marker-aq5-references", "aq5", "Confirm which comment-supplied sources belong in the final bibliography.")}{technical_note("This short list retains verified references directly connected to physician source comments. The Word comments remain preserved separately and are not treated as formal references.")}{references()}</div></details>
  <details class="fold" id="physician-comments" data-origin="technical-insert"><summary>Physician comments</summary><div class="fold-body">{review_marker("review-marker-aq5-comments", "aq5", "Twelve Word comments remain preserved for disposition.")}{render_comments(data)}</div></details>
  <details class="fold" id="version-history" data-origin="technical-insert"><summary>Version history</summary><div class="fold-body">{version_history()}</div></details>
</article>
"""
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Physician-authored manuscript draft v1.1</title><style>{CSS}</style></head><body><div class="layout">{toc()}{article}</div><script>{SCRIPT}</script></body></html>'
    markdown = f"""<!-- GENERATED FILE. The canonical physician Word extraction is the prose authority. -->
# Physician-authored manuscript draft

English working manuscript. Version 1.1, October 26, 2026.

Current change: consolidated open author queries, corrected the supplementary note-panel placement, and added bidirectional review navigation. Physician-authored text is unchanged.

The HTML contains six open author queries covering front matter, table numbering, figure numbering, difference-from-50% reporting, citations and Word comments, and final physician review.

The physician-authored blocks below are copied verbatim from `{SOURCE_DOCX.name}`. Technical tables, figures, captions, and cross-reference notes are generated by `render_workshop_draft.py` and marked in the HTML with `data-origin="technical-insert"`.

{markdown_body}

## Supplementary material

Generated technical content: Supplementary Tables S1-S3 and Supplementary Figures S1-S2.

## Physician comments

All 12 Word comments, including anchor text and paragraph context, are rendered in the HTML from the canonical JSON extraction.
"""
    return document, markdown


class PreservationParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, str | None, str | None, list[str]]] = []
        self.paragraphs: dict[str, str] = {}
        self.paragraph_order: list[str] = []
        self.cells: dict[str, str] = {}
        self.cell_order: list[str] = []
        self.comments: dict[str, str] = {}
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.main_figures: list[str] = []
        self.supplementary_figures: list[str] = []
        self.technical_table_containers: list[str] = []
        self.review_markers: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"): self.ids.append(str(values["id"]))
        if values.get("href", "").startswith("#"): self.hrefs.append(str(values["href"])[1:])
        if values.get("data-main-figure"): self.main_figures.append(str(values["data-main-figure"]))
        if values.get("data-supplementary-figure"): self.supplementary_figures.append(str(values["data-supplementary-figure"]))
        if "table-wrap" in str(values.get("class", "")).split():
            if values.get("data-origin") != "technical-insert": raise AssertionError("Generated table container lacks technical-insert provenance")
            self.technical_table_containers.append(str(values.get("id", "")))
        if "review-marker" in str(values.get("class", "")).split() and values.get("id"):
            self.review_markers[str(values["id"])] = str(values.get("data-review-query", ""))
        capture_type = None
        capture_id = None
        source_block = values.get("data-source-block")
        if values.get("data-origin") == "physician" and source_block:
            capture_type = "cell" if str(source_block).startswith("T") else "paragraph"
            capture_id = str(source_block)
        elif values.get("data-comment-text-id") is not None:
            capture_type = "comment-text"
            capture_id = str(values["data-comment-text-id"])
        self.stack.append((tag, capture_type, capture_id, []))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs); self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        for _, capture_type, _, parts in self.stack:
            if capture_type in {"paragraph", "cell", "comment-text"}: parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack: return
        open_tag, capture_type, capture_id, parts = self.stack.pop()
        if open_tag != tag: return
        if capture_type == "paragraph" and capture_id:
            self.paragraphs[capture_id] = "".join(parts); self.paragraph_order.append(capture_id)
        elif capture_type == "cell" and capture_id:
            self.cells[capture_id] = "".join(parts); self.cell_order.append(capture_id)
        elif capture_type == "comment-text" and capture_id: self.comments[capture_id] = "".join(parts)


def validate(document: str, markdown: str, data: dict[str, object]) -> None:
    parser = PreservationParser(); parser.feed(document)
    expected_paragraphs = {item["id"]: item["raw_visible_text"] for item in data["paragraphs"]}
    expected_cells = {cell["id"]: cell["raw_visible_text"] for table in data["tables"] for row in table["rows"] for cell in row}
    expected_comments = {item["id"]: item["raw_visible_text"] for item in data["comments"]}
    if parser.paragraphs != expected_paragraphs:
        missing = sorted(set(expected_paragraphs) - set(parser.paragraphs)); changed = sorted(key for key in expected_paragraphs if parser.paragraphs.get(key) != expected_paragraphs[key])
        raise AssertionError(f"Physician paragraphs changed: missing={missing}, changed={changed}")
    if parser.paragraph_order != list(expected_paragraphs): raise AssertionError("Physician paragraph sequence changed")
    if parser.cells != expected_cells: raise AssertionError("Physician native table cells changed")
    if parser.cell_order != list(expected_cells): raise AssertionError("Physician native table-cell sequence changed")
    if parser.comments != expected_comments: raise AssertionError("Physician comments changed")
    markdown_physician_ids = re.findall(r"<!-- physician:(P\d{3}) -->", markdown)
    expected_markdown_ids = list(expected_paragraphs)
    if markdown_physician_ids != expected_markdown_ids:
        raise AssertionError(f"Generated Markdown physician sequence changed: expected {len(expected_markdown_ids)}, found {len(markdown_physician_ids)}")
    if len(parser.ids) != len(set(parser.ids)): raise AssertionError("Duplicate HTML IDs")
    missing_anchors = sorted(set(parser.hrefs) - set(parser.ids))
    if missing_anchors: raise AssertionError(f"Missing internal anchors: {missing_anchors}")
    if parser.main_figures != ["1", "2", "3", "4"]: raise AssertionError(f"Unexpected main figures: {parser.main_figures}")
    if parser.supplementary_figures != ["S1", "S2"]: raise AssertionError(f"Unexpected supplementary figures: {parser.supplementary_figures}")
    expected_table_containers = ["methods-table-m1", "methods-table-m2", "table-1", "table-2", "table-3", "supplementary-table-s1", "supplementary-table-s2", "supplementary-table-s3"]
    if parser.technical_table_containers != expected_table_containers: raise AssertionError(f"Unexpected generated table containers: {parser.technical_table_containers}")
    for label in ["Methods Table M1", "Methods Table M2", "Table 1", "Table 2", "Table 3", "Supplementary Table S1", "Supplementary Table S2", "Supplementary Table S3"]:
        if document.count(f"<strong>{label}.</strong>") != 1: raise AssertionError(f"Table label missing or duplicated: {label}")
    if '<details class="fold references" id="references" data-origin="technical-insert">' not in document or '<ol data-origin="technical-insert">' not in document:
        raise AssertionError("References provenance marker is missing")
    if document.count("The evaluated core fields are defined in Supplementary Table S1.") != 1:
        raise AssertionError("Supplementary Table S1 cross-reference is missing")
    if "if(target instanceof HTMLDetailsElement||target.tagName==='DETAILS')target.open=true" not in document or "target.focus({preventScroll:true})" not in document:
        raise AssertionError("Closed-details target reveal behavior is missing")
    expected_queries = [f"author-query-aq{index}" for index in range(1, 7)]
    if any(document.count(f'id="{query_id}"') != 1 for query_id in expected_queries): raise AssertionError("Author-query IDs are missing or duplicated")
    expected_markers = {
        "review-marker-aq1-front-matter": "aq1",
        "review-marker-aq2-m1": "aq2", "review-marker-aq2-m2": "aq2", "review-marker-aq2-results": "aq2",
        "review-marker-aq3-figure2": "aq3", "review-marker-aq3-core": "aq3", "review-marker-aq3-notes": "aq3", "review-marker-aq3-editorial": "aq3", "review-marker-aq3-ablation": "aq3",
        "review-marker-aq4-methods": "aq4", "review-marker-aq4-table3": "aq4",
        "review-marker-aq5-citation": "aq5", "review-marker-aq5-references": "aq5", "review-marker-aq5-comments": "aq5",
        "review-marker-aq6-clinical": "aq6",
    }
    if parser.review_markers != expected_markers: raise AssertionError(f"Review-marker set changed: {parser.review_markers}")
    for marker_id, query_id in expected_markers.items():
        if document.count(f'href="#{marker_id}"') != 1 or document.count(f'href="#author-query-{query_id}"') < 1:
            raise AssertionError(f"Bidirectional review link is missing for {marker_id}")
    if '<details class="author-queries" id="author-queries" data-origin="technical-insert">' not in document: raise AssertionError("Default-closed author-query panel is missing")
    if 'id="draft-metadata"' in document or "Draft information and source provenance" in document: raise AssertionError("Removed draft-information block remains")
    if "Version 1.1 · October 26, 2026" not in document or "draft v1.1" not in document: raise AssertionError("Version 1.1 metadata is missing")
    if document.count('class="version-entry"') != 11 or any(f'id="version-{item}"' not in document for item in ["1-1", "1-0", "0-25", "0-24", "0-23", "0-22", "0-21", "0-20", "0-19", "0-18", "0-17"]):
        raise AssertionError("Nested version history is incomplete")
    if 'lang="zh"' in document or 'id="chinese-version"' in document: raise AssertionError("Chinese manuscript shell remains")
    if "Difference from 50%" in document: raise AssertionError("Removed Difference from 50% column remains")
    for stale in ["figure5_staged_ablation.svg", "figure6_clinical_handoff.svg", 'data-main-figure="5"', 'data-main-figure="6"']:
        if stale in document: raise AssertionError(f"Retired main figure remains: {stale}")
    if len(markdown.split()) < 3000: raise AssertionError("Generated Markdown is unexpectedly short")
    if len(document.encode("utf-8")) >= 209_715_200: raise AssertionError("HTML exceeds 200 MiB")


def main() -> None:
    data = load_canonical()
    document, markdown = build_html(data)
    validate(document, markdown, data)
    SOURCE_MD.write_text(markdown, encoding="utf-8")
    OUTPUT_HTML.write_text(document, encoding="utf-8")
    print(f"Wrote {SOURCE_MD}")
    print(f"Wrote {OUTPUT_HTML}")
    print("Preserved 162/162 physician text blocks (127 paragraphs and 35 native-table cells) and 12/12 comments; rendered 4 main and 2 supplementary figures.")


if __name__ == "__main__":
    main()
