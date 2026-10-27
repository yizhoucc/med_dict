#!/usr/bin/env python3
"""Build the English-only physician-authored manuscript HTML (version 1.2).

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
HIDDEN_SOURCE_BLOCKS = {"P050", "P071", "P089", "P091", "P094", "P099", "P105"}

REVISED_PARAGRAPHS = {
    "P006": "Although EHRs contains structured fields, it is estimated that 80% of clinical data exist in unstructured form, and a large proportion of clinically relevant information is encoded in free-text narratives. Oncology notes are particularly challenging because a patient can have a complex medical history that contains multiple diagnoses, prior and future treatments, and change in disease status. Key information may be distributed across different sections, expressed with uncertainty, linked through complex timelines, or even documented incorrectly. As commonly seen in real-world practice, copy-forward laboratory, imaging, and genomic results further length notes, further complicating the extraction task. In a study of outpatient documentations across a decade, notes became longer and more redundant, and most of it through templates or copy-paste [22]. LLMs must therefore not only identify the relevant information, but also determine its relationship to patient’s current diseases. ",
    "P030": "Each model output passes through five verification stages. The first two address structural errors though JSON formatting repair and schema validation. The remaining three assess clinical content: Semantic alignment, faithfulness check, and temporal filtering. These verification stages ensure that the extracted information was complete, clinically relevant, and consistent with the source note. See Table 1.",
    "P063": "The clinical evaluation presented the source note and two structured outputs through a system-label-masked A/B interface. Evaluators selected A better, B better, or tie for each field. They were blinded from the harness and models. The physician participated in the human-in-the-loop development was not one of the final evaluators. Before evaluating the test set, the authors designed 7 core-fields as clinically important for understanding disease status and current management, see Supplementary Table S1",
    "P068": "See Table 2 for the 4 conditions.",
    "P080": "The primary outcome was the adjusted probability of harness preference among directional judgments. Predicted probabilities were standardized over the observed covariate distribution, and 95% confidence intervals (CIs) were calculated using the delta method on the logit scale. Adjusted probabilities and adjusted odds ratios (ORs) were reported.",
    "P085": "All 4 physicians evaluated both cancer types and completed 2160 comparisons. Across 2,160 required note-field judgments, the harness was preferred 699 times, 1,290 comparisons were ties, and the baseline was preferred 171 times, see Table 3, Table 4, and Figure 2. These correspond to 32.36%, 59.72%, and 7.92% of all judgments. Among the 870 directional judgments, the crude harness share was 80.34% (note-cluster robust 95% CI 76.14% to 83.96%). ",
    "P086": "For breast cancer note, all four physicians independently favored the harness on the breast-cancer set. Their pooled breast result was 363 harness preferences, 80 baseline preferences, and 677 ties. The first physician's per-note result was 18 harness wins, one baseline win, and one tie. The second and third physicians each recorded a harness win in all 20 notes. The fourth recorded 16 harness wins, two baseline wins, and two ties. When the four breast evaluations were combined within each note, all 20 notes had a positive harness-minus-baseline margin (Supplementary Figure S1). ",
    "P092": "The adjusted probability of a harness preference among directional judgments was 79.98% (95% CI 75.75% to 83.63%). The breast-cancer estimate was 81.66% (95% CI, 76.02% to 86.21%), the pancreatic-cancer estimate was 78.15% (95% CI, 71.47% to 83.62). The same standardized estimates on the odds scale were 3.995 overall (95% CI, 3.12 to 5.11; p<0.05), 4.45 for breast cancer (95% CI, 3.17 to 6.25; p<0.05), and 3.576 for pancreatic cancer (95% CI, 2.51 to 5.11; p <0.05), see Table 5. ",
    "P095": "Sensitivity analyses supported the primary findings. When ties were included, the net preference margin favored the harness (+24.4 percentage points; 95% CI, +20.4 to +28.5). At the note level, the harness was preferred in 39 of 40 cases (97.5%). Results remained consistent when each physician was excluded individually; adjusted harness-preference probabilities ranged from 77.08% to 83.53% (Supplementary Table S3).",
    "P097": "Next, we analyzed which questions have the most influence on the results. In all the 1,040 evaluations, physicians preferred the harness in 418 comparisons (40.2%), the baseline in 77 (7.4%), and 545 (52.4%) were ties. When ties were excluded, 84.4% favored the harness. The harness was preferred more often than the baseline across all 7 fields, with the largest difference seen in identifying active anticancer medications (132 vs 3). Differences were smaller for distant metastasis and breast cancer type and receptor status, where ties were more common (Figure 4).",
    "P100": "We also analyzed physician preferences for each individual note to determine whether the overall advantage was consistent across cases. After combining physician ratings for each note, the harness was preferred in all 20 breast cancer notes and 19 of 20 pancreatic cancer notes. The overall ratings also favored the harness in each evaluator-by-cancer group. These findings were consistent with the primary analysis and suggest that the observed advantage was not driven by a small number of clinical notes (Supplementary Figure S1).",
    "P117": "Our ablation analysis suggested that verification was an important component of the inference harness (Supplementary Figure S2).  Adding decomposed extraction alone did not produce an overall improvement while adding layers of verification gates resulted in significantly more favorable evaluations. Additionally, we found that the clinical rules developed from breast cancer did not provide further improvement for pancreatic cancer, this suggests that restrictive rules that performed well in one cancer type may not improve performance in another. More restrictive rules may introduce trade-offs, especially when the terminology or treatment differ between cancer types. For example, some rules developed for breast cancer were specific to estrogen receptor (ER), progesterone receptor (PR), and HER2 status, which are used to classify breast cancer subtypes but are not applicable to PDAC.",
}

XREF_TARGETS = {
    "[22]": "lit-22",
    "Table 1": "table-1",
    "Table 2": "table-2",
    "Table 3": "table-3",
    "Table 4": "table-4",
    "Table 5": "table-5",
    "Supplementary Table S1": "supplementary-table-s1",
    "Supplementary Table S2": "supplementary-table-s2",
    "Supplementary Table S3": "supplementary-table-s3",
    "Figure 1": "figure-1",
    "Figure 2": "figure-2",
    "Figure 3": "figure-3",
    "Figure 4": "figure-4",
    "Supplementary Figure S1": "figure-s1",
    "Supplementary Figure S2": "figure-s2",
}


CSS = r"""
:root{--ink:#1e2832;--muted:#65717c;--navy:#173f5f;--blue:#0072b2;--orange:#d55e00;--tie:#b8b8b8;--green:#009e73;--paper:#fff;--page:#eef2f5;--line:#d5dde4;--soft:#f6f8fa}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:var(--page);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;line-height:1.62}
.layout{display:grid;grid-template-columns:270px minmax(0,940px);gap:28px;width:min(1260px,calc(100% - 36px));margin:24px auto 70px;align-items:start}.toc{position:sticky;top:18px;max-height:calc(100vh - 36px);overflow:auto;padding:18px 15px;border:1px solid var(--line);border-radius:12px;background:#fffffff5}.toc strong{display:block;margin-bottom:9px;color:var(--navy)}.toc a{display:block;padding:4px 6px;border-radius:5px;color:#3e5160;text-decoration:none;font-size:12.5px}.toc a:hover{background:#eaf4fa}.toc .sub{padding-left:18px;font-size:11.5px;color:var(--muted)}
article{min-width:0;padding:45px 54px 68px;border:1px solid var(--line);border-radius:14px;background:var(--paper);box-shadow:0 12px 34px #142b4012}h1{margin:0;color:var(--navy);font-family:Georgia,"Times New Roman",serif;font-size:36px;line-height:1.18}h2,h3{color:var(--navy)}h3{margin:27px 0 9px;font-size:19px}.subtitle{margin:8px 0 20px;color:var(--muted)}
details{border:0}summary{cursor:pointer}.fold>summary{padding:13px 16px;color:var(--navy);font-weight:750;list-style:none}.fold>summary::-webkit-details-marker,.version-entry>summary::-webkit-details-marker{display:none}.fold>summary::before,.version-entry>summary::before{content:"▸";display:inline-block;width:1.2em}.fold[open]>summary::before,.version-entry[open]>summary::before{content:"▾"}.current-change{margin:14px 0 20px;padding:10px 13px;border-left:4px solid var(--blue);background:#eef7fb;color:#264b61;font-size:13px}.version-entry{margin:8px 0;border:1px solid var(--line);border-radius:7px;background:#fafbfc}.version-entry>summary{padding:9px 11px;color:#30485a;font-weight:700;list-style:none}.version-entry-body{padding:0 13px 10px;color:#4b5964;font-size:13px}.version-entry-body p{margin:5px 0}
.fold{margin:22px 0;border-top:1px solid var(--line)}.fold>summary{margin:0;padding:20px 0 10px;font-family:Georgia,"Times New Roman",serif;font-size:25px}.fold-body{padding:0 0 8px}.source-empty,.source-audit{display:none!important}.physician-paragraph,.manuscript-paragraph{margin:11px 0}.physician-subheading{scroll-margin-top:15px}.xref,.citation,.comment-anchor a{color:var(--blue);text-decoration:none}.xref:hover,.citation:hover,.comment-anchor a:hover{text-decoration:underline}.comment-anchor{margin-left:.18em;font-size:.72em;vertical-align:super}.comment-anchor a{font-weight:650}.comment-backlink{display:inline-block;margin-top:7px;font-size:12px}
.table-wrap{margin:20px 0 28px;overflow-x:auto}.table-caption,.figure-caption{margin:0 0 8px;color:#314452;font-size:13px}.table-caption strong,.figure-caption strong{color:var(--navy)}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px 9px;border:1px solid #d6dde3;vertical-align:top}th{background:#e9f1f6;color:#173f5f;text-align:left}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}tbody tr:nth-child(even){background:#fafbfc}.physician-native-table th{background:#eaf4ef;color:#18533f}
.figure{margin:24px 0 30px}.figure img{display:block;width:100%;height:auto;border:1px solid #e0e5e9;background:white}.figure-caption{margin-top:9px}.comment{margin:13px 0;padding:12px 14px;border:1px solid var(--line);border-radius:7px;background:#fafbfc}.comment-meta{font-size:12px;color:var(--muted)}.comment-text{margin:7px 0;white-space:pre-wrap}.comment-context{margin-top:7px;color:#4e5c67;font-size:12px}.literature-workspace li{margin:11px 0}.reference-note{display:block;margin-top:3px;color:#4e5c67;font-size:12px}.workspace-note{color:#4e5c67;font-size:13px}.version-list li{margin:6px 0}.source-hash{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;overflow-wrap:anywhere}
@media(max-width:900px){.layout{display:block;width:min(100% - 18px,940px);margin:10px auto 40px}.toc{position:relative;top:auto;max-height:none;margin-bottom:12px}.toc a.sub{display:none}article{padding:30px 24px 48px}h1{font-size:30px}.metadata-grid{grid-template-columns:1fr}.table-wrap{margin-left:-8px;margin-right:-8px}.figure{margin-left:-8px;margin-right:-8px}}
@media(max-width:440px){article{padding:24px 15px 40px}h1{font-size:27px}.fold>summary{font-size:22px}table{font-size:11px}th,td{padding:6px 5px}.figure-caption,.table-caption{padding:0 3px}}
@media print{body{background:#fff}.layout{display:block;width:auto;margin:0}.toc,.fold>summary::before,.version-entry>summary::before{display:none!important}article{padding:0;border:0;box-shadow:none}.fold:not([open])>.fold-body,.version-entry:not([open])>.version-entry-body{display:block!important}.fold>summary,.version-entry>summary{display:block!important;padding:20px 0 8px;list-style:none}.figure,.table-wrap,.comment{break-inside:avoid}}
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


def link_crossrefs(text: str) -> str:
    pattern = re.compile("|".join(re.escape(label) for label in sorted(XREF_TARGETS, key=len, reverse=True)))
    parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        parts.append(esc(text[cursor:match.start()]))
        label = match.group(0)
        parts.append(
            f'<a class="{ "citation" if label.startswith("[") else "xref"}" '
            f'href="#{XREF_TARGETS[label]}">{esc(label)}</a>'
        )
        cursor = match.end()
    parts.append(esc(text[cursor:]))
    return "".join(parts)


def source_audit(block_id: str, text: str) -> str:
    return (
        f'<span class="source-audit" hidden aria-hidden="true" data-origin="physician" '
        f'data-source-block="{esc(block_id)}">{esc(text)}</span>'
    )


def comment_anchors(comment_ids: list[str]) -> str:
    return "".join(
        f'<sup class="comment-anchor" id="comment-anchor-{esc(comment_id)}">'
        f'<a href="#physician-comment-{esc(comment_id)}" aria-label="Comment {esc(comment_id)}">{esc(comment_id)}</a></sup>'
        for comment_id in comment_ids
    )


def manuscript_paragraph(text: str, *, paragraph_id: str | None = None) -> str:
    id_attr = f' id="{esc(paragraph_id)}"' if paragraph_id else ""
    return f'<p class="manuscript-paragraph" data-origin="technical-insert"{id_attr}>{link_crossrefs(text)}</p>'


def render_physician_paragraph(paragraph: dict[str, object], comment_ids: list[str]) -> str:
    block_id = str(paragraph["id"])
    text = str(paragraph["raw_visible_text"])
    anchors = comment_anchors(comment_ids)
    if block_id in HIDDEN_SOURCE_BLOCKS:
        return source_audit(block_id, text) + anchors
    if block_id in REVISED_PARAGRAPHS:
        return (
            source_audit(block_id, text)
            + f'<p class="manuscript-paragraph" id="{esc(block_id)}" data-origin="manuscript-edit">'
            + link_crossrefs(REVISED_PARAGRAPHS[block_id])
            + anchors
            + "</p>"
        )
    source_span = (
        f'<span data-origin="physician" data-source-block="{esc(block_id)}">{esc(text)}</span>'
    )
    if not text:
        return f'<div class="source-empty" id="{esc(block_id)}">{source_span}</div>{anchors}'
    if text in SUBHEADINGS:
        slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
        return f'<h3 class="physician-subheading" id="{slug}">{source_span}{anchors}</h3>'
    return f'<p class="physician-paragraph" id="{esc(block_id)}">{source_span}{anchors}</p>'


def render_native_table(table: dict[str, object], label: str, title: str) -> str:
    rows = table["rows"]
    parts = [
        f'<div class="table-wrap" data-origin="technical-insert" id="{esc(label.lower().replace(" ", "-"))}">',
        f'<p class="table-caption"><strong>{esc(label)}.</strong> {esc(title)}</p>',
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
        f'<p class="table-caption"><strong>{esc(label)}.</strong> {esc(title)}</p>',
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
        f'<figcaption class="figure-caption"><strong>{esc(label)}.</strong> {esc(caption)}</figcaption></figure>'
    )


def table_3() -> str:
    return result_table(
        "Table 3", "Physician preferences for breast-cancer extraction",
        ["Physician", "Required judgments", "Harness", "Baseline", "Tie", "Harness share among directional judgments"],
        [["01", "280", "84", "22", "174", "79.2%"], ["02", "280", "79", "8", "193", "90.8%"], ["03", "280", "119", "24", "137", "83.2%"], ["04", "280", "81", "26", "173", "75.7%"], ["All", "1,120", "363", "80", "677", "81.9%"]],
        table_id="table-3",
    )


def table_4() -> str:
    return result_table(
        "Table 4", "Physician preferences for PDAC extraction",
        ["Physician", "Required judgments", "Harness", "Baseline", "Tie", "Harness share among directional judgments"],
        [["01", "260", "75", "14", "171", "84.3%"], ["02", "260", "86", "9", "165", "90.5%"], ["03", "260", "103", "30", "127", "77.4%"], ["04", "260", "72", "38", "150", "65.5%"], ["All", "1,040", "336", "91", "613", "78.7%"]],
        table_id="table-4",
    )


def table_5() -> str:
    return result_table(
        "Table 5", "Adjusted probability of inference-harness preference among directional judgments",
        ["Scope", "Directional judgments", "Harness", "Baseline", "Note clusters", "Adjusted probability (95% CI)", "Adjusted odds ratio (95% CI)", "P value"],
        [["Overall", "870", "699", "171", "40", "79.98% (75.75%-83.63%)", "3.995 (3.124-5.110)", "2.67e-28"], ["Breast cancer", "443", "363", "80", "20", "81.66% (76.02%-86.21%)", "4.452 (3.169-6.253)", "7.04e-18"], ["PDAC", "427", "336", "91", "20", "78.15% (71.47%-83.62%)", "3.576 (2.505-5.105)", "2.30e-12"]],
        table_id="table-5",
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
            f'<strong>Paragraph context:</strong> {esc(comment["anchor_paragraph_context"])}<br>'
            f'<a class="comment-backlink" href="#comment-anchor-{esc(comment_id)}">Back to comment anchor</a></div></div>'
        )
    return "".join(parts)


LITERATURE_ENTRIES = [
    ("Chen D, Alnassar SA, Avison KE, Huang RS, Raman S. Large Language Model Applications for Health Information Extraction in Oncology: Scoping Review. JMIR Cancer. 2025;11:e65984.", "https://doi.org/10.2196/65984", "This review maps 24 oncology information-extraction studies and provides the broad literature frame used to limit our novelty claims."),
    ("Sushil M, Kennedy VE, Mandair D, Miao BY, Zack T, Butte AJ. CORAL: Expert-Curated Oncology Reports to Advance Language Model Inference. NEJM AI. 2024;1(4).", "https://doi.org/10.1056/AIdbp2300110", "CORAL supplies our real longitudinal breast and pancreatic oncology notes and establishes the zero-shot benchmark that our harness extends."),
    ("Wiest IC, Ferber D, Zhu J, et al. Privacy-preserving large language models for structured medical information retrieval. npj Digital Medicine. 2024;7:257.", "https://doi.org/10.1038/s41746-024-01233-2", "This study shows that locally deployed open models, prompt design, and constrained output can support privacy-preserving clinical extraction."),
    ("Bhattarai K, Oh IY, Sierra JM, et al. Leveraging GPT-4 for identifying cancer phenotypes in electronic health records: a performance comparison between GPT-4, GPT-3.5-turbo, Flan-T5, Llama-3-8B, and spaCy's rule-based and machine learning-based methods. JAMIA Open. 2024;7(3):ooae060.", "https://doi.org/10.1093/jamiaopen/ooae060", "This longitudinal lung-cancer study compares model families and rule-based methods, while our experiment holds the model fixed and changes the inference workflow."),
    ("Tariq A, Sikha M, Kurian AW, et al. Open-Source Hybrid Large Language Model Integrated System for Extraction of Breast Cancer Treatment Pathway From Free-Text Clinical Notes. JCO Clinical Cancer Informatics. 2025;9:e2500002.", "https://doi.org/10.1200/CCI-25-00002", "This large hybrid breast-cancer system has external validation and supervised fine-tuning, providing a stronger scale comparator but a different adaptation strategy."),
    ("Dao N, Quesada L, Hassan SM, et al. Generative artificial intelligence for automated data extraction from unstructured medical text. JAMIA Open. 2025;8(5):ooaf097.", "https://doi.org/10.1093/jamiaopen/ooaf097", "Its engineered preload, schema and range validation, and retry loop are the closest published architectural precedent for our verification stages."),
    ("Zhang K, Huang T, Malin BA, et al. Introducing mCODEGPT as a zero-shot information extraction from clinical free text data tool for cancer research. Communications Medicine. 2025;5:422.", "https://doi.org/10.1038/s43856-025-01116-x", "mCODEGPT shows that hierarchical prompting can outperform a single-step strategy, although it uses synthetic oncology notes and does not evaluate a full verification harness."),
    ("Grothey B, Odenkirchen J, Brkic A, et al. Comprehensive testing of large language models for extraction of structured data in pathology. Communications Medicine. 2025;5:96.", "https://doi.org/10.1038/s43856-025-00808-8", "This prostate-pathology study documents substantial effects from model, prompt, language, and quantization choices on a narrower report type."),
    ("Qwen, Yang A, Yang B, et al. Qwen2.5 Technical Report. arXiv:2412.15115. 2025.", "https://arxiv.org/abs/2412.15115", "This report documents the base model family used in both study conditions; the inference harness is our added system layer."),
    ("Huang J, Yang DM, Rong R, et al. A critical assessment of using ChatGPT for extracting structured data from clinical notes. npj Digital Medicine. 2024;7:106.", "https://doi.org/10.1038/s41746-024-01079-8", "Iterative prompt engineering achieved strong pathology extraction but still exposed specialized terminology and TNM-staging errors that motivate explicit clinical checks."),
    ("van Koevorden JW, Aben N, Struben V, et al. Validating large language model-assisted data extraction from clinical notes. ESMO Real World Data and Digital Oncology. 2026;12:100718.", "https://doi.org/10.1016/j.esmorw.2026.100718", "Six experts evaluated 29 extracted categories in head and neck oncology, making this a close clinical-validation comparator with stronger reference-based error classification than our preference study."),
    ("Corso F, Peppoloni V, Mazzeo L, et al. Clinician expertise and prompt engineering enhance cancer information extraction in electronic health records by small language models. Communications Medicine. 2026.", "https://doi.org/10.1038/s43856-026-01790-5", "This study shows that few-shot design and clinical expertise improve locally deployable models, while our work adds multi-stage verification and deterministic cross-field rules."),
    ("Abhyankar S, Rao RM, Salehi M, Liang JJ, Deckard J, Sagiraju S. Large Language Models to Extract Cancer Staging Data From Clinical Documentation at Scale. JCO Clinical Cancer Informatics. 2026;10:e2500388.", "https://doi.org/10.1200/CCI-25-00388", "This study used 700 annotated notes for model development and testing, then processed more than two million notes across five cancers; our contribution instead concerns inference-time controls across a broader schema without weight updates."),
    ("Passweg LP, Schwenke JM, Schönenberger CM, et al. Data Extraction From Oncology Imaging Reports by Large Language Models: A Comparative Accuracy Study. JCO Clinical Cancer Informatics. 2026;10:e2600002.", "https://doi.org/10.1200/CCI-26-00002", "Its senior-physician-adjudicated comparison found metastasis classification easier than treatment-response assessment, matching the field-dependent difficulty seen in our results."),
    ("Dubey M, Chong KJ, Pun YR, et al. Prompt Engineering for Eastern Cooperative Oncology Group Status Extraction: Comparing Large Language Model Techniques. JCO Clinical Cancer Informatics. 2026;10:e2500226.", "https://doi.org/10.1200/CCI-25-00226", "Its comparison of simple prompting, chain-of-thought, double filtering, and rules provides direct evidence that inference design changes oncology extraction quality even when the target is a single field."),
    ("Zheng L, Chiang WL, Sheng Y, et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems. 2023;36:46595-46623.", "https://doi.org/10.52202/075280-2020", "This work supports LLM judges as scalable development tools but documents position, verbosity, self-enhancement, and reasoning biases that preclude treating them as our clinical endpoint."),
    ("Wang P, Li L, Chen L, et al. Large Language Models are not Fair Evaluators. Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. 2024:9440-9450.", "https://doi.org/10.18653/v1/2024.acl-long.511", "This paper isolates positional bias in LLM comparison judgments and reinforces the need for human final evaluation and cautious interpretation of automated review."),
    ("Gallifant J, Afshar M, Ameen S, et al. The TRIPOD-LLM reporting guideline for studies using large language models. Nature Medicine. 2025;31:60-69.", "https://doi.org/10.1038/s41591-024-03425-5", "TRIPOD-LLM supports explicit reporting of models, prompts, evaluation procedures, task-specific metrics, and human oversight in this pilot."),
    ("Tam TYC, Sivarajkumar S, Kapoor S, et al. A framework for human evaluation of large language models in healthcare derived from literature review. npj Digital Medicine. 2024;7:258.", "https://doi.org/10.1038/s41746-024-01258-7", "Its review of 142 studies and QUEST framework directly inform evaluator selection, rating dimensions, reliability assessment, and adjudication for a stronger follow-up study."),
    ("Estevez M, Singh N, Dyson L, et al. Ensuring Reliability of Curated Electronic Health Record-Derived Data: The Validation of Accuracy for Large Language Model-/Machine Learning-Extracted Information and Data (VALID) Framework. JCO Clinical Cancer Informatics. 2026;10:e2500215.", "https://doi.org/10.1200/CCI-25-00215", "VALID requires variable-level expert-reference benchmarking, internal consistency and plausibility checks, and replication, clarifying why our preference endpoint is not yet an absolute correctness assessment."),
    ("Kong HJ. Managing Unstructured Big Data in Healthcare System. Healthcare Informatics Research. 2019;25(1):1.", "https://doi.org/10.4258/hir.2019.25.1.1", "This article supports the broad estimate that much healthcare information remains unstructured; the estimate is not oncology-specific."),
    ("Rule A, Bedrick S, Chiang MF, Hribar MR. Length and Redundancy of Outpatient Progress Notes Across a Decade at an Academic Medical Center. JAMA Network Open. 2021;4(7):e2115334.", "https://doi.org/10.1001/jamanetworkopen.2021.15334", "This study of nearly 3 million outpatient notes documents increasing note length and redundancy and the contribution of copied and templated text."),
    ("Hein D, Christie A, Holcomb M, et al. Iterative refinement and goal articulation to optimize large language models for clinical information extraction. npj Digital Medicine. 2025;8:301.", "https://doi.org/10.1038/s41746-025-01686-z", "This study provides a close precedent for human-in-the-loop refinement, error ontology development, and general instructions rather than case-specific fixes."),
    ("Van Veen D, Van Uden C, Blankemeier L, et al. Adapted large language models can outperform medical experts in clinical text summarization. Nature Medicine. 2024;30:1134-1142.", "https://doi.org/10.1038/s41591-024-02855-5", "Its blinded reader study used a scale with an equivalent midpoint, supporting retention of ties in comparative clinical evaluation."),
    ("Ayers JW, Poliak A, Dredze M, et al. Comparing Physician and Artificial Intelligence Chatbot Responses to Patient Questions Posted to a Public Social Media Forum. JAMA Internal Medicine. 2023;183(6):589-596.", "https://doi.org/10.1001/jamainternmed.2023.1838", "This physician-rated comparison reported the preference proportion with a confidence interval, providing a clear precedent for probability-scale reporting."),
    ("Goh E, Gallo R, Hom J, et al. Large Language Model Influence on Diagnostic Reasoning. JAMA Network Open. 2024;7(10):e2440969.", "https://doi.org/10.1001/jamanetworkopen.2024.40969", "Its mixed-effects analysis included participant and case effects, illustrating how repeated clinical observations should be modeled rather than treated as independent."),
]


def literature_workspace() -> str:
    entries = "".join(
        f'<li id="lit-{index}" data-literature-entry="{index}">{esc(citation)} '
        f'<a href="{esc(url)}">{esc(url)}</a>'
        f'<span class="reference-note"><strong>Draft relevance:</strong> {esc(relevance)}</span></li>'
        for index, (citation, url, relevance) in enumerate(LITERATURE_ENTRIES, 1)
    )
    return (
        '<details class="fold literature-workspace" id="literature-workspace">'
        '<summary>Author literature-review workspace (not manuscript body)</summary><div class="fold-body">'
        '<p class="workspace-note">These historical entries support drafting and discussion development; every citation and link requires pre-submission verification.</p>'
        f'<ol>{entries}</ol></div></details>'
    )


def technical_inserts_after(block_id: str) -> str:
    inserts = {
        "P019": manuscript_paragraph("The study workflow is shown in Figure 1.", paragraph_id="workflow-cross-reference"),
        "P038": figure("1", "Development, adaptation, and evaluation workflow", "Development, transfer, and evaluation pathway. Breast-cancer physician review informed the inference harness. The failure history and clinical rubric then supported author-controlled PDAC adaptation without target-domain physician review of outputs. Four physicians compared the final harness with the same-model single-prompt baseline on the 40-note holdout set."),
        "P082": manuscript_paragraph("Exploratory core-field analysis used one joint population-averaged logistic GEE on 495 directional core judgments. Field, physician, and cancer type were fixed effects; repeated judgments were clustered by note with an exchangeable working correlation. Field-specific probabilities were standardized over the physician and cancer distribution observed for that field. Breast type and receptor status applied only to breast-cancer notes. Ties were excluded. These estimates describe preference rather than accuracy, and no pairwise field tests or multiplicity claims were made.", paragraph_id="core-field-statistical-analysis"),
        "P085": table_3() + table_4(),
        "P087": figure("2", "Physician preference overview", "Physician preferences across eight physician-by-cancer evaluations. Panel A retains harness, baseline, and tie judgments. Panel B shows the harness share among directional judgments."),
        "P092": table_5(),
        "P095": manuscript_paragraph("Pairwise exact agreement ranged from 73.1% to 82.9%, with Cohen's kappa ranging from 0.511 to 0.646 (Figure 3 and Supplementary Table S2).", paragraph_id="agreement-results") + figure("3", "Inter-physician agreement", "Pairwise exact agreement and Cohen's kappa among four physicians in breast cancer and PDAC. Each breast-cancer pair shared 280 judgments, and each PDAC pair shared 260 judgments."),
        "P097": figure("4", "Core-field raw and adjusted preference estimates", "Core-field preference results. Panel A retains the tie-inclusive raw distribution for seven physician-prioritized fields. Panel B shows core-field adjusted harness-preference probabilities with 95% confidence intervals from one joint exploratory GEE fitted to 495 directional core judgments. The model included field, physician, and cancer fixed effects, clustered repeated judgments by note with an exchangeable working correlation, and standardized each field over its observed physician and cancer distribution. Breast type and receptor status applied only to breast-cancer notes. Ties were excluded from Panel B. The estimates report preference, not accuracy. No pairwise field tests or multiplicity claims were made."),
    }
    return inserts.get(block_id, "")


def build_scientific_sections(data: dict[str, object]) -> tuple[str, str]:
    paragraphs = {item["id"]: item for item in data["paragraphs"]}
    tables = {item["id"]: item for item in data["tables"]}
    comment_ids_by_paragraph: dict[str, list[str]] = {}
    for comment in data["comments"]:
        comment_ids_by_paragraph.setdefault(str(comment["anchor_paragraph_id"]), []).append(str(comment["id"]))
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
            rendered = render_physician_paragraph(paragraph, comment_ids_by_paragraph.get(block_id, []))
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
                current_parts.append(render_native_table(tables[table_id], "Table 1", "Inference harness components"))
            elif table_id == "T2":
                current_parts.append(render_native_table(tables[table_id], "Table 2", "Staged technical ablation conditions"))
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
        ("v1.2", "2026-10-26", "Converted the review draft into a clean manuscript view, made the main tables sequential, corrected all approved table and figure cross-references, retained the 12 Word comments in a neutral end section, and restored the 26-entry author literature-review workspace."),
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
    links = [("#section-introduction", "Introduction", False), ("#section-methods", "Methods", False)]
    for heading in ["Study Design", "Dataset", "Baseline model and model development", "Inference Harness", "Clinician-in-the-loop breast-cancer refinement", "Model-assist pancreatic-cancer refinement", "Clinical evaluation data collection and scoring rubric", "Staged Technical ablation", "Statistical analysis"]:
        links.append(("#" + re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-"), heading, True))
    links.extend([("#section-results", "Results", False), ("#performance-across-clinical-extraction-fields", "Performance across Clinical Extraction Fields", True), ("#section-discussion", "Discussion", False), ("#supplementary-material", "Supplementary material", False), ("#physician-comments", "Comments", False), ("#version-history", "Version history", False), ("#literature-workspace", "Literature-review workspace", False)])
    return '<nav class="toc" data-origin="technical-insert"><strong>Contents</strong>' + "".join(f'<a class="{"sub" if sub else ""}" href="{href}" data-origin="technical-insert">{esc(label)}</a>' for href, label, sub in links) + "</nav>"


def build_html(data: dict[str, object]) -> tuple[str, str]:
    sections, markdown_body = build_scientific_sections(data)
    article = f"""
<article>
  <h1>Physician-authored manuscript draft</h1>
  <p class="subtitle">English working manuscript · Version 1.2 · October 26, 2026</p>
  <p class="current-change"><strong>Current change:</strong> clean manuscript view with sequential table numbering, approved cross-reference corrections, neutral Word-comment navigation, and the restored author literature-review workspace.</p>
  {sections}
  {build_supplement()}
  <details class="fold" id="physician-comments"><summary>Comments</summary><div class="fold-body">{render_comments(data)}</div></details>
  <details class="fold" id="version-history"><summary>Version history</summary><div class="fold-body">{version_history()}</div></details>
  {literature_workspace()}
</article>
"""
    document = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Physician-authored manuscript draft v1.2</title><style>{CSS}</style></head><body><div class="layout">{toc()}{article}</div><script>{SCRIPT}</script></body></html>'
    markdown = f"""<!-- GENERATED FILE. The canonical physician Word extraction is the prose authority. -->
# Physician-authored manuscript draft

English working manuscript. Version 1.2, October 26, 2026.

Current change: clean manuscript view with sequential table numbering, approved cross-reference corrections, neutral Word-comment navigation, and the restored author literature-review workspace.

The canonical physician blocks below mirror `{SOURCE_DOCX.name}` for audit purposes. The clean HTML applies only the approved cross-reference and reporting edits while preserving all source blocks in its hidden audit representation.

{markdown_body}

## Supplementary material

Generated technical content: Supplementary Tables S1-S3 and Supplementary Figures S1-S2.

## Physician comments

All 12 Word comments, including anchor text and paragraph context, are rendered in the HTML from the canonical JSON extraction.

## Author literature-review workspace

The HTML restores 26 annotated literature-review entries as a default-closed author workspace outside the manuscript body.
"""
    return document, markdown


class PreservationParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, str | None, str | None, list[str], bool, bool]] = []
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
        self.literature_entries: list[str] = []
        self.comment_anchor_ids: list[str] = []
        self.comment_backlinks: list[str] = []
        self.hidden_depth = 0
        self.non_manuscript_depth = 0
        self.visible_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"): self.ids.append(str(values["id"]))
        if values.get("href", "").startswith("#"): self.hrefs.append(str(values["href"])[1:])
        if values.get("data-main-figure"): self.main_figures.append(str(values["data-main-figure"]))
        if values.get("data-supplementary-figure"): self.supplementary_figures.append(str(values["data-supplementary-figure"]))
        if values.get("data-literature-entry"): self.literature_entries.append(str(values["data-literature-entry"]))
        if str(values.get("id", "")).startswith("comment-anchor-"): self.comment_anchor_ids.append(str(values["id"]))
        if "comment-backlink" in str(values.get("class", "")).split(): self.comment_backlinks.append(str(values.get("href", "")))
        if "table-wrap" in str(values.get("class", "")).split():
            if values.get("data-origin") != "technical-insert": raise AssertionError("Generated table container lacks technical-insert provenance")
            self.technical_table_containers.append(str(values.get("id", "")))
        capture_type = None
        capture_id = None
        source_block = values.get("data-source-block")
        if values.get("data-origin") == "physician" and source_block:
            capture_type = "cell" if str(source_block).startswith("T") else "paragraph"
            capture_id = str(source_block)
        elif values.get("data-comment-text-id") is not None:
            capture_type = "comment-text"
            capture_id = str(values["data-comment-text-id"])
        starts_hidden = "hidden" in values or "source-audit" in str(values.get("class", "")).split()
        if starts_hidden: self.hidden_depth += 1
        starts_non_manuscript = values.get("id") in {"physician-comments", "version-history", "literature-workspace"}
        if starts_non_manuscript: self.non_manuscript_depth += 1
        self.stack.append((tag, capture_type, capture_id, [], starts_hidden, starts_non_manuscript))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs); self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self.hidden_depth == 0 and self.non_manuscript_depth == 0: self.visible_parts.append(data)
        for _, capture_type, _, parts, _, _ in self.stack:
            if capture_type in {"paragraph", "cell", "comment-text"}: parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if not self.stack: return
        open_tag, capture_type, capture_id, parts, starts_hidden, starts_non_manuscript = self.stack.pop()
        if open_tag != tag: return
        if capture_type == "paragraph" and capture_id:
            self.paragraphs[capture_id] = "".join(parts); self.paragraph_order.append(capture_id)
        elif capture_type == "cell" and capture_id:
            self.cells[capture_id] = "".join(parts); self.cell_order.append(capture_id)
        elif capture_type == "comment-text" and capture_id: self.comments[capture_id] = "".join(parts)
        if starts_hidden: self.hidden_depth -= 1
        if starts_non_manuscript: self.non_manuscript_depth -= 1

    @property
    def visible_text(self) -> str:
        return " ".join("".join(self.visible_parts).split())


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
    expected_table_containers = ["table-1", "table-2", "table-3", "table-4", "table-5", "supplementary-table-s1", "supplementary-table-s2", "supplementary-table-s3"]
    if parser.technical_table_containers != expected_table_containers: raise AssertionError(f"Unexpected generated table containers: {parser.technical_table_containers}")
    for label in ["Table 1", "Table 2", "Table 3", "Table 4", "Table 5", "Supplementary Table S1", "Supplementary Table S2", "Supplementary Table S3"]:
        if document.count(f"<strong>{label}.</strong>") != 1: raise AssertionError(f"Table label missing or duplicated: {label}")
    if parser.literature_entries != [str(index) for index in range(1, 27)]: raise AssertionError("The 26-entry literature workspace is incomplete")
    if '<details class="fold literature-workspace" id="literature-workspace">' not in document: raise AssertionError("Literature workspace is missing")
    if "if(target instanceof HTMLDetailsElement||target.tagName==='DETAILS')target.open=true" not in document or "target.focus({preventScroll:true})" not in document:
        raise AssertionError("Closed-details target reveal behavior is missing")
    if "@media print" not in document or ".fold:not([open])>.fold-body,.version-entry:not([open])>.version-entry-body{display:block!important}" not in document:
        raise AssertionError("Print view does not force closed manuscript content to render")
    expected_comment_anchors = [f"comment-anchor-{item['id']}" for item in data["comments"]]
    if parser.comment_anchor_ids != expected_comment_anchors: raise AssertionError("Comment anchors are incomplete")
    if parser.comment_backlinks != [f"#comment-anchor-{item['id']}" for item in data["comments"]]: raise AssertionError("Comment backlinks are incomplete")
    if 'id="draft-metadata"' in document or "Draft information and source provenance" in document: raise AssertionError("Removed draft-information block remains")
    if "Version 1.2 · October 26, 2026" not in document or "draft v1.2" not in document: raise AssertionError("Version 1.2 metadata is missing")
    if document.count('class="version-entry"') != 12 or any(f'id="version-{item}"' not in document for item in ["1-2", "1-1", "1-0", "0-25", "0-24", "0-23", "0-22", "0-21", "0-20", "0-19", "0-18", "0-17"]):
        raise AssertionError("Nested version history is incomplete")
    if 'lang="zh"' in document or 'id="chinese-version"' in document: raise AssertionError("Chinese manuscript shell remains")
    visible_text_lower = parser.visible_text.lower()
    if re.search(r"[\u4e00-\u9fff]", parser.visible_text): raise AssertionError("Chinese text remains in the visible manuscript")
    for forbidden in ["technical insert", "physician source note", "review marker", "[todo]", "figure 4 删掉", "difference from 50%", "(citation)"]:
        if forbidden in visible_text_lower: raise AssertionError(f"Clean manuscript contains review artifact: {forbidden}")
    for obsolete in ["author-queries", "review-marker", "source-highlight", "physician-editorial", "technical-note", "class=\"badge\""]:
        if obsolete in document: raise AssertionError(f"Removed review UI remains: {obsolete}")
    for source_block in HIDDEN_SOURCE_BLOCKS:
        if f'data-source-block="{source_block}"' not in document or f'id="{source_block}"' in document:
            raise AssertionError(f"Hidden source placeholder audit is incomplete: {source_block}")
    required_crossrefs = {
        "P030": "#table-1", "P063": "#supplementary-table-s1", "P068": "#table-2",
        "P085": "#table-3", "P086": "#figure-s1", "P092": "#table-5",
        "P095": "#supplementary-table-s3", "P097": "#figure-4", "P100": "#figure-s1", "P117": "#figure-s2",
    }
    for paragraph_id, href in required_crossrefs.items():
        match = re.search(rf'id="{paragraph_id}"[^>]*>(.*?)</p>', document, flags=re.S)
        if not match or f'href="{href}"' not in match.group(1): raise AssertionError(f"Cross-reference missing from {paragraph_id}: {href}")
    if 'href="#table-4"' not in re.search(r'id="P085"[^>]*>(.*?)</p>', document, flags=re.S).group(1) or 'href="#figure-2"' not in re.search(r'id="P085"[^>]*>(.*?)</p>', document, flags=re.S).group(1):
        raise AssertionError("P085 does not cite Tables 3-4 and Figure 2")
    if 'href="#figure-3"' not in document or 'href="#supplementary-table-s2"' not in document: raise AssertionError("Agreement Results cross-references are missing")
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
