#!/usr/bin/env python3
"""Generate four main figures and two supplementary figures for the manuscript.

SVG primitives keep the output deterministic across machines. The lightweight
XLSX reader is sufficient for the two physician workbooks and avoids requiring
openpyxl for figure regeneration.
"""

from __future__ import annotations

import csv
import html
import re
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree as ET


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT = HERE / "draft_figures"
ABLATION_SUMMARY = ROOT / "results/staged_ablation_annotated_20261001_093433/JUDGE_CODEX_DECODED_ANALYSIS.md"
RATER_FILES = {
    "Simo": [HERE / "simo_breast_pdac_blind_scores_20260921.csv"],
    "Kevin": [HERE / "kevin_breast_pdac_blind_scores_20261005.csv"],
    "Bolun": [
        HERE / "bolun_breast_blind_scores_20260921.xlsx",
        HERE / "bolun_pdac_blind_scores_and_letter_selection_20261006.xlsx",
    ],
    "Zhengrui": [HERE / "zhengrui_breast_pdac_blind_scores_20261005.csv"],
}
ADJUSTED_RESULTS = HERE / "adjusted_gee_results.csv"
CORE_FIELD_RESULTS = HERE / "core_field_adjusted_gee.csv"

# Fixed colorblind-safe visual language across all panels.
PL = "#0072B2"
BL = "#D55E00"
TIE = "#B8B8B8"
PHYSICIAN = "#009E73"
LLM = "#7A5FA8"
TEXT = "#222222"
MUTED = "#5E6870"
NEUTRAL = "#65737E"
GRID = "#E6E6E6"
PANEL_FILL = "#FAFBFC"
PANEL_STROKE = "#D7DEE3"
HARNESS_SOFT = "#E7F2F8"
BASELINE_SOFT = "#FBECE5"
PHYSICIAN_SOFT = "#E6F5F0"
LLM_SOFT = "#F0ECF5"
NEUTRAL_SOFT = "#F0F2F3"

BREAST_FIELDS = [
    "current_meds",
    "stage",
    "distant_met",
    "metastasis",
    "response",
    "type_receptor",
    "genetic_results",
    "genetic_plan",
    "supportive_meds",
    "procedure_plan",
    "imaging_plan",
    "lab_plan",
    "medication_plan",
    "recent_changes",
]
PDAC_FIELDS = [field for field in BREAST_FIELDS if field != "type_receptor"]
CORE_BREAST = BREAST_FIELDS[:7]
CORE_PDAC = [
    "current_meds",
    "stage",
    "distant_met",
    "metastasis",
    "response",
    "genetic_results",
]
FIELD_LABELS = {
    "current_meds": "Active anticancer medications",
    "stage": "Stage",
    "distant_met": "Distant metastasis",
    "metastasis": "Regional / overall metastasis",
    "response": "Treatment response",
    "type_receptor": "Breast type / receptors",
    "genetic_results": "Molecular / genetic results",
}
CORE_FIELD_ORDER = [
    "current_meds",
    "stage",
    "response",
    "distant_met",
    "genetic_results",
    "metastasis",
    "type_receptor",
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - 64
    return value - 1


def load_xlsx(path: Path) -> list[dict[str, str]]:
    """Read the ``blind_scores`` sheet with the Python standard library."""
    main_ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    doc_rel_ns = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    with zipfile.ZipFile(path) as archive:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{main_ns}}}si"):
                shared.append("".join(node.text or "" for node in item.iter(f"{{{main_ns}}}t")))

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationship_id = None
        for sheet in workbook.find(f"{{{main_ns}}}sheets"):
            if sheet.attrib.get("name") == "blind_scores":
                relationship_id = sheet.attrib[f"{{{doc_rel_ns}}}id"]
                break
        if relationship_id is None:
            raise ValueError(f"blind_scores sheet not found in {path.name}")

        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for relation in rels.findall(f"{{{rel_ns}}}Relationship"):
            if relation.attrib["Id"] == relationship_id:
                target = relation.attrib["Target"]
                break
        if target is None:
            raise ValueError(f"worksheet relationship missing in {path.name}")
        sheet_path = target.lstrip("/")
        if not sheet_path.startswith("xl/"):
            sheet_path = "xl/" + sheet_path
        sheet = ET.fromstring(archive.read(sheet_path))

        matrix: list[list[str]] = []
        for row in sheet.findall(f".//{{{main_ns}}}row"):
            values: list[str] = []
            for cell in row.findall(f"{{{main_ns}}}c"):
                index = _column_index(cell.attrib["r"])
                while len(values) <= index:
                    values.append("")
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{main_ns}}}v")
                if cell_type == "inlineStr":
                    inline = cell.find(f"{{{main_ns}}}is")
                    value = "" if inline is None else "".join(
                        node.text or "" for node in inline.iter(f"{{{main_ns}}}t")
                    )
                elif value_node is None:
                    value = ""
                elif cell_type == "s":
                    value = shared[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[index] = value.strip()
            matrix.append(values)
    if not matrix:
        return []
    headers = matrix[0]
    return [
        {key: row[index].strip() if index < len(row) else "" for index, key in enumerate(headers)}
        for row in matrix[1:]
        if row and row[0]
    ]


def load_raters() -> dict[str, list[dict[str, str]]]:
    raters: dict[str, list[dict[str, str]]] = {}
    for name, paths in RATER_FILES.items():
        rows: list[dict[str, str]] = []
        for path in paths:
            rows.extend(load_xlsx(path) if path.suffix == ".xlsx" else load_csv(path))
        keys = [(row["sample"], row["field"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate sample-field judgments for {name}")
        raters[name] = rows
    return raters


def subset(rows: list[dict[str, str]], prefix: str, fields: list[str]) -> list[dict[str, str]]:
    allowed = set(fields)
    return [row for row in rows if row["sample"].startswith(prefix) and row["field"] in allowed]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#FFFFFF"/>',
        '<defs>',
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#65737E"/></marker>',
        '<marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#0072B2"/></marker>',
        '<marker id="arrow-purple" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#7A5FA8"/></marker>',
        '<marker id="arrow-green" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#009E73"/></marker>',
        '<marker id="arrow-orange" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#D55E00"/></marker>',
        '</defs>',
        f'<g font-family="Arial, Helvetica, sans-serif" fill="{TEXT}">',
    ]


def svg_end(out: list[str], path: Path) -> None:
    out.extend(["</g>", "</svg>"])
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def text(out: list[str], x: float, y: float, value: str, size: int = 16, *, anchor: str = "start", weight: int = 400, fill: str = TEXT, italic: bool = False) -> None:
    # The manuscript embeds 1,320-1,500 px SVGs at about 780 px. A 16 px
    # minimum keeps labels legible at that display width.
    size = max(size, 16)
    style = ' font-style="italic"' if italic else ""
    out.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}"{style}>{esc(value)}</text>'
    )


def multiline(out: list[str], x: float, y: float, lines: list[str], size: int = 16, *, anchor: str = "middle", weight_first: bool = False, fill: str = TEXT, leading: float = 1.25) -> None:
    size = max(size, 16)
    for index, line in enumerate(lines):
        text(out, x, y + index * size * leading, line, size, anchor=anchor, weight=700 if weight_first and index == 0 else 400, fill=fill)


def round_rect(out: list[str], x: float, y: float, width: float, height: float, *, fill: str = "white", stroke: str = PANEL_STROKE, radius: int = 10, stroke_width: float = 1.5) -> None:
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>')


def panel(out: list[str], x: float, y: float, width: float, height: float, letter: str, title: str) -> None:
    round_rect(out, x, y, width, height, fill=PANEL_FILL, stroke=PANEL_STROKE, radius=13)
    out.append(f'<circle cx="{x + 27:.1f}" cy="{y + 28:.1f}" r="17" fill="#263C4A"/>')
    text(out, x + 27, y + 35, letter, 20, anchor="middle", weight=700, fill="white")
    text(out, x + 55, y + 35, title, 21, weight=700)


def box(out: list[str], x: float, y: float, width: float, height: float, lines: list[str], *, fill: str = "white", stroke: str = PANEL_STROKE, size: int = 15, weight_first: bool = True) -> None:
    size = max(size, 16)
    round_rect(out, x, y, width, height, fill=fill, stroke=stroke, radius=9)
    start = y + height / 2 - (len(lines) - 1) * size * 0.64 + size * 0.34
    multiline(out, x + width / 2, start, lines, size, weight_first=weight_first)


def arrow(out: list[str], x1: float, y1: float, x2: float, y2: float, *, color: str = "#65737E", dashed: bool = False, curved: bool = False) -> None:
    dash = ' stroke-dasharray="8 7"' if dashed else ""
    markers = {LLM: "arrow-purple", PL: "arrow-blue", PHYSICIAN: "arrow-green", BL: "arrow-orange"}
    marker = markers.get(color, "arrow")
    if curved:
        midpoint = min(y1, y2) - 28
        path = f"M{x1:.1f},{y1:.1f} C{x1:.1f},{midpoint:.1f} {x2:.1f},{midpoint:.1f} {x2:.1f},{y2:.1f}"
        out.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#{marker})"{dash}/>')
    else:
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="2.2" marker-end="url(#{marker})"{dash}/>')


def plot_development_pathway() -> None:
    width, height = 1500, 1180
    out = svg_start(width, height)
    panel(out, 30, 25, 1440, 200, "A", "Data partitions and isolation")
    data_cards = [
        (90, "Breast development", ["100 unannotated notes", "Harness development"], NEUTRAL, NEUTRAL_SOFT, "B"),
        (545, "PDAC development", ["100 unannotated notes", "Cross-cancer adaptation"], NEUTRAL, NEUTRAL_SOFT, "C"),
        (1000, "Locked holdout", ["20 breast + 20 PDAC", "No development use"], NEUTRAL, NEUTRAL_SOFT, "D"),
    ]
    for x, heading, lines, color, fill, route in data_cards:
        box(out, x, 78, 390, 92, [heading] + lines, fill=fill, stroke=color, size=16)
        text(out, x + 195, 197, f"to panel {route}", 13, anchor="middle", weight=700, fill=color)

    panel(out, 30, 250, 700, 480, "B", "Physician-informed breast-cancer development")
    bx, by, bw, bh, gap = 70, 322, 140, 82, 27
    breast_nodes = [
        (["Development", "note"], "white", PANEL_STROKE),
        (["Frozen model", "+ harness"], HARNESS_SOFT, PL),
        (["Structured", "outputs"], "white", PL),
        (["Physician +", "investigator", "source review"], PHYSICIAN_SOFT, PHYSICIAN),
    ]
    for index, (lines, fill, stroke) in enumerate(breast_nodes):
        x = bx + index * (bw + gap)
        box(out, x, by, bw, bh, lines, fill=fill, stroke=stroke, size=13)
        if index:
            arrow(out, x - gap + 5, by + bh / 2, x - 5, by + bh / 2)
    box(out, 500, 486, 190, 82, ["Failure taxonomy", "+ clinical rubric"], fill=PHYSICIAN_SOFT, stroke=PHYSICIAN, size=14)
    box(out, 285, 486, 190, 82, ["Prompt, gate, hook", "revision"], fill=HARNESS_SOFT, stroke=PL, size=14)
    box(out, 70, 486, 190, 82, ["Regression testing"], fill="white", stroke=PL, size=15)
    arrow(out, 641, 404, 595, 486)
    arrow(out, 500, 527, 475, 527)
    arrow(out, 285, 527, 260, 527)
    out.append(f'<path d="M165,568 V605 H487 V425 C487,410 400,404 307,404" fill="none" stroke="{PL}" stroke-width="2.2" marker-end="url(#arrow-blue)"/>')
    text(out, 377, 710, "Iterative review; model weights remain frozen", 13, anchor="middle", fill=MUTED)

    panel(out, 770, 250, 700, 480, "C", "Model-assisted PDAC adaptation")
    box(out, 800, 300, 640, 62, ["Transferred failure history + clinical rubric + reusable harness"], fill=LLM_SOFT, stroke=LLM, size=15)
    text(out, 1118, 385, "knowledge transfer from B", 13, anchor="middle", weight=700, fill=LLM)
    out.append(f'<line x1="730" y1="515" x2="800" y2="331" stroke="{LLM}" stroke-width="2.3" stroke-dasharray="8 7" marker-end="url(#arrow-purple)"/>')
    cx, cy, cw, ch, cg = 802, 416, 142, 78, 25
    pdac_nodes = [
        (["100 PDAC", "dev notes"], "white", PANEL_STROKE),
        (["Harness", "outputs"], HARNESS_SOFT, PL),
        (["Source-grounded", "LLM review"], LLM_SOFT, LLM),
        (["Candidate", "revision"], LLM_SOFT, LLM),
    ]
    for index, (lines, fill, stroke) in enumerate(pdac_nodes):
        x = cx + index * (cw + cg)
        box(out, x, cy, cw, ch, lines, fill=fill, stroke=stroke, size=13)
        if index:
            arrow(out, x - cg + 4, cy + ch / 2, x - 4, cy + ch / 2, color=LLM if index >= 2 else PL)
    box(out, 980, 560, 200, 75, ["Investigator", "decision"], fill="white", stroke="#65737E", size=15)
    box(out, 1240, 560, 180, 75, ["Regression", "testing"], fill=HARNESS_SOFT, stroke=PL, size=15)
    arrow(out, 1315, 494, 1110, 560)
    arrow(out, 1180, 598, 1240, 598)
    out.append(
        f'<path d="M1330,635 C1330,645 1320,648 1310,648 H890 '
        f'C880,648 873,640 873,630 V494" fill="none" stroke="{PL}" '
        'stroke-width="2.2" marker-end="url(#arrow-blue)"/>'
    )
    round_rect(out, 840, 658, 560, 48, fill="white", stroke=NEUTRAL, radius=7)
    out.append(f'<circle cx="870" cy="682" r="15" fill="{PHYSICIAN}"/>')
    text(out, 870, 688, "P", 16, anchor="middle", weight=700, fill="white")
    text(out, 1125, 688, "No target-domain physician review of PDAC outputs", 14, anchor="middle", weight=700, fill=PHYSICIAN)

    panel(out, 30, 755, 1440, 390, "D", "Independent holdout evaluation")
    box(out, 75, 842, 230, 88, ["40 locked notes", "20 breast + 20 PDAC"], fill=NEUTRAL_SOFT, stroke=NEUTRAL, size=15)
    box(out, 365, 825, 260, 120, ["Same frozen", "Qwen2.5-32B", "+ same schema"], fill="white", stroke="#65737E", size=15)
    arrow(out, 305, 886, 365, 886)
    box(out, 705, 795, 235, 100, ["Inference harness", "+ supporting excerpts"], fill=HARNESS_SOFT, stroke=PL, size=15)
    box(out, 705, 935, 235, 100, ["Single-prompt", "baseline"], fill=BASELINE_SOFT, stroke=BL, size=15)
    arrow(out, 625, 875, 705, 845, color=PL)
    arrow(out, 625, 895, 705, 985, color=BL)
    box(out, 1010, 842, 220, 112, ["Identity-masked", "fixed A/B", "field comparison"], fill="white", stroke="#65737E", size=15)
    arrow(out, 940, 845, 1010, 885, color=PL)
    arrow(out, 940, 985, 1010, 915, color=BL)
    box(out, 1290, 795, 130, 100, ["4", "physicians"], fill=PHYSICIAN_SOFT, stroke=PHYSICIAN, size=15)
    box(out, 1290, 935, 130, 100, ["2,160 field", "judgments"], fill="white", stroke=PHYSICIAN, size=14)
    arrow(out, 1230, 898, 1290, 845, color=PHYSICIAN)
    arrow(out, 1355, 895, 1355, 935, color=PHYSICIAN)
    text(out, 750, 1110, "Primary analysis: logistic GEE on non-tie judgments, clustered by note", 15, anchor="middle", fill=MUTED)
    svg_end(out, OUTPUT / "figure1_development_pathway_rough.svg")


def legend(out: list[str], x: float, y: float, items: list[tuple[str, str]], *, size: int = 13, gap: int = 125) -> None:
    for index, (color, label) in enumerate(items):
        current = x + index * gap
        out.append(f'<rect x="{current:.1f}" y="{y - 12:.1f}" width="18" height="13" rx="2" fill="{color}"/>')
        text(out, current + 25, y, label, size)


def plot_evaluator_distribution(raters: dict[str, list[dict[str, str]]]) -> None:
    groups: list[tuple[str, list[dict[str, str]]]] = []
    for index, rows in enumerate(raters.values(), start=1):
        groups.append((f"Physician {index}  |  Breast", subset(rows, "b", BREAST_FIELDS)))
        groups.append((f"Physician {index}  |  PDAC", subset(rows, "p", PDAC_FIELDS)))
    width, height = 1500, 720
    out = svg_start(width, height)
    panel(out, 25, 20, 900, 655, "A", "All required judgments")
    panel(out, 950, 20, 525, 655, "B", "Harness share among directional judgments")
    legend(out, 380, 82, [(PL, "Harness"), (TIE, "Tie"), (BL, "Baseline")], gap=120)
    left, top, plot_w, bar_h, row_gap = 240, 112, 610, 42, 65
    for tick in range(0, 101, 20):
        x = left + plot_w * tick / 100
        out.append(f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" y2="{top + 7 * row_gap + bar_h}" stroke="{GRID}"/>')
        text(out, x, 655, f"{tick}%", 12, anchor="middle", fill=MUTED)
    shares: list[float] = []
    for index, (label, rows) in enumerate(groups):
        counts = Counter(row["score"] for row in rows)
        total = len(rows)
        y = top + index * row_gap
        text(out, left - 15, y + 27, label, 14, anchor="end")
        x = left
        for key, color, label_color in [("A", PL, "white"), ("TIE", TIE, TEXT), ("B", BL, "white")]:
            value = counts[key]
            segment = plot_w * value / total
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{segment:.1f}" height="{bar_h}" fill="{color}"/>')
            if segment >= 30:
                text(out, x + segment / 2, y + 27, str(value), 13, anchor="middle", weight=700, fill=label_color)
            elif value:
                text(out, x + segment / 2, y - 5, str(value), 11, anchor="middle", weight=700, fill=color)
            x += segment
        directional = counts["A"] + counts["B"]
        shares.append(counts["A"] / directional)
        text(out, 867, y + 27, f"n={total}", 12, fill=MUTED)

    x0, x1 = 1040, 1415
    for tick in [50, 60, 70, 80, 90, 100]:
        x = x0 + (tick - 50) / 50 * (x1 - x0)
        stroke = "#818C94" if tick == 50 else GRID
        dash = "4 4" if tick == 50 else "0"
        out.append(f'<line x1="{x:.1f}" y1="102" x2="{x:.1f}" y2="620" stroke="{stroke}" stroke-dasharray="{dash}"/>')
        text(out, x, 655, f"{tick}%", 12, anchor="middle", fill=MUTED)
    for index, share in enumerate(shares):
        y = top + index * row_gap + bar_h / 2
        x = x0 + (share * 100 - 50) / 50 * (x1 - x0)
        out.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x:.1f}" y2="{y:.1f}" stroke="#AEB7BD" stroke-width="3"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{PL}" stroke="white" stroke-width="2"/>')
        text(out, min(x + 14, 1450), y + 5, f"{share:.1%}", 13, weight=700, fill=PL)
    text(out, 1228, 92, "50% = no directional preference", 12, anchor="middle", fill=MUTED)
    svg_end(out, OUTPUT / "figure2_evaluator_distribution_rough.svg")


def pairwise_agreement(r1: list[dict[str, str]], r2: list[dict[str, str]], prefix: str, fields: list[str]) -> tuple[float, float, int]:
    m1 = {(row["sample"], row["field"]): row["score"] for row in subset(r1, prefix, fields)}
    m2 = {(row["sample"], row["field"]): row["score"] for row in subset(r2, prefix, fields)}
    keys = sorted(set(m1) & set(m2))
    total = len(keys)
    same = sum(m1[key] == m2[key] for key in keys)
    order = ["A", "TIE", "B"]
    counts_1 = Counter(m1[key] for key in keys)
    counts_2 = Counter(m2[key] for key in keys)
    observed = same / total
    expected = sum(counts_1[value] * counts_2[value] for value in order) / total**2
    kappa = (observed - expected) / (1 - expected)
    return observed, kappa, total


def interpolate_hex(low: str, high: str, fraction: float) -> str:
    fraction = min(max(fraction, 0.0), 1.0)
    lo = [int(low[index:index + 2], 16) for index in (1, 3, 5)]
    hi = [int(high[index:index + 2], 16) for index in (1, 3, 5)]
    rgb = [round(a + (b - a) * fraction) for a, b in zip(lo, hi)]
    return "#" + "".join(f"{value:02X}" for value in rgb)


def draw_agreement_matrix(out: list[str], raters: dict[str, list[dict[str, str]]], x0: int, y0: int, title: str, prefix: str, fields: list[str], n_per_pair: int) -> None:
    names = list(raters)
    values = {
        (a, b): pairwise_agreement(raters[a], raters[b], prefix, fields)
        for a, b in combinations(names, 2)
    }
    cell, left, top = 112, x0 + 155, y0 + 130
    text(out, x0 + 370, y0 + 42, title, 20, anchor="middle", weight=700)
    text(out, x0 + 370, y0 + 70, f"n={n_per_pair} shared field judgments per pair", 13, anchor="middle", fill=MUTED)
    for index in range(4):
        label = f"Physician {index + 1}"
        text(out, left + (index + 0.5) * cell, top - 18, label, 13, anchor="middle")
        text(out, left - 16, top + (index + 0.55) * cell, label, 13, anchor="end")
    for row in range(4):
        for column in range(4):
            x, y = left + column * cell, top + row * cell
            if row == column:
                out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="#F0F2F3" stroke="white" stroke-width="3"/>')
                text(out, x + cell / 2, y + cell / 2 + 6, "self", 14, anchor="middle", fill=MUTED)
            elif column > row:
                agreement, kappa, _ = values[(names[row], names[column])]
                fraction = (kappa + 0.10) / 0.80
                fill = interpolate_hex(PHYSICIAN_SOFT, PHYSICIAN, fraction)
                ink = "white" if fraction > 0.58 else TEXT
                out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="white" stroke-width="3"/>')
                text(out, x + cell / 2, y + 46, f"{agreement:.1%}", 18, anchor="middle", weight=700, fill=ink)
                text(out, x + cell / 2, y + 72, f"κ {kappa:.2f}", 15, anchor="middle", fill=ink)
            else:
                out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="#FBFBFB" stroke="white" stroke-width="3"/>')


def plot_interrater_matrix(raters: dict[str, list[dict[str, str]]]) -> None:
    width, height = 1500, 760
    out = svg_start(width, height)
    panel(out, 25, 20, 710, 665, "A", "Breast cancer")
    panel(out, 765, 20, 710, 665, "B", "PDAC")
    draw_agreement_matrix(out, raters, 15, 55, "Pairwise agreement", "b", BREAST_FIELDS, 280)
    draw_agreement_matrix(out, raters, 755, 55, "Pairwise agreement", "p", PDAC_FIELDS, 260)
    gradient_x, gradient_y, gradient_w = 520, 715, 460
    for index in range(100):
        out.append(f'<rect x="{gradient_x + gradient_w * index / 100:.1f}" y="{gradient_y}" width="{gradient_w / 100 + 1:.1f}" height="15" fill="{interpolate_hex(PHYSICIAN_SOFT, PHYSICIAN, index / 99)}"/>')
    text(out, gradient_x - 12, gradient_y + 13, "κ -0.10", 12, anchor="end", fill=MUTED)
    text(out, gradient_x + gradient_w + 12, gradient_y + 13, "κ 0.70", 12, fill=MUTED)
    text(out, 750, 755, "Color encodes Cohen's κ; each populated cell also reports exact agreement", 13, anchor="middle", fill=MUTED)
    svg_end(out, OUTPUT / "figure3_interrater_matrix_rough.svg")


def completed_rows(raters: dict[str, list[dict[str, str]]], breast_fields: list[str], pdac_fields: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rater_rows in raters.values():
        rows.extend(subset(rater_rows, "b", breast_fields))
        rows.extend(subset(rater_rows, "p", pdac_fields))
    return rows


def core_field_counts(raters: dict[str, list[dict[str, str]]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in completed_rows(raters, CORE_BREAST, CORE_PDAC):
        counts[row["field"]][row["score"]] += 1
    return counts


def draw_core_field_profile(
    out: list[str], raters: dict[str, list[dict[str, str]]]
) -> None:
    by_field = core_field_counts(raters)
    fields = CORE_FIELD_ORDER
    text(out, 65, 92, "Raw preference counts include ties", 14, fill=MUTED)
    legend(out, 300, 124, [(BL, "Baseline"), (TIE, "Tie"), (PL, "Harness")], gap=115)
    center, scale, top, row_gap, bar_h = 450, 1.75, 170, 91, 42
    for tick in [-100, -50, 0, 50, 100]:
        x = center + tick * scale
        stroke = "#7A858D" if tick == 0 else GRID
        out.append(f'<line x1="{x:.1f}" y1="145" x2="{x:.1f}" y2="795" stroke="{stroke}"/>')
        text(out, x, 824, f"{abs(tick)}%" if tick else "0", 12, anchor="middle", fill=MUTED)
    text(out, center - 112, 862, "Baseline preferred", 13, anchor="middle", fill=BL)
    text(out, center + 112, 862, "Harness preferred", 13, anchor="middle", fill=PL)
    text(out, 656, 150, "Harness / baseline / tie", 13, weight=700)
    for index, field in enumerate(fields):
        counts = by_field[field]
        total = sum(counts.values())
        y = top + index * row_gap
        text(out, 250, y + 27, FIELD_LABELS[field], 14, anchor="end")
        bl_pct, tie_pct, pl_pct = (counts[key] / total * 100 for key in ("B", "TIE", "A"))
        tie_half = tie_pct / 2
        bl_x = center - (bl_pct + tie_half) * scale
        out.append(f'<rect x="{bl_x:.1f}" y="{y}" width="{bl_pct * scale:.1f}" height="{bar_h}" fill="{BL}"/>')
        out.append(f'<rect x="{center - tie_half * scale:.1f}" y="{y}" width="{tie_pct * scale:.1f}" height="{bar_h}" fill="{TIE}"/>')
        out.append(f'<rect x="{center + tie_half * scale:.1f}" y="{y}" width="{pl_pct * scale:.1f}" height="{bar_h}" fill="{PL}"/>')
        if bl_pct * scale > 38:
            text(out, bl_x + bl_pct * scale / 2, y + 28, str(counts["B"]), 13, anchor="middle", weight=700, fill="white")
        if tie_pct * scale > 42:
            text(out, center, y + 28, str(counts["TIE"]), 13, anchor="middle", weight=700)
        if pl_pct * scale > 38:
            text(out, center + tie_half * scale + pl_pct * scale / 2, y + 28, str(counts["A"]), 13, anchor="middle", weight=700, fill="white")
        margin = (counts["A"] - counts["B"]) / total
        text(out, 656, y + 20, f"{counts['A']} / {counts['B']} / {counts['TIE']}", 13)
        text(out, 656, y + 41, f"n={total}; net {margin:+.1%}", 12, fill=MUTED)


def note_margins(rows: list[dict[str, str]], prefix: str, fields: list[str]) -> dict[str, tuple[int, int]]:
    net: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in subset(rows, prefix, fields):
        net[row["sample"]] += {"A": 1, "B": -1, "TIE": 0}[row["score"]]
        totals[row["sample"]] += 1
    return {sample: (net[sample], totals[sample]) for sample in net}


def draw_margin_panel(out: list[str], x0: int, y0: int, width: int, title: str, margins: dict[str, tuple[int, int]], summary: str) -> None:
    ordered = sorted(margins.items(), key=lambda item: item[1][0] / item[1][1])
    left, right, top, row_gap = x0 + 100, x0 + width - 35, y0 + 75, 31
    xmin, xmax = -0.2, 0.8
    text(out, x0 + 20, y0 + 27, title, 19, weight=700)
    text(out, x0 + width - 20, y0 + 27, summary, 14, anchor="end", weight=700, fill=PL)
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]:
        x = left + (tick - xmin) / (xmax - xmin) * (right - left)
        stroke = "#737E86" if tick == 0 else GRID
        out.append(f'<line x1="{x:.1f}" y1="{top - 22}" x2="{x:.1f}" y2="{top + 19 * row_gap + 16}" stroke="{stroke}"/>')
        text(out, x, top + 19 * row_gap + 40, f"{tick:.0%}", 11, anchor="middle", fill=MUTED)
    zero_x = left + (0 - xmin) / (xmax - xmin) * (right - left)
    for index, (label, (net, total)) in enumerate(ordered):
        value = net / total
        y = top + index * row_gap
        x = left + (value - xmin) / (xmax - xmin) * (right - left)
        text(out, left - 13, y + 5, label, 12, anchor="end")
        color = PL if value > 0 else BL if value < 0 else TIE
        out.append(f'<line x1="{zero_x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{y}" r="6.5" fill="{color}" stroke="white" stroke-width="1.5"/>')
    text(out, x0 + width / 2, top + 19 * row_gap + 67, "Normalized margin: (harness preferences - baseline preferences) / completed judgments", 12, anchor="middle", fill=MUTED)


def plot_supplementary_note_margins(raters: dict[str, list[dict[str, str]]]) -> None:
    all_rows = [row for rows in raters.values() for row in rows]
    breast = note_margins(all_rows, "b", BREAST_FIELDS)
    pdac = note_margins(all_rows, "p", PDAC_FIELDS)
    width, height = 1500, 1160
    out = svg_start(width, height)
    panel(out, 25, 20, 1450, 285, "A", "Compact note-level distribution")
    draw_compact_note_distribution(out, 75, 95, 1350, breast, pdac)
    panel(out, 25, 325, 710, 810, "B", "Breast cancer note detail")
    panel(out, 765, 325, 710, 810, "C", "PDAC note detail")
    draw_margin_panel(out, 35, 373, 680, "Notes ranked by preference margin", breast, "20/20 positive")
    draw_margin_panel(out, 775, 373, 680, "Notes ranked by preference margin", pdac, "19/20 positive; 1 negative")
    svg_end(out, OUTPUT / "supplementary_figure_s1_note_margins.svg")


def draw_compact_note_distribution(
    out: list[str],
    x0: int,
    y0: int,
    width: int,
    breast: dict[str, tuple[int, int]],
    pdac: dict[str, tuple[int, int]],
) -> None:
    left, right = x0 + 170, x0 + width - 35
    xmin, xmax = -0.2, 0.8
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]:
        x = left + (tick - xmin) / (xmax - xmin) * (right - left)
        stroke = "#59656D" if tick == 0 else GRID
        dash = "6 5" if tick == 0 else "0"
        out.append(
            f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0 + 135}" '
            f'stroke="{stroke}" stroke-dasharray="{dash}"/>'
        )
        text(out, x, y0 + 163, f"{tick:.0%}", 12, anchor="middle", fill=MUTED)
    for index, (label, margins, summary) in enumerate(
        [("Breast cancer", breast, "20/20 positive"), ("PDAC", pdac, "19/20 positive; 1 negative")]
    ):
        y = y0 + 35 + index * 70
        values = [net / total for net, total in margins.values()]
        frequencies = Counter(values)
        used: Counter[float] = Counter()
        text(out, left - 20, y + 5, label, 15, anchor="end", weight=700)
        text(out, right, y - 24, summary, 12, anchor="end", weight=700, fill=PL)
        out.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#AEB7BD"/>')
        for value in sorted(values):
            rank = used[value]
            count = frequencies[value]
            used[value] += 1
            jitter = (rank - (count - 1) / 2) * 8
            x = left + (value - xmin) / (xmax - xmin) * (right - left)
            color = PL if value > 0 else BL if value < 0 else TIE
            out.append(
                f'<circle cx="{x:.1f}" cy="{y + jitter:.1f}" r="5.5" fill="{color}" '
                'stroke="white" stroke-width="1"/>'
            )
    text(out, x0 + width / 2, y0 + 193, "Normalized note margin", 13, anchor="middle", fill=MUTED)


def draw_core_field_probability(out: list[str]) -> None:
    rows_by_field = {row["field"]: row for row in load_csv(CORE_FIELD_RESULTS)}
    text(out, 825, 92, "P(harness preference | directional judgment)", 14, fill=MUTED)
    probability_left, probability_width = 1080, 250
    probability_top, row_gap = 185, 92
    probability_min, probability_max = 0.30, 1.00
    for tick in [0.3, 0.5, 0.7, 0.9, 1.0]:
        x = probability_left + probability_width * (tick - probability_min) / (probability_max - probability_min)
        stroke = "#59656D" if tick == 0.5 else GRID
        dash = "6 5" if tick == 0.5 else "0"
        out.append(f'<line x1="{x:.1f}" y1="135" x2="{x:.1f}" y2="{probability_top + 6 * row_gap + 30}" stroke="{stroke}" stroke-dasharray="{dash}"/>')
        text(out, x, probability_top + 6 * row_gap + 60, f"{tick:.0%}", 12, anchor="middle", fill=MUTED)
    text(out, 1410, 135, "Adjusted estimate", 13, anchor="middle", weight=700)
    for index, field in enumerate(CORE_FIELD_ORDER):
        row = rows_by_field[field]
        y = probability_top + index * row_gap
        estimate = float(row["adjusted_probability"])
        low = float(row["probability_ci_low"])
        high = float(row["probability_ci_high"])
        xpos = lambda value: probability_left + probability_width * (value - probability_min) / (probability_max - probability_min)
        text(out, probability_left - 18, y + 6, FIELD_LABELS[field], 14, anchor="end")
        out.append(f'<line x1="{xpos(low):.1f}" y1="{y}" x2="{xpos(high):.1f}" y2="{y}" stroke="{PL}" stroke-width="4"/>')
        out.append(f'<line x1="{xpos(low):.1f}" y1="{y - 9}" x2="{xpos(low):.1f}" y2="{y + 9}" stroke="{PL}" stroke-width="2"/>')
        out.append(f'<line x1="{xpos(high):.1f}" y1="{y - 9}" x2="{xpos(high):.1f}" y2="{y + 9}" stroke="{PL}" stroke-width="2"/>')
        out.append(f'<circle cx="{xpos(estimate):.1f}" cy="{y}" r="8" fill="{PL}" stroke="white" stroke-width="2"/>')
        text(out, 1410, y + 6, f"{estimate:.1%} ({low:.1%}-{high:.1%})", 12, anchor="middle")
    text(out, xpos(0.5) + 5, 162, "Equal preference", 12, fill=MUTED)
    text(out, 1110, 860, "Joint exploratory GEE with shared physician and cancer effects", 13, anchor="middle", fill=MUTED)
    text(out, 1110, 886, "Ties excluded; preference is not accuracy", 13, anchor="middle", fill=MUTED)


def draw_note_margin_distribution(
    out: list[str], raters: dict[str, list[dict[str, str]]]
) -> None:
    all_rows = [row for rows in raters.values() for row in rows]
    groups = [
        ("Breast", note_margins(all_rows, "b", BREAST_FIELDS), "20/20 positive"),
        ("PDAC", note_margins(all_rows, "p", PDAC_FIELDS), "19/20 positive; 1 negative"),
    ]
    left, right, xmin, xmax = 1000, 1435, -0.2, 0.8
    top, bottom = 600, 855
    for tick in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]:
        x = left + (tick - xmin) / (xmax - xmin) * (right - left)
        stroke = "#59656D" if tick == 0 else GRID
        dash = "6 5" if tick == 0 else "0"
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="{stroke}" stroke-dasharray="{dash}"/>')
        text(out, x, 883, f"{tick:.0%}", 12, anchor="middle", fill=MUTED)
    for group_index, (label, margins, summary) in enumerate(groups):
        y = 665 + group_index * 125
        values = [net / total for net, total in margins.values()]
        frequencies = Counter(values)
        used: Counter[float] = Counter()
        text(out, left - 22, y + 5, label, 15, anchor="end", weight=700)
        text(out, right, y - 43, summary, 13, anchor="end", weight=700, fill=PL)
        out.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" stroke="#AEB7BD" stroke-width="2"/>')
        for value in sorted(values):
            rank = used[value]
            count = frequencies[value]
            used[value] += 1
            jitter = (rank - (count - 1) / 2) * 12
            x = left + (value - xmin) / (xmax - xmin) * (right - left)
            color = PL if value > 0 else BL if value < 0 else TIE
            out.append(
                f'<circle cx="{x:.1f}" cy="{y + jitter:.1f}" r="7" fill="{color}" '
                'stroke="white" stroke-width="1.5"/>'
            )
    text(out, 1218, 912, "Normalized note margin", 13, anchor="middle", fill=MUTED)


def plot_effect_profile(raters: dict[str, list[dict[str, str]]]) -> None:
    width, height = 1500, 940
    out = svg_start(width, height)
    panel(out, 20, 20, 740, 900, "A", "Core-field raw preference distribution")
    panel(out, 780, 20, 700, 900, "B", "Core-field adjusted preference probability")
    draw_core_field_profile(out, raters)
    draw_core_field_probability(out)
    svg_end(out, OUTPUT / "figure4_effect_profile.svg")


def load_ablation_counts() -> tuple[dict[str, dict[str, tuple[int, int, int]]], dict[str, float]]:
    markdown = ABLATION_SUMMARY.read_text(encoding="utf-8")
    comparisons = {
        "A vs B": "A→B",
        "B vs C": "B→C",
        "C vs D": "C→D",
        "A baseline vs B decomposition": "A→B",
        "B decomposition vs C verification": "B→C",
        "C verification vs D full harness": "C→D",
    }
    data: dict[str, dict[str, tuple[int, int, int]]] = defaultdict(dict)
    pvalues: dict[str, float] = {}
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0] in comparisons and cells[1] in {"Breast", "PDAC"}:
            data[comparisons[cells[0]]][cells[1]] = tuple(map(int, cells[2:5]))
        elif len(cells) == 6 and cells[0] in comparisons and cells[1].isdigit():
            stage = comparisons[cells[0]]
            data[stage]["Overall"] = tuple(map(int, cells[1:4]))
            pvalues[stage] = float(cells[5])
    if any(set(data[stage]) != {"Overall", "Breast", "PDAC"} for stage in ("A→B", "B→C", "C→D")):
        raise ValueError("Could not parse all staged-ablation counts")
    return data, pvalues


def draw_diverging_counts(out: list[str], x0: float, y: float, width: float, counts: tuple[int, int, int], *, total: int) -> None:
    earlier, later, ties = counts
    center = x0 + width / 2
    scale = width / 2 / total
    left_edge = center - (earlier + ties / 2) * scale
    out.append(f'<rect x="{left_edge:.1f}" y="{y}" width="{earlier * scale:.1f}" height="34" fill="{NEUTRAL}"/>')
    out.append(f'<rect x="{center - ties / 2 * scale:.1f}" y="{y}" width="{ties * scale:.1f}" height="34" fill="{TIE}"/>')
    out.append(f'<rect x="{center + ties / 2 * scale:.1f}" y="{y}" width="{later * scale:.1f}" height="34" fill="{PL}"/>')
    if earlier:
        text(out, left_edge + earlier * scale / 2, y + 23 if earlier * scale > 28 else y - 5, str(earlier), 12, anchor="middle", weight=700, fill="white" if earlier * scale > 28 else NEUTRAL)
    if ties:
        text(out, center, y + 23 if ties * scale > 25 else y - 5, str(ties), 12, anchor="middle", weight=700)
    if later:
        text(out, center + ties / 2 * scale + later * scale / 2, y + 23 if later * scale > 28 else y - 5, str(later), 12, anchor="middle", weight=700, fill="white" if later * scale > 28 else PL)


def plot_staged_ablation() -> None:
    data, pvalues = load_ablation_counts()
    width, height = 1500, 760
    out = svg_start(width, height)
    stages = [
        ["A", "Single prompt"],
        ["B", "Decomposition +", "orchestration"],
        ["C", "Verification", "gates"],
        ["D", "Deterministic", "rules"],
    ]
    stage_xs = [110, 460, 810, 1160]
    for index, stage in enumerate(stages):
        x = stage_xs[index]
        box(out, x, 35, 230, 76, stage, fill=HARNESS_SOFT if index else BASELINE_SOFT, stroke=PL if index else BL, size=14)
        if index:
            arrow(out, stage_xs[index - 1] + 230, 73, x, 73, color=PL)
    text(out, 750, 143, "Each panel tests whether one added layer improves the immediately preceding condition", 15, anchor="middle", fill=MUTED)
    legend(out, 500, 178, [(NEUTRAL, "Earlier condition"), (TIE, "Tie"), (PL, "Condition with added layer")], gap=190)
    stage_keys = ["A→B", "B→C", "C→D"]
    added_layers = {
        "A→B": "+ decomposition and orchestration",
        "B→C": "+ verification gates",
        "C→D": "+ deterministic rules",
    }
    panel_xs = [35, 520, 1005]
    for key, px in zip(stage_keys, panel_xs):
        round_rect(out, px, 205, 460, 455, fill=PANEL_FILL, stroke=PL if key == "B→C" else PANEL_STROKE, radius=12, stroke_width=2.2 if key == "B→C" else 1.5)
        text(out, px + 25, 242, key.replace("→", " → "), 20, weight=700)
        text(out, px + 25, 270, added_layers[key], 16, weight=700 if key == "B→C" else 400, fill=PL if key == "B→C" else MUTED)
        text(out, px + 435, 240, f"overall p={pvalues[key]:.6g}", 13, anchor="end", weight=700 if key == "B→C" else 400, fill=PL if key == "B→C" else MUTED)
        for row_index, scope in enumerate(("Overall", "Breast", "PDAC")):
            y = 315 + row_index * 95
            counts = data[key][scope]
            total = 40 if scope == "Overall" else 20
            text(out, px + 25, y + 23, scope, 14, weight=700 if scope == "Overall" else 400)
            draw_diverging_counts(out, px + 105, y, 320, counts, total=total)
            directional = counts[0] + counts[1]
            later_share = counts[1] / directional if directional else 0.5
            text(out, px + 265, y + 58, f"later-stage share among non-ties: {later_share:.1%}", 11, anchor="middle", fill=MUTED)
        if key == "B→C":
            round_rect(out, px + 85, 600, 290, 35, fill=HARNESS_SOFT, stroke=PL, radius=17)
            text(out, px + 230, 623, "Clear overall verification gain", 13, anchor="middle", weight=700, fill=PL)
    text(out, 750, 705, "One blinded non-physician Codex judge; secondary technical evidence", 14, anchor="middle", weight=700, fill=LLM)
    text(out, 750, 732, "Exact two-sided sign tests exclude ties; overall p values are exploratory and unadjusted", 12, anchor="middle", fill=MUTED)
    svg_end(out, OUTPUT / "supplementary_figure_s2_staged_ablation.svg")


def plot_human_model_complementarity(raters: dict[str, list[dict[str, str]]]) -> None:
    counts = core_field_counts(raters)
    margins = {
        field: (values["A"] - values["B"]) / sum(values.values())
        for field, values in counts.items()
    }
    largest = max(margins, key=margins.get)
    smallest = min(margins, key=margins.get)
    width, height = 1500, 720
    out = svg_start(width, height)
    text(out, 40, 44, "Results-informed clinical handoff", 22, weight=700)
    text(out, 40, 70, "A conceptual interpretation of how structured outputs can enter physician review", 13, fill=MUTED)
    box(out, 50, 290, 175, 105, ["Source", "oncology note"], fill="white", stroke="#65737E", size=17)
    box(out, 285, 275, 225, 135, ["Inference harness", "structured fields", "+ supporting excerpts"], fill=HARNESS_SOFT, stroke=PL, size=16)
    arrow(out, 225, 342, 285, 342, color=PL)
    lanes = [
        (125, "1", "Directly structure", ["Explicit receptor status", "Completed test result"], HARNESS_SOFT, PL),
        (295, "2", "Reconcile and surface evidence", ["Active medications and plans", "Long histories; stage and metastasis"], NEUTRAL_SOFT, NEUTRAL),
        (465, "3", "Escalate for clinical interpretation", ["Treatment response", "Uncertain metastatic disease"], PHYSICIAN_SOFT, PHYSICIAN),
    ]
    for y, number, heading, examples, fill, stroke in lanes:
        round_rect(out, 575, y, 700, 125, fill=fill, stroke=stroke, radius=12, stroke_width=2)
        out.append(f'<circle cx="615" cy="{y + 37}" r="20" fill="{stroke}"/>')
        text(out, 615, y + 44, number, 18, anchor="middle", weight=700, fill="white")
        text(out, 650, y + 40, heading, 18, weight=700)
        text(out, 650, y + 71, examples[0], 14)
        text(out, 650, y + 97, examples[1], 14)
        arrow(out, 510, 342, 575, y + 62, color=stroke)
    out.append(f'<path d="M1325,130 h28 v455 h-28" fill="none" stroke="{PHYSICIAN}" stroke-width="4"/>')
    text(out, 1385, 330, "Physician", 17, anchor="middle", weight=700, fill=PHYSICIAN)
    text(out, 1385, 354, "oversight", 17, anchor="middle", weight=700, fill=PHYSICIAN)
    text(out, 1385, 383, "across all", 13, anchor="middle", fill=MUTED)
    text(out, 1385, 402, "handoff lanes", 13, anchor="middle", fill=MUTED)
    round_rect(out, 150, 600, 515, 72, fill="#F7F9FA", stroke=PANEL_STROKE, radius=9)
    text(out, 175, 628, "Observed core-field range", 14, weight=700)
    text(out, 175, 653, f"Largest net margin: {FIELD_LABELS[largest]} ({margins[largest]:+.1%})", 13, fill=PL)
    round_rect(out, 695, 600, 515, 72, fill="#F7F9FA", stroke=PANEL_STROKE, radius=9)
    text(out, 720, 628, "Observed core-field range", 14, weight=700)
    text(out, 720, 653, f"Smallest net margin: {FIELD_LABELS[smallest]} ({margins[smallest]:+.1%})", 13, fill=MUTED)
    text(out, 750, 704, "Conceptual interpretation only; time, workload, and risk were not measured", 13, anchor="middle", fill=MUTED)
    svg_end(out, OUTPUT / "figure6_clinical_handoff.svg")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raters = load_raters()
    required = completed_rows(raters, BREAST_FIELDS, PDAC_FIELDS)
    required_counts = Counter(row["score"] for row in required)
    expected = Counter({"A": 699, "B": 171, "TIE": 1290})
    if required_counts != expected:
        raise ValueError(f"Physician-rating totals changed: {required_counts} != {expected}")
    if sum(sum(values.values()) for values in core_field_counts(raters).values()) != 1040:
        raise ValueError("Core-field judgment total must remain 1,040")
    expected_core = {
        "current_meds": Counter({"A": 132, "B": 3, "TIE": 25}),
        "stage": Counter({"A": 69, "B": 7, "TIE": 84}),
        "distant_met": Counter({"A": 23, "B": 5, "TIE": 132}),
        "metastasis": Counter({"A": 74, "B": 26, "TIE": 60}),
        "response": Counter({"A": 62, "B": 12, "TIE": 86}),
        "type_receptor": Counter({"A": 24, "B": 14, "TIE": 42}),
        "genetic_results": Counter({"A": 34, "B": 10, "TIE": 116}),
    }
    if core_field_counts(raters) != expected_core:
        raise ValueError("Core-field counts changed")
    all_rows = [row for rows in raters.values() for row in rows]
    breast_margins = note_margins(all_rows, "b", BREAST_FIELDS)
    pdac_margins = note_margins(all_rows, "p", PDAC_FIELDS)
    if (len(breast_margins), len(pdac_margins)) != (20, 20):
        raise ValueError("Expected 20 notes in each cancer cohort")
    if sum(net > 0 for net, _ in breast_margins.values()) != 20:
        raise ValueError("Breast note-level margins changed")
    if (
        sum(net > 0 for net, _ in pdac_margins.values()),
        sum(net < 0 for net, _ in pdac_margins.values()),
    ) != (19, 1):
        raise ValueError("PDAC note-level margins changed")
    adjusted = load_csv(ADJUSTED_RESULTS)
    adjusted_expected = {
        "Overall": (0.7998, 0.7575, 0.8363),
        "Breast cancer": (0.8166, 0.7602, 0.8621),
        "PDAC": (0.7815, 0.7147, 0.8362),
    }
    for row in adjusted:
        observed = tuple(
            round(float(row[key]), 4)
            for key in ("adjusted_probability", "probability_ci_low", "probability_ci_high")
        )
        if observed != adjusted_expected[row["scope"]]:
            raise ValueError(f"Adjusted probability changed for {row['scope']}: {observed}")
    plot_development_pathway()
    plot_evaluator_distribution(raters)
    plot_interrater_matrix(raters)
    plot_effect_profile(raters)
    plot_staged_ablation()
    plot_supplementary_note_margins(raters)
    print(f"Wrote 4 main SVG figures and 2 supplementary SVG figures to {OUTPUT}")


if __name__ == "__main__":
    main()
