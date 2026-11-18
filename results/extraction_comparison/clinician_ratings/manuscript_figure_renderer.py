#!/usr/bin/env python3
"""Render the six manuscript figures with one publication style.

The physician author's November 2026 figures define the panel structures. All
counts and estimates come from the existing project analysis files. Both SVG
and PNG outputs are generated directly from the same Matplotlib figures.
"""

from __future__ import annotations

import csv
import math
import re
import zipfile
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch


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

# Shared semantic visual language. Ablation stages B and C use slate shades so
# physician green and LLM-review purple retain one meaning throughout.
HARNESS = "#0072B2"
BASELINE = "#D55E00"
TIE = "#B8B8B8"
PHYSICIAN = "#009E73"
LLM_REVIEW = "#7A5FA8"
TEXT = "#243342"
MUTED = "#5F6D78"
GRID = "#E6E6E6"
PANEL_FILL = "#FAFBFC"
PANEL_EDGE = "#CFD8DE"
STAGE_B = "#718096"
STAGE_C = "#4A5568"
WHITE = "#FFFFFF"

FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
PANEL_TITLE_SIZE = 11.5
DATA_SIZE = 9.5
LEGEND_SIZE = 9
TICK_SIZE = 8
FOOTNOTE_SIZE = 8
AXIS_WIDTH = 2.0
DATA_WIDTH = 2.2
REFERENCE_WIDTH = 1.0
FIGURE_WIDTH = 7.5

BREAST_FIELDS = [
    "current_meds", "stage", "distant_met", "metastasis", "response",
    "type_receptor", "genetic_results", "genetic_plan", "supportive_meds",
    "procedure_plan", "imaging_plan", "lab_plan", "medication_plan", "recent_changes",
]
PDAC_FIELDS = [field for field in BREAST_FIELDS if field != "type_receptor"]
CORE_BREAST = BREAST_FIELDS[:7]
CORE_PDAC = ["current_meds", "stage", "distant_met", "metastasis", "response", "genetic_results"]
CORE_FIELD_ORDER = [
    "current_meds", "stage", "response", "distant_met", "genetic_results",
    "metastasis", "type_receptor",
]
FIELD_LABELS = {
    "current_meds": "Active medications",
    "stage": "Stage",
    "response": "Treatment response",
    "distant_met": "Distant metastasis",
    "genetic_results": "Molecular or genetic results",
    "metastasis": "Regional or overall metastasis",
    "type_receptor": "Breast type and receptors",
}

FIGURES = {
    "figure1_development_pathway_rough": (6.0, "Development, adaptation, and evaluation workflow"),
    "figure2_evaluator_distribution_rough": (5.5, "Physician preference overview"),
    "figure3_interrater_matrix_rough": (3.8, "Agreement among physicians"),
    "figure4_effect_profile": (3.0, "Physician preference by core oncology field"),
    "supplementary_figure_s1_note_margins": (1.8, "Note level preference margins"),
    "supplementary_figure_s2_staged_ablation": (3.4, "Stepwise ablation of the inference harness"),
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY[:-1],
        "font.size": DATA_SIZE,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": TEXT,
        "axes.linewidth": AXIS_WIDTH,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.titleweight": "bold",
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": TICK_SIZE,
        "ytick.labelsize": TICK_SIZE,
        "legend.fontsize": LEGEND_SIZE,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "svg.hashsalt": "med-dict-manuscript-v2.1.0",
        "savefig.facecolor": WHITE,
        "figure.facecolor": WHITE,
    })


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{key: (value or "").strip() for key, value in row.items()} for row in csv.DictReader(handle)]


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference).group(0)
    value = 0
    for letter in letters:
        value = value * 26 + ord(letter) - 64
    return value - 1


def load_xlsx(path: Path) -> list[dict[str, str]]:
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
        relationship_id = next(
            sheet.attrib[f"{{{doc_rel_ns}}}id"]
            for sheet in workbook.find(f"{{{main_ns}}}sheets")
            if sheet.attrib.get("name") == "blind_scores"
        )
        rels = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = next(relation.attrib["Target"] for relation in rels.findall(f"{{{rel_ns}}}Relationship") if relation.attrib["Id"] == relationship_id)
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
                value_node = cell.find(f"{{{main_ns}}}v")
                if cell.attrib.get("t") == "inlineStr":
                    inline = cell.find(f"{{{main_ns}}}is")
                    value = "" if inline is None else "".join(node.text or "" for node in inline.iter(f"{{{main_ns}}}t"))
                elif value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared[int(value_node.text)]
                else:
                    value = value_node.text or ""
                values[index] = value.strip()
            matrix.append(values)
    headers = matrix[0]
    return [
        {key: row[index].strip() if index < len(row) else "" for index, key in enumerate(headers)}
        for row in matrix[1:] if row and row[0]
    ]


def load_raters() -> dict[str, list[dict[str, str]]]:
    raters: dict[str, list[dict[str, str]]] = {}
    for name, paths in RATER_FILES.items():
        rows: list[dict[str, str]] = []
        for path in paths:
            rows.extend(load_xlsx(path) if path.suffix == ".xlsx" else load_csv(path))
        keys = [(row["sample"], row["field"]) for row in rows]
        assert len(keys) == len(set(keys)), f"Duplicate sample and field judgments for {name}"
        raters[name] = rows
    return raters


def subset(rows: list[dict[str, str]], prefix: str, fields: list[str]) -> list[dict[str, str]]:
    allowed = set(fields)
    return [row for row in rows if row["sample"].startswith(prefix) and row["field"] in allowed]


def completed_rows(raters: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rater_rows in raters.values():
        rows.extend(subset(rater_rows, "b", BREAST_FIELDS))
        rows.extend(subset(rater_rows, "p", PDAC_FIELDS))
    return rows


def core_field_counts(raters: dict[str, list[dict[str, str]]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for rater_rows in raters.values():
        for row in subset(rater_rows, "b", CORE_BREAST) + subset(rater_rows, "p", CORE_PDAC):
            counts[row["field"]][row["score"]] += 1
    return counts


def note_margins(rows: list[dict[str, str]], prefix: str, fields: list[str]) -> dict[str, tuple[int, int]]:
    net: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for row in subset(rows, prefix, fields):
        net[row["sample"]] += {"A": 1, "B": -1, "TIE": 0}[row["score"]]
        totals[row["sample"]] += 1
    return {sample: (net[sample], totals[sample]) for sample in net}


def pairwise_agreement(r1: list[dict[str, str]], r2: list[dict[str, str]], prefix: str, fields: list[str]) -> tuple[float, float, int]:
    m1 = {(row["sample"], row["field"]): row["score"] for row in subset(r1, prefix, fields)}
    m2 = {(row["sample"], row["field"]): row["score"] for row in subset(r2, prefix, fields)}
    keys = sorted(set(m1) & set(m2))
    total = len(keys)
    observed = sum(m1[key] == m2[key] for key in keys) / total
    counts_1 = Counter(m1[key] for key in keys)
    counts_2 = Counter(m2[key] for key in keys)
    expected = sum(counts_1[value] * counts_2[value] for value in ["A", "TIE", "B"]) / total**2
    return observed, (observed - expected) / (1 - expected), total


def load_ablation_counts() -> tuple[dict[str, tuple[int, int, int]], dict[str, float]]:
    markdown = ABLATION_SUMMARY.read_text(encoding="utf-8")
    labels = {
        "A vs B": "A to B", "B vs C": "B to C", "C vs D": "C to D",
        "A baseline vs B decomposition": "A to B",
        "B decomposition vs C verification": "B to C",
        "C verification vs D full harness": "C to D",
    }
    counts: dict[str, tuple[int, int, int]] = {}
    pvalues: dict[str, float] = {}
    for line in markdown.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and cells[0] in labels and cells[1].isdigit():
            key = labels[cells[0]]
            counts[key] = tuple(map(int, cells[1:4]))
            pvalues[key] = float(cells[5])
    return counts, pvalues


def assert_data(raters: dict[str, list[dict[str, str]]]) -> None:
    assert Counter(row["score"] for row in completed_rows(raters)) == Counter({"A": 699, "TIE": 1290, "B": 171})
    expected_core = {
        "current_meds": Counter({"A": 132, "TIE": 25, "B": 3}),
        "stage": Counter({"A": 69, "TIE": 84, "B": 7}),
        "response": Counter({"A": 62, "TIE": 86, "B": 12}),
        "distant_met": Counter({"A": 23, "TIE": 132, "B": 5}),
        "genetic_results": Counter({"A": 34, "TIE": 116, "B": 10}),
        "metastasis": Counter({"A": 74, "TIE": 60, "B": 26}),
        "type_receptor": Counter({"A": 24, "TIE": 42, "B": 14}),
    }
    assert core_field_counts(raters) == expected_core
    all_rows = [row for rows in raters.values() for row in rows]
    breast = note_margins(all_rows, "b", BREAST_FIELDS)
    pdac = note_margins(all_rows, "p", PDAC_FIELDS)
    assert (len(breast), len(pdac)) == (20, 20)
    assert (sum(net > 0 for net, _ in breast.values()), sum(net == 0 for net, _ in breast.values())) == (20, 0)
    assert (sum(net > 0 for net, _ in pdac.values()), sum(net < 0 for net, _ in pdac.values()), sum(net == 0 for net, _ in pdac.values())) == (19, 1, 0)
    expected_probabilities = {
        "Overall": (0.7998, 0.7575, 0.8363),
        "Breast cancer": (0.8166, 0.7602, 0.8621),
        "PDAC": (0.7815, 0.7147, 0.8362),
    }
    for row in load_csv(ADJUSTED_RESULTS):
        observed = tuple(round(float(row[key]), 4) for key in ("adjusted_probability", "probability_ci_low", "probability_ci_high"))
        assert observed == expected_probabilities[row["scope"]]
    expected_core_probability = {
        "current_meds": (0.977, 0.911, 0.994),
        "stage": (0.894, 0.776, 0.954),
        "response": (0.817, 0.701, 0.894),
        "distant_met": (0.785, 0.550, 0.916),
        "genetic_results": (0.775, 0.611, 0.883),
        "metastasis": (0.744, 0.659, 0.814),
        "type_receptor": (0.624, 0.386, 0.815),
    }
    for row in load_csv(CORE_FIELD_RESULTS):
        observed = tuple(round(float(row[key]), 3) for key in ("adjusted_probability", "probability_ci_low", "probability_ci_high"))
        assert observed == expected_core_probability[row["field"]]
    counts, pvalues = load_ablation_counts()
    assert counts == {"A to B": (17, 17, 6), "B to C": (2, 24, 14), "C to D": (19, 16, 5)}
    assert pvalues == {"A to B": 1.0, "B to C": 0.0000105, "C to D": 0.736}


def clean_axis(ax, *, grid_axis: str | None = None) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(AXIS_WIDTH)
    ax.tick_params(width=AXIS_WIDTH, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)


def panel_title(ax, letter: str, title: str) -> None:
    ax.text(0, 1.04, letter, transform=ax.transAxes, fontsize=PANEL_TITLE_SIZE, fontweight="bold", va="bottom", ha="left", color=TEXT)
    ax.text(0.10, 1.04, title, transform=ax.transAxes, fontsize=PANEL_TITLE_SIZE, fontweight="bold", va="bottom", ha="left", color=TEXT)


def add_box(ax, xy, width, height, text, *, edge=PANEL_EDGE, face=WHITE, text_color=TEXT, weight="normal", size=8.4, lw=1.4, radius=0.02) -> FancyBboxPatch:
    patch = FancyBboxPatch(xy, width, height, boxstyle=f"round,pad=0.008,rounding_size={radius}", transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=lw, clip_on=False)
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, transform=ax.transAxes, ha="center", va="center", color=text_color, fontsize=size, fontweight=weight, linespacing=1.15)
    return patch


def add_arrow(ax, start, end, *, color=MUTED, connectionstyle="arc3", lw=1.6, style="-|>") -> None:
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes, arrowstyle=style, mutation_scale=10, linewidth=lw, color=color, connectionstyle=connectionstyle, clip_on=False))


def finalize_svg(path: Path, width: float, height: float, title: str, description: str) -> None:
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    tree = ET.parse(path)
    root = tree.getroot()
    namespace = root.tag.split("}")[0].strip("{")
    root.set("width", f"{width:g}in")
    root.set("height", f"{height:g}in")
    root.set("role", "img")
    root.set("aria-labelledby", "figure-title figure-description")
    for child in list(root):
        if child.tag in {f"{{{namespace}}}title", f"{{{namespace}}}desc"}:
            root.remove(child)
    title_node = ET.Element(f"{{{namespace}}}title", {"id": "figure-title"})
    title_node.text = title
    desc_node = ET.Element(f"{{{namespace}}}desc", {"id": "figure-description"})
    desc_node.text = description
    root.insert(0, desc_node)
    root.insert(0, title_node)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def save_figure(fig, stem: str, description: str) -> None:
    height, title = FIGURES[stem]
    svg = OUTPUT / f"{stem}.svg"
    png_stem = stem.replace("_rough", "") + "_print.png"
    png = OUTPUT / png_stem
    fig.set_size_inches(FIGURE_WIDTH, height, forward=True)
    fig.savefig(svg, format="svg", dpi=300, metadata={"Title": title, "Description": description, "Date": None})
    fig.savefig(png, format="png", dpi=300, metadata={"Title": title, "Description": description})
    finalize_svg(svg, FIGURE_WIDTH, height, title, description)
    plt.close(fig)


def plot_figure_1() -> None:
    fig = plt.figure(figsize=(FIGURE_WIDTH, 6.0))
    fig.subplots_adjust(0, 0, 1, 1)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    panels = {
        "A": (0.035, 0.825, 0.93, 0.145, "Study datasets"),
        "B": (0.035, 0.505, 0.45, 0.285, "Physician informed breast\ncancer development"),
        "C": (0.515, 0.505, 0.45, 0.285, "LLM assisted PDAC\nadaptation"),
        "D": (0.035, 0.205, 0.93, 0.26, "Independent physician holdout evaluation"),
        "E": (0.035, 0.035, 0.93, 0.13, "Staged technical ablation"),
    }
    for letter, (x, y, w, h, title) in panels.items():
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.015", transform=ax.transAxes, facecolor=PANEL_FILL, edgecolor=PANEL_EDGE, linewidth=1.2))
        ax.text(x + 0.014, y + h - 0.026, letter, transform=ax.transAxes, fontsize=PANEL_TITLE_SIZE, fontweight="bold", va="top")
        ax.text(x + 0.043, y + h - 0.026, title, transform=ax.transAxes, fontsize=10.5 if "\n" in title else PANEL_TITLE_SIZE, fontweight="bold", va="top", linespacing=1.05)

    # A. Three nonoverlapping datasets.
    a_cards = [
        (0.075, "Breast development\n100 notes", "Harness development"),
        (0.375, "PDAC adaptation\n100 notes", "Cross cancer adaptation"),
        (0.675, "Locked holdout\n20 breast + 20 PDAC", "No development use"),
    ]
    for x, heading, footer in a_cards:
        add_box(ax, (x, 0.848), 0.25, 0.063, heading, edge=MUTED, face=WHITE, weight="bold", size=8.4)
        ax.text(x + 0.125, 0.836, footer, transform=ax.transAxes, ha="center", va="top", fontsize=8, color=MUTED)

    # B. Three physician development steps and a visible feedback loop.
    b_x = [0.055, 0.205, 0.355]
    b_text = ["Generate\nstructured outputs", "Physician review\nof selected outputs", "Error guided\nharness refinement"]
    b_edges = [HARNESS, PHYSICIAN, HARNESS]
    b_faces = [WHITE, "#E9F6F2", "#E8F2F8"]
    for i, x in enumerate(b_x):
        add_box(ax, (x, 0.59), 0.11, 0.08, b_text[i], edge=b_edges[i], face=b_faces[i], weight="bold" if i == 2 else "normal", size=8.0)
        if i:
            add_arrow(ax, (x - 0.035, 0.63), (x - 0.004, 0.63), color=b_edges[i])
    add_arrow(ax, (0.41, 0.585), (0.11, 0.58), color=PHYSICIAN, connectionstyle="arc3,rad=-0.34", lw=1.7)
    ax.text(0.26, 0.552, "Feedback loop", transform=ax.transAxes, fontsize=8, color=PHYSICIAN, ha="center", fontweight="bold")
    ax.text(0.26, 0.512, "Model weights remain frozen", transform=ax.transAxes, fontsize=8, color=MUTED, ha="center", bbox={"facecolor": PANEL_FILL, "edgecolor": "none", "pad": 0.6})

    # C. Three adaptation steps and the required scope statement.
    c_x = [0.535, 0.685, 0.835]
    c_text = ["Transfer breast\nderived harness", "LLM assisted\nerror review", "PDAC specific\nharness refinement"]
    c_edges = [HARNESS, LLM_REVIEW, HARNESS]
    c_faces = ["#E8F2F8", "#F0ECF5", "#E8F2F8"]
    for i, x in enumerate(c_x):
        add_box(ax, (x, 0.59), 0.11, 0.08, c_text[i], edge=c_edges[i], face=c_faces[i], weight="bold" if i == 2 else "normal", size=8.0)
        if i:
            add_arrow(ax, (x - 0.035, 0.63), (x - 0.004, 0.63), color=c_edges[i])
    add_box(ax, (0.56, 0.52), 0.36, 0.04, "No target-domain physician review of PDAC outputs", edge=PHYSICIAN, face=WHITE, text_color=PHYSICIAN, weight="bold", size=8.0)

    # D. The main clinical endpoint.
    add_box(ax, (0.055, 0.27), 0.115, 0.085, "40 locked notes\n20 breast + 20 PDAC", edge=MUTED, face=WHITE, weight="bold", size=8.0)
    add_arrow(ax, (0.171, 0.312), (0.205, 0.312))
    add_box(ax, (0.205, 0.26), 0.13, 0.105, "Same frozen\nQwen2.5-32B\nSame schema", edge=MUTED, face=WHITE, size=8.0)
    add_arrow(ax, (0.336, 0.33), (0.37, 0.35), color=HARNESS)
    add_arrow(ax, (0.336, 0.29), (0.37, 0.272), color=BASELINE)
    add_box(ax, (0.37, 0.32), 0.16, 0.065, "Harness with\nsupporting excerpts", edge=HARNESS, face="#E8F2F8", text_color=HARNESS, weight="bold", size=8.0)
    add_box(ax, (0.37, 0.245), 0.16, 0.052, "Single prompt baseline", edge=BASELINE, face="#FBEDE7", text_color=BASELINE, weight="bold", size=8.0)
    add_arrow(ax, (0.532, 0.35), (0.57, 0.33), color=HARNESS)
    add_arrow(ax, (0.532, 0.272), (0.57, 0.30), color=BASELINE)
    add_box(ax, (0.57, 0.27), 0.135, 0.085, "Identity masked\ncomparison", edge=MUTED, face=WHITE, weight="bold", size=8.0)
    add_arrow(ax, (0.707, 0.312), (0.74, 0.312), color=PHYSICIAN)
    add_box(ax, (0.74, 0.27), 0.1, 0.085, "4 physicians\n2,160 judgments", edge=PHYSICIAN, face="#E9F6F2", text_color=PHYSICIAN, weight="bold", size=8.0)
    add_arrow(ax, (0.842, 0.312), (0.875, 0.312), color=PHYSICIAN)
    add_box(ax, (0.875, 0.25), 0.07, 0.125, "Logistic GEE\nNon-ties\nClustered\nby note", edge=MUTED, face=WHITE, size=8.0)
    ax.text(0.5, 0.225, "Independent physician evaluation of the locked holdout set", transform=ax.transAxes, fontsize=8, ha="center", color=MUTED)

    # E. A separate automated development analysis. No D to E arrow is drawn.
    labels = [("A", "Single prompt", BASELINE), ("B", "Decomposed", STAGE_B), ("C", "Verified", STAGE_C), ("D", "Full harness", HARNESS)]
    for i, (letter, name, color) in enumerate(labels):
        x = 0.075 + i * 0.145
        add_box(ax, (x, 0.052), 0.115, 0.04, f"{letter}  {name}", edge=color, face=WHITE, text_color=color, weight="bold", size=8.0)
        if i:
            add_arrow(ax, (x - 0.029, 0.072), (x - 0.003, 0.072), color=color)
    ax.text(0.69, 0.082, "40 locked notes", transform=ax.transAxes, fontsize=8.2, fontweight="bold", color=MUTED)
    ax.text(0.69, 0.055, "Blinded nonphysician LLM judge", transform=ax.transAxes, fontsize=8.2, color=LLM_REVIEW)

    save_figure(fig, "figure1_development_pathway_rough", "Five panels show the study datasets, physician informed breast cancer development, LLM assisted PDAC adaptation without target domain physician review, independent physician holdout evaluation, and a separate cumulative technical ablation.")


def plot_figure_2() -> None:
    fig = plt.figure(figsize=(FIGURE_WIDTH, 5.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.75], hspace=0.55, left=0.14, right=0.965, top=0.94, bottom=0.09)
    ax_a = fig.add_subplot(grid[0])
    ax_b = fig.add_subplot(grid[1])
    panel_title(ax_a, "A", "Full preference distribution")
    panel_title(ax_b, "B", "Harness share among directional judgments")

    rows = [
        ("Overall", 699, 1290, 171, 2160),
        ("Breast cancer", 363, 677, 80, 1120),
        ("PDAC", 336, 613, 91, 1040),
    ]
    for y, (label, harness, tie, baseline, total) in enumerate(rows):
        left = 0
        for value, color, hatch, ink in [(harness, HARNESS, "", WHITE), (tie, TIE, "..", TEXT), (baseline, BASELINE, "//", WHITE)]:
            pct = value / total * 100
            ax_a.barh(y, pct, left=left, color=color, edgecolor=WHITE, linewidth=0.8, height=0.62, hatch=hatch, zorder=2)
            if pct >= 7:
                ax_a.text(left + pct / 2, y, f"{value}\n{pct:.1f}%", ha="center", va="center", fontsize=8, color=ink, fontweight="bold" if color != TIE else "normal")
            left += pct
        ax_a.text(101, y, f"n={total:,}", va="center", fontsize=8, color=MUTED)
    ax_a.set_yticks(range(3), [row[0] for row in rows])
    ax_a.invert_yaxis()
    ax_a.set_xlim(0, 109)
    ax_a.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    clean_axis(ax_a, grid_axis="x")
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(handles=[Patch(facecolor=HARNESS, label="Harness"), Patch(facecolor=TIE, hatch="..", label="Tie"), Patch(facecolor=BASELINE, hatch="//", label="Baseline")], loc="upper right", bbox_to_anchor=(1.0, 1.24), ncol=3, frameon=False)

    directional = [
        ("Breast cancer, pooled", 363, 443, 81.9, True),
        ("   Physician 1", 84, 106, 79.2, False),
        ("   Physician 2", 79, 87, 90.8, False),
        ("   Physician 3", 119, 143, 83.2, False),
        ("   Physician 4", 81, 107, 75.7, False),
        ("PDAC, pooled", 336, 427, 78.7, True),
        ("   Physician 1", 75, 89, 84.3, False),
        ("   Physician 2", 86, 95, 90.5, False),
        ("   Physician 3", 103, 133, 77.4, False),
        ("   Physician 4", 72, 110, 65.5, False),
    ]
    y_values = np.arange(len(directional))
    for y, (label, numerator, denominator, pct, pooled) in zip(y_values, directional):
        ax_b.hlines(y, 50, pct, color=HARNESS if pooled else "#7AAFD0", linewidth=DATA_WIDTH if pooled else 1.5, zorder=2)
        marker = "D" if pooled else "o"
        ax_b.plot(pct, y, marker=marker, markersize=7 if pooled else 5.5, color=HARNESS, markeredgecolor=WHITE, markeredgewidth=0.8, zorder=3)
        ax_b.text(100.7, y, f"{numerator}/{denominator}  {pct:.1f}%", va="center", fontsize=8, fontweight="bold" if pooled else "normal", color=TEXT)
    ax_b.axvline(50, color=MUTED, linewidth=REFERENCE_WIDTH, linestyle="--", zorder=1)
    ax_b.set_xlim(48, 112)
    ax_b.set_xticks([50, 60, 70, 80, 90, 100], ["50%", "60%", "70%", "80%", "90%", "100%"])
    labels = ax_b.set_yticks(y_values, [row[0] for row in directional])
    ax_b.invert_yaxis()
    for tick, row in zip(ax_b.get_yticklabels(), directional):
        tick.set_fontweight("bold" if row[4] else "normal")
    clean_axis(ax_b, grid_axis="x")
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    ax_b.set_xlabel("Unadjusted harness share among directional judgments", fontsize=9)
    ax_b.text(50.3, -0.65, "Equal preference", fontsize=8, color=MUTED, ha="left")
    save_figure(fig, "figure2_evaluator_distribution_rough", "Panel A shows harness, tie, and baseline judgments overall and by cancer. Panel B shows the unadjusted directional harness share for pooled and individual physician results in each cancer.")


def plot_figure_3(raters: dict[str, list[dict[str, str]]]) -> None:
    names = list(raters)
    expected: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    for prefix, fields, cancer in [("b", BREAST_FIELDS, "Breast cancer"), ("p", PDAC_FIELDS, "PDAC")]:
        for a, b in combinations(names, 2):
            expected[(cancer, a, b)] = pairwise_agreement(raters[a], raters[b], prefix, fields)

    fig, axes = plt.subplots(1, 2, figsize=(FIGURE_WIDTH, 3.8), gridspec_kw={"wspace": 0.28, "left": 0.08, "right": 0.94, "top": 0.84, "bottom": 0.22})
    cmap = LinearSegmentedColormap.from_list("kappa_green", ["#EAF6F2", PHYSICIAN])
    norm = Normalize(vmin=-0.10, vmax=0.70)
    for ax, (prefix, fields, cancer, letter, n_per_pair) in zip(axes, [("b", BREAST_FIELDS, "Breast cancer", "A", 280), ("p", PDAC_FIELDS, "PDAC", "B", 260)]):
        matrix = np.full((4, 4), np.nan)
        values = {}
        for i, j in combinations(range(4), 2):
            agreement, kappa, count = pairwise_agreement(raters[names[i]], raters[names[j]], prefix, fields)
            assert count == n_per_pair
            matrix[i, j] = kappa
            values[(i, j)] = (agreement, kappa)
        ax.imshow(matrix, cmap=cmap, norm=norm, aspect="equal")
        for i in range(4):
            for j in range(4):
                if i == j:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#F0F2F3", edgecolor=WHITE, linewidth=2))
                    ax.text(j, i, "self", ha="center", va="center", fontsize=8, color=MUTED)
                elif j > i:
                    agreement, kappa = values[(i, j)]
                    ink = WHITE if kappa > 0.40 else TEXT
                    ax.text(j, i - 0.08, f"{agreement:.1%}", ha="center", va="center", fontsize=9, fontweight="bold", color=ink)
                    ax.text(j, i + 0.20, f"κ {kappa:.2f}", ha="center", va="center", fontsize=8, color=ink)
                else:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#FBFCFC", edgecolor=WHITE, linewidth=2))
        ax.set_xticks(range(4), [f"Physician {i}" for i in range(1, 5)], rotation=25, ha="right")
        ax.set_yticks(range(4), [f"Physician {i}" for i in range(1, 5)])
        ax.tick_params(length=0)
        ax.set_title(f"{letter}   {cancer}\nn = {n_per_pair} shared judgments per pair", loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Recompute after plotting to ensure every displayed agreement and kappa is unchanged.
    observed = {}
    for prefix, fields, cancer in [("b", BREAST_FIELDS, "Breast cancer"), ("p", PDAC_FIELDS, "PDAC")]:
        for a, b in combinations(names, 2):
            observed[(cancer, a, b)] = pairwise_agreement(raters[a], raters[b], prefix, fields)
    assert observed == expected
    color_ax = fig.add_axes([0.34, 0.095, 0.32, 0.025])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal")
    cb.set_ticks([-0.10, 0.10, 0.30, 0.50, 0.70])
    cb.set_label("Cohen's κ", fontsize=8)
    cb.outline.set_visible(False)
    save_figure(fig, "figure3_interrater_matrix_rough", "Two upper triangular matrices report exact pairwise agreement and Cohen's kappa for four physicians in breast cancer and PDAC. Color redundantly encodes kappa.")


def plot_figure_4(raters: dict[str, list[dict[str, str]]]) -> None:
    counts = core_field_counts(raters)
    rows = [("Overall", 418, 545, 77, 1040)] + [
        (FIELD_LABELS[field], counts[field]["A"], counts[field]["TIE"], counts[field]["B"], sum(counts[field].values()))
        for field in CORE_FIELD_ORDER
    ]
    expected_rows = [
        ("Overall", 418, 545, 77, 1040),
        ("Active medications", 132, 25, 3, 160),
        ("Stage", 69, 84, 7, 160),
        ("Treatment response", 62, 86, 12, 160),
        ("Distant metastasis", 23, 132, 5, 160),
        ("Molecular or genetic results", 34, 116, 10, 160),
        ("Regional or overall metastasis", 74, 60, 26, 160),
        ("Breast type and receptors", 24, 42, 14, 80),
    ]
    assert rows == expected_rows

    compact_labels = {
        "current_meds": "Active meds",
        "stage": "Stage",
        "response": "Response",
        "distant_met": "Distant metastasis",
        "genetic_results": "Molecular/genetic",
        "metastasis": "Regional/overall",
        "type_receptor": "Type or receptors",
    }
    fig = plt.figure(figsize=(FIGURE_WIDTH, 3.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 0.92, 1.28], left=0.025, right=0.985, top=0.80, bottom=0.17, wspace=0.31)
    ax_a, ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_title(ax_a, "A", "Raw counts")
    panel_title(ax_b, "B", "Full distributions")
    panel_title(ax_c, "C", "Adjusted probabilities")

    ax_a.axis("off")
    table_labels = ["Overall"] + [f"{index}  {compact_labels[field]}" for index, field in enumerate(CORE_FIELD_ORDER, 1)]
    table_data = [[label, row[1], row[2], row[3], row[4]] for label, row in zip(table_labels, rows)]
    table = ax_a.table(cellText=table_data, colLabels=["Field", "H", "T", "B", "n"], cellLoc="right", colLoc="right", loc="upper left", bbox=[0, 0, 1, 0.95], colWidths=[0.53, 0.12, 0.12, 0.12, 0.13])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.7)
        cell.set_facecolor(PANEL_FILL if r == 0 else WHITE)
        if r == 0 or r == 1:
            cell.get_text().set_fontweight("bold")
        if c == 0:
            cell.get_text().set_ha("left")
    ax_a.text(0, -0.09, "H, harness; T, tie; B, baseline", transform=ax_a.transAxes, fontsize=FOOTNOTE_SIZE, color=MUTED)

    fields = CORE_FIELD_ORDER
    y = np.arange(len(fields))
    left = np.zeros(len(fields))
    for key, color, hatch, label, ink in [("A", HARNESS, "", "Harness", WHITE), ("TIE", TIE, "..", "Tie", TEXT), ("B", BASELINE, "//", "Baseline", WHITE)]:
        values = np.array([counts[field][key] / sum(counts[field].values()) * 100 for field in fields])
        ax_b.barh(y, values, left=left, color=color, hatch=hatch, edgecolor=WHITE, linewidth=0.6, height=0.62, label=label)
        for i, (value, start) in enumerate(zip(values, left)):
            if value >= 12:
                ax_b.text(start + value / 2, i, f"{value:.0f}%", ha="center", va="center", fontsize=7.5, color=ink)
        left += values
    ax_b.set_yticks(y, [str(index) for index in range(1, 8)], fontsize=8)
    ax_b.invert_yaxis()
    ax_b.set_ylim(6.55, -1.55)
    ax_b.set_xlim(0, 100)
    ax_b.set_xticks([0, 50, 100], ["0%", "50%", "100%"])
    clean_axis(ax_b, grid_axis="x")
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=8, handlelength=1.1, columnspacing=0.6)

    result_rows = {row["field"]: row for row in load_csv(CORE_FIELD_RESULTS)}
    for i, field in enumerate(fields):
        row = result_rows[field]
        estimate = float(row["adjusted_probability"]) * 100
        low = float(row["probability_ci_low"]) * 100
        high = float(row["probability_ci_high"]) * 100
        ax_c.hlines(i, low, high, color=HARNESS, linewidth=DATA_WIDTH)
        ax_c.plot(estimate, i, marker="o", markersize=5.5, color=HARNESS, markeredgecolor=WHITE, markeredgewidth=0.8)
        ax_c.text(103, i, f"{estimate:.1f} ({low:.1f}-{high:.1f})", va="center", fontsize=8)
    ax_c.axvline(50, color=MUTED, linewidth=REFERENCE_WIDTH, linestyle="--")
    ax_c.set_xlim(30, 154)
    ax_c.set_xticks([30, 50, 70, 90], ["30%", "50%", "70%", "90%"])
    ax_c.set_yticks(y, [str(index) for index in range(1, 8)], fontsize=8)
    ax_c.invert_yaxis()
    ax_c.set_ylim(6.55, -1.55)
    clean_axis(ax_c, grid_axis="x")
    ax_c.spines["left"].set_visible(False)
    ax_c.tick_params(axis="y", length=0)
    ax_c.set_xlabel("Adjusted harness preference", fontsize=8)
    save_figure(fig, "figure4_effect_profile", "Panel A reports overall and field specific raw counts. Panel B shows seven complete preference distributions including ties. Panel C shows seven adjusted harness preference probabilities and 95 percent confidence intervals; no overall adjusted point is shown.")


def plot_s1(raters: dict[str, list[dict[str, str]]]) -> None:
    all_rows = [row for rows in raters.values() for row in rows]
    groups = [("Breast cancer", note_margins(all_rows, "b", BREAST_FIELDS)), ("PDAC", note_margins(all_rows, "p", PDAC_FIELDS))]
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 1.8))
    fig.subplots_adjust(left=0.17, right=0.975, top=0.78, bottom=0.32)
    ax.axvline(0, color=MUTED, linewidth=REFERENCE_WIDTH, linestyle="--")
    for y, (label, margins) in enumerate(groups):
        values = sorted(net / total for net, total in margins.values())
        freq = Counter(values)
        used: Counter[float] = Counter()
        for value in values:
            rank = used[value]
            used[value] += 1
            jitter = (rank - (freq[value] - 1) / 2) * 0.035
            marker = ">" if value > 0 else "<" if value < 0 else "o"
            color = HARNESS if value > 0 else BASELINE if value < 0 else TIE
            ax.scatter(value, y + jitter, color=color, marker=marker, s=28, edgecolors=WHITE, linewidth=0.6, zorder=3)
    ax.set_yticks([0, 1], ["Breast cancer", "PDAC"])
    ax.invert_yaxis()
    ax.set_xlim(-0.2, 0.8)
    ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8], ["-20%", "0%", "20%", "40%", "60%", "80%"])
    clean_axis(ax, grid_axis="x")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Normalized note margin", fontsize=9)
    ax.set_title("Note level preference margins", loc="left", fontsize=PANEL_TITLE_SIZE, fontweight="bold", pad=10)
    ax.text(0.79, 0, "20/20 positive", ha="right", va="center", fontsize=8, color=HARNESS, fontweight="bold")
    ax.text(0.79, 1, "19/20 positive, 1 negative", ha="right", va="center", fontsize=8, color=HARNESS, fontweight="bold")
    ax.text(0.98, 1.18, "Pooled across four physicians", transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    save_figure(fig, "supplementary_figure_s1_note_margins", "Note level preference margins are positive for all 20 breast cancer notes and 19 of 20 PDAC notes; one PDAC note is negative and none are at the reference line. Position, marker direction, and color provide redundant encoding.")


def plot_s2() -> None:
    counts, pvalues = load_ablation_counts()
    fig = plt.figure(figsize=(FIGURE_WIDTH, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.25], left=0.035, right=0.98, top=0.86, bottom=0.24, wspace=0.18)
    ax_l, ax_r = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for ax in (ax_l, ax_r):
        ax.set_axis_off()
    panel_title(ax_l, "A", "Cumulative configurations")
    panel_title(ax_r, "B", "Adjacent comparisons")

    configs = [
        ("A", "Single prompt baseline", "One inference request per note", BASELINE),
        ("B", "Decomposed extraction", "Add field decomposition", STAGE_B),
        ("C", "Verified extraction", "Add semantic refinement, faithfulness review,\nand temporal verification", STAGE_C),
        ("D", "Full harness", "Add deterministic inline processing\nand complete POST hooks", HARNESS),
    ]
    y_positions = [0.79, 0.56, 0.33, 0.10]
    for i, ((letter, title, detail, color), y) in enumerate(zip(configs, y_positions)):
        add_box(ax_l, (0.02, y), 0.94, 0.145, f"{letter}   {title}\n{detail}", edge=color, face=WHITE, text_color=color if i in (0, 3) else TEXT, weight="bold" if i in (0, 3) else "normal", size=8.0, lw=1.7)
        if i:
            add_arrow(ax_l, (0.49, y + 0.19), (0.49, y + 0.15), color=color, lw=1.6)

    comparisons = [
        ("A to B", "A", "B", BASELINE, STAGE_B, "P = 1.000"),
        ("B to C", "B", "C", STAGE_B, STAGE_C, "P = 1.05 × 10⁻⁵"),
        ("C to D", "C", "D", STAGE_C, HARNESS, "P = .736"),
    ]
    y_positions = [0.76, 0.45, 0.14]
    for (key, earlier, later, earlier_color, later_color, ptext), y in zip(comparisons, y_positions):
        e, l, ties = counts[key]
        ax_r.text(0.01, y + 0.16, f"{earlier} to {later}", transform=ax_r.transAxes, fontsize=9, fontweight="bold", color=TEXT)
        total = e + l + ties
        left = 0.01
        for value, color, hatch, label in [(e, earlier_color, "//", earlier), (l, later_color, "", later), (ties, TIE, "..", "Tie")]:
            width = 0.60 * value / total
            rect = plt.Rectangle((left, y), width, 0.12, transform=ax_r.transAxes, facecolor=color, edgecolor=WHITE, hatch=hatch, linewidth=0.6)
            ax_r.add_patch(rect)
            ax_r.text(left + width / 2, y + 0.06, str(value), transform=ax_r.transAxes, ha="center", va="center", fontsize=8, color=WHITE if color != TIE else TEXT, fontweight="bold")
            left += width
        ax_r.text(0.66, y + 0.06, f"Earlier {e}   Later {l}   Tie {ties}", transform=ax_r.transAxes, va="center", fontsize=8)
        ax_r.text(0.66, y - 0.03, ptext, transform=ax_r.transAxes, va="center", fontsize=8, fontweight="bold", fontfamily="DejaVu Sans")
    fig.text(0.5, 0.105, "40 notes per comparison. Ties were excluded from two sided exact sign tests. The three P values are unadjusted.", ha="center", fontsize=FOOTNOTE_SIZE, color=MUTED)
    fig.text(0.5, 0.06, "One blinded nonphysician OpenAI LLM judge evaluated all 120 comparisons.", ha="center", fontsize=FOOTNOTE_SIZE, color=LLM_REVIEW)
    assert pvalues == {"A to B": 1.0, "B to C": 0.0000105, "C to D": 0.736}
    save_figure(fig, "supplementary_figure_s2_staged_ablation", "Panel A defines four cumulative configurations from the single prompt baseline through the full harness. Panel B reports earlier, later, and tie counts with exact unadjusted P values for 120 blinded comparisons by one nonphysician OpenAI LLM judge.")


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raters = load_raters()
    assert_data(raters)
    plot_figure_1()
    plot_figure_2()
    plot_figure_3(raters)
    plot_figure_4(raters)
    plot_s1(raters)
    plot_s2()
    print(f"Wrote six SVG and six 300 DPI PNG figures to {OUTPUT}")


if __name__ == "__main__":
    main()
