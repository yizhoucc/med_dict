#!/usr/bin/env python3
"""Build the historical PDAC development trajectory from saved artifacts.

This script does not call an LLM and does not reinterpret clinical correctness.
It extracts the counts and finding text already written by auto_review.py, checks
their internal consistency, records missing artifacts, and renders an SVG plot.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Checkpoint:
    trajectory_index: int
    cohort: str
    iteration: int
    label: str
    samples_expected: int
    result_relpath: str | None
    review_relpath: str | None
    secondary_review_relpaths: tuple[str, ...] = ()


CHECKPOINTS = [
    Checkpoint(1, "30-sample development subset", 1, "D1", 30,
               "results/v32_pdac_30_results.txt", "results/v32_pdac_30_review.md",
               ("results/v32_pdac_30_manual_review.md",)),
    Checkpoint(2, "30-sample development subset", 2, "D2", 30,
               None, "results/v32_pdac_30_iter2_review.md"),
    Checkpoint(3, "30-sample development subset", 3, "D3", 30,
               None, "results/v32_pdac_30_iter3_review.md"),
    Checkpoint(4, "30-sample development subset", 4, "D4", 30,
               "results/v32_pdac_30_iter4_results.txt", "results/v32_pdac_30_iter4_review.md"),
    Checkpoint(5, "30-sample development subset", 5, "D5", 30,
               "results/v32_pdac_30_iter5_results.txt", "results/v32_pdac_30_iter5_review.md",
               ("results/v32_pdac_30_iter5_manual_review.md",)),
    Checkpoint(6, "30-sample development subset", 6, "D6", 30,
               "results/v32_pdac_30_iter6_results.txt", "results/v32_pdac_30_iter6_review.md",
               ("results/v32_pdac_30_iter6_manual_review.md",)),
    Checkpoint(7, "30-sample development subset", 7, "D7", 30,
               "results/v32_pdac_30_iter7_results.txt", "results/v32_pdac_30_iter7_review.md"),
    Checkpoint(8, "30-sample development subset", 8, "D8", 30,
               None, "results/v32_pdac_30_iter8_review.md"),
    Checkpoint(9, "30-sample development subset", 9, "D9", 30,
               None, "results/v32_pdac_30_iter9_review.md"),
    Checkpoint(10, "100-sample development set", 2, "F2", 100,
               "results/v32_pdac_full_iter2_results.txt", "results/v32_pdac_full_iter2_review.md",
               ("results/v32_pdac_full_iter2_manual_review.md",)),
    Checkpoint(11, "100-sample development set", 3, "F3", 100,
               "results/v32_pdac_full_iter3_results.txt", "results/v32_pdac_full_iter3_review.md"),
    Checkpoint(12, "100-sample development set", 4, "F4", 100,
               "results/v32_pdac_full_iter4_results.txt", "results/v32_pdac_full_iter4_review.md",
               ("results/v32_pdac_full_iter4_manual_review.md",
                "results/v32_pdac_full_iter4_doctor_feedback.md")),
    Checkpoint(13, "100-sample development set", 5, "F5", 100,
               "results/v32_pdac_full_iter5_results.txt", "results/v32_pdac_full_iter5_review.md",
               ("results/v32_pdac_full_iter5_manual_review.md",)),
    Checkpoint(14, "100-sample development set", 6, "F6", 100,
               "results/v32_pdac_full_iter6_results.txt", None),
    Checkpoint(15, "100-sample development set", 7, "F7", 100,
               "results/v32_pdac_full_iter7_results.txt", None,
               ("results/v32_pdac_full_iter7_doctor_review.md",)),
    Checkpoint(16, "100-sample development set", 8, "F8", 100,
               "results/v32_pdac_full_iter8_results.txt", None),
    Checkpoint(17, "100-sample development set", 9, "F9", 100,
               "results/v32_pdac_full_iter9_results.txt", None),
]


SUMMARY_PATTERNS = {
    "samples": re.compile(r"^- \*\*Samples\*\*:\s*(\d+)\s*$", re.MULTILINE),
    "clean": re.compile(r"^- \*\*Clean\*\*:\s*(\d+)/(\d+)\s*$", re.MULTILINE),
    "p0": re.compile(r"^- \*\*P0\*\*.*?:\s*(\d+)\s*$", re.MULTILINE),
    "p1": re.compile(r"^- \*\*P1\*\*.*?:\s*(\d+)\s*$", re.MULTILINE),
    "p2": re.compile(r"^- \*\*P2\*\*.*?:\s*(\d+)\s*$", re.MULTILINE),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_last_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
            cwd=ROOT,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, ValueError):
        return ""


def flatten_schema(value: object, prefix: str = "") -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            if isinstance(child, dict):
                keys.extend(flatten_schema(child, child_prefix))
            else:
                keys.append(child_prefix)
    return keys


def parse_result(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    header = re.search(r"^Source:\s*(.*?)\s*\|\s*Run started:\s*(.*?)$", text, re.MULTILINE)
    rows = [int(value) for value in re.findall(r"^RESULTS FOR ROW (\d+)\s*$", text, re.MULTILINE)]
    keypoints_match = re.search(
        r"--- Column: keypoints ---\n(.*?)\n\n--- Column:", text, re.DOTALL
    )
    schema_keys: list[str] = []
    schema_parse_ok = False
    if keypoints_match:
        try:
            keypoints = json.loads(keypoints_match.group(1).strip())
            schema_keys = flatten_schema(keypoints)
            schema_parse_ok = True
        except json.JSONDecodeError:
            pass
    signature = hashlib.sha256("\n".join(sorted(schema_keys)).encode()).hexdigest()[:12] if schema_keys else ""
    return {
        "result_source_config": header.group(1).strip() if header else "",
        "result_run_started": header.group(2).strip() if header else "",
        "result_sample_count": len(rows),
        "result_row_ids": ";".join(map(str, rows)),
        "result_schema_parse_ok": int(schema_parse_ok),
        "result_schema_key_count": len(schema_keys),
        "result_schema_signature": signature,
        "result_schema_keys": ";".join(sorted(schema_keys)),
    }


def classify_domain(section: str, field: str) -> str:
    if section == "letter":
        return "letter_output"
    field_l = field.lower()
    if any(token in field_l for token in ("cancer_diagnosis", "type_of_cancer", "stage", "metastasis")):
        return "diagnosis_stage_metastasis"
    if any(token in field_l for token in (
        "current_medications", "treatment_changes", "medication_plan", "therapy_plan",
        "radiotherapy_plan", "surgery_plan",
    )):
        return "medications_treatment"
    if any(token in field_l for token in ("treatment_goals", "response_assessment")):
        return "goals_response"
    if any(token in field_l for token in (
        "imaging_plan", "lab_plan", "genetic_testing_plan", "referral_plan",
        "follow_up_plan", "advance_care",
    )):
        return "follow_up_plans"
    if any(token in field_l for token in ("reason_for_visit", "lab_results", "clinical_findings")):
        return "visit_labs_findings"
    return "other_extraction"


def classify_mechanism(issue: str) -> str:
    issue_l = issue.lower()
    if any(token in issue_l for token in (
        "jargon", "grammar", "voice", "garbled", "unreadable", "readability",
        "incomplete sentence", "sentence fragment", "too complex",
    )):
        return "language_generation"
    if any(token in issue_l for token in (
        "stage", "staging", "metast", "diagnosis", "cancer type", "tumor origin",
    )):
        return "diagnosis_staging"
    if any(token in issue_l for token in (
        "missing", "omitted", "omission", "incomplete", "does not mention",
        "doesn't mention", "does not include", "not include", "lacks", "fails to",
    )):
        return "omission_completeness"
    if any(token in issue_l for token in (
        "current", "past", "future", "planned", "recommended", "continue", "resume",
        "started", "stopped", "temporal", "date", "recent", "historical",
    )):
        return "temporal_status"
    if any(token in issue_l for token in (
        "misclass", "classification", "should be under", "wrong field", "field is empty",
    )):
        return "field_scope"
    if any(token in issue_l for token in (
        "inaccurate", "incorrect", "contradict", "not supported", "fabricat",
        "hallucinat", "misleading", "does not match",
    )):
        return "factual_support"
    return "other"


def parse_review(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    generated_match = re.search(r"^Generated:\s*(.*?)$", text, re.MULTILINE)
    reviewer_match = re.search(r"^Reviewer:\s*(.*?)$", text, re.MULTILINE)
    summary: dict[str, int] = {}
    for key, pattern in SUMMARY_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Missing {key} summary in {path}")
        summary[key] = int(match.group(1))

    findings: list[dict[str, object]] = []
    current_row: int | None = None
    current_section = ""
    detail_started = False
    for line in text.splitlines():
        row_match = re.match(r"^## ROW (\d+)\b", line)
        if row_match:
            detail_started = True
            current_row = int(row_match.group(1))
            current_section = ""
            continue
        if not detail_started:
            continue
        if line.startswith("### Extraction"):
            current_section = "extraction"
            continue
        if line.startswith("### Letter"):
            current_section = "letter"
            continue
        if not line.startswith("| P"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        severity = cells[0] if cells else ""
        if severity not in {"P0", "P1", "P2"}:
            continue
        if current_section == "extraction" and len(cells) >= 3:
            field = cells[1]
            issue = cells[2]
        elif current_section == "letter" and len(cells) >= 2:
            field = "letter"
            issue = cells[1]
        else:
            raise ValueError(f"Unrecognized finding row in {path}: {line}")
        findings.append({
            "row_id": current_row,
            "severity": severity,
            "section": current_section,
            "field": field,
            "issue": issue,
            "domain_category": classify_domain(current_section, field),
            "mechanism_category": classify_mechanism(issue),
        })

    detail_counts = Counter(str(finding["severity"]) for finding in findings)
    for severity in ("P0", "P1", "P2"):
        expected = summary[severity.lower()]
        observed = detail_counts[severity]
        if observed != expected:
            raise ValueError(
                f"Summary/detail mismatch in {path}: {severity} summary={expected}, detail={observed}"
            )

    review_rows = sorted({int(finding["row_id"]) for finding in findings if finding["row_id"] is not None})
    all_rows = sorted({int(value) for value in re.findall(r"^## ROW (\d+)\b", text, re.MULTILINE)})
    return {
        "review_generated": generated_match.group(1).strip() if generated_match else "",
        "reviewer_declared": reviewer_match.group(1).strip() if reviewer_match else "",
        "review_samples": summary["samples"],
        "review_clean": summary["clean"],
        "review_p0": summary["p0"],
        "review_p1": summary["p1"],
        "review_p2": summary["p2"],
        "review_total_flags": summary["p0"] + summary["p1"] + summary["p2"],
        "review_row_count": len(all_rows),
        "review_row_ids": ";".join(map(str, all_rows)),
        "rows_with_flags": len(review_rows),
        "findings": findings,
    }


def per_100(value: int | None, samples: int | None) -> float | None:
    if value is None or not samples:
        return None
    return value * 100.0 / samples


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def svg_escape(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x: float, y: float, text_value: object, *, size: int = 14,
             weight: str = "normal", anchor: str = "start", fill: str = "#202124",
             rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}"{transform}>{svg_escape(text_value)}</text>'
    )


def draw_line_panel(
    elements: list[str], x0: float, y0: float, width: float, height: float,
    title: str, values_by_series: dict[str, list[float | None]], labels: list[str],
    colors: dict[str, str], y_label: str,
) -> None:
    left, right, top, bottom = 64, 18, 38, 55
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom
    all_values = [v for series in values_by_series.values() for v in series if v is not None]
    ymax = max(all_values) if all_values else 1.0
    ymax = max(1.0, math.ceil(ymax / 10.0) * 10.0)
    elements.append(svg_text(x0 + 8, y0 + 23, title, size=16, weight="bold"))
    for tick in range(6):
        value = ymax * tick / 5
        y = py1 - (py1 - py0) * tick / 5
        elements.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        elements.append(svg_text(px0 - 8, y + 4, f"{value:.0f}", size=11, anchor="end", fill="#5F6368"))
    n = len(labels)
    x_positions = [px0 + (px1 - px0) * i / (n - 1) for i in range(n)]
    divider_x = (x_positions[8] + x_positions[9]) / 2
    elements.append(f'<line x1="{divider_x:.1f}" y1="{py0}" x2="{divider_x:.1f}" y2="{py1}" stroke="#9CA3AF" stroke-width="1.5" stroke-dasharray="5 5"/>')
    for i, label in enumerate(labels):
        elements.append(svg_text(x_positions[i], py1 + 20, label, size=10, anchor="middle", fill="#4B5563"))
    elements.append(svg_text((px0 + x_positions[8]) / 2, py1 + 42, "30-sample subset", size=11, anchor="middle", fill="#6B7280"))
    elements.append(svg_text((x_positions[9] + px1) / 2, py1 + 42, "100-sample set", size=11, anchor="middle", fill="#6B7280"))
    elements.append(svg_text(x0 + 17, (py0 + py1) / 2, y_label, size=11, anchor="middle", fill="#4B5563", rotate=-90))
    legend_x = px1 - 10
    for legend_index, (series_name, values) in enumerate(reversed(list(values_by_series.items()))):
        color = colors[series_name]
        ly = y0 + 18 + legend_index * 17
        elements.append(f'<line x1="{legend_x - 118}" y1="{ly - 4}" x2="{legend_x - 96}" y2="{ly - 4}" stroke="{color}" stroke-width="3"/>')
        elements.append(svg_text(legend_x - 90, ly, series_name, size=10, fill="#374151"))
        points: list[tuple[float, float]] = []
        for i, value in enumerate(values):
            if i == 9 and points:
                if len(points) >= 2:
                    elements.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="2.5"/>')
                points = []
            if value is None:
                if len(points) >= 2:
                    elements.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="2.5"/>')
                points = []
                continue
            y = py1 - (value / ymax) * (py1 - py0)
            points.append((x_positions[i], y))
            elements.append(f'<circle cx="{x_positions[i]:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')
        if len(points) >= 2:
            elements.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in points) + f'" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for i in range(13, n):
        x = x_positions[i]
        elements.append(f'<line x1="{x-4:.1f}" y1="{py0+8:.1f}" x2="{x+4:.1f}" y2="{py0+16:.1f}" stroke="#9CA3AF"/>')
        elements.append(f'<line x1="{x+4:.1f}" y1="{py0+8:.1f}" x2="{x-4:.1f}" y2="{py0+16:.1f}" stroke="#9CA3AF"/>')


def draw_heatmap_panel(
    elements: list[str], x0: float, y0: float, width: float, height: float,
    title: str, values: dict[str, list[float | None]], labels: list[str],
) -> None:
    left, right, top, bottom = 175, 18, 44, 55
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom
    rows = list(values)
    all_values = [v for row in values.values() for v in row if v is not None]
    vmax = max(all_values) if all_values else 1.0
    elements.append(svg_text(x0 + 8, y0 + 23, title, size=16, weight="bold"))
    cell_w = (px1 - px0) / len(labels)
    cell_h = (py1 - py0) / len(rows)
    for row_index, row_name in enumerate(rows):
        y = py0 + row_index * cell_h
        elements.append(svg_text(px0 - 8, y + cell_h * 0.65, row_name.replace("_", " "), size=10, anchor="end", fill="#4B5563"))
        for col_index, value in enumerate(values[row_name]):
            x = px0 + col_index * cell_w
            if value is None:
                fill = "#F3F4F6"
                display = "NA"
            else:
                ratio = value / vmax if vmax else 0
                red = int(241 - 110 * ratio)
                green = int(245 - 150 * ratio)
                blue = int(249 - 80 * ratio)
                fill = f"#{red:02x}{green:02x}{blue:02x}"
                display = f"{value:.0f}"
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#FFFFFF" stroke-width="1"/>')
            elements.append(svg_text(x + cell_w / 2, y + cell_h * 0.66, display, size=9, anchor="middle", fill="#111827"))
    for i, label in enumerate(labels):
        x = px0 + (i + 0.5) * cell_w
        elements.append(svg_text(x, py1 + 19, label, size=10, anchor="middle", fill="#4B5563"))
    divider_x = px0 + 9 * cell_w
    elements.append(f'<line x1="{divider_x:.1f}" y1="{py0}" x2="{divider_x:.1f}" y2="{py1}" stroke="#111827" stroke-width="2"/>')
    elements.append(svg_text((px0 + px0 + 9 * cell_w) / 2, py1 + 42, "30-sample subset", size=11, anchor="middle", fill="#6B7280"))
    elements.append(svg_text((px0 + 9 * cell_w + px1) / 2, py1 + 42, "100-sample set", size=11, anchor="middle", fill="#6B7280"))


def write_svg(path: Path, main_rows: list[dict[str, object]], category_rows: list[dict[str, object]]) -> None:
    width, height = 1500, 1120
    labels = [str(row["label"]) for row in main_rows]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#FFFFFF"/>',
        svg_text(48, 38, "PDAC development trajectory from archived automated reviews", size=23, weight="bold"),
        svg_text(48, 62, "Rates are automated reviewer flags per 100 samples. They are not clinician-confirmed error rates.", size=13, fill="#4B5563"),
    ]
    p1 = [row["auto_p1_per_100_samples"] for row in main_rows]
    p2 = [row["auto_p2_per_100_samples"] for row in main_rows]
    ext_p1 = [row["auto_extraction_p1_per_100_samples"] for row in main_rows]
    letter_p1 = [row["auto_letter_p1_per_100_samples"] for row in main_rows]
    draw_line_panel(elements, 40, 82, 700, 410, "A. Major flags recorded by Qwen reviewer", {"P1": p1}, labels, {"P1": "#B91C1C"}, "Flags per 100 samples")
    draw_line_panel(elements, 760, 82, 700, 410, "B. Minor flags recorded by Qwen reviewer", {"P2": p2}, labels, {"P2": "#D97706"}, "Flags per 100 samples")
    draw_line_panel(elements, 40, 515, 700, 410, "C. P1 flags by output section", {"Extraction": ext_p1, "Letter": letter_p1}, labels, {"Extraction": "#2563EB", "Letter": "#7C3AED"}, "Flags per 100 samples")

    domains = [
        "diagnosis_stage_metastasis", "medications_treatment", "goals_response",
        "follow_up_plans", "visit_labs_findings", "letter_output", "other_extraction",
    ]
    category_lookup: dict[tuple[int, str], float] = {}
    for row in category_rows:
        if row["severity"] == "P1" and row["aggregation"] == "domain":
            category_lookup[(int(row["trajectory_index"]), str(row["category"]))] = float(row["flags_per_100_samples"])
    heatmap = {}
    for domain in domains:
        heatmap[domain] = []
        for row in main_rows:
            if not int(row["auto_review_available"]):
                heatmap[domain].append(None)
            else:
                heatmap[domain].append(
                    category_lookup.get((int(row["trajectory_index"]), domain), 0.0)
                )
    draw_heatmap_panel(elements, 760, 515, 700, 410, "D. P1 flags by reviewed output area", heatmap, labels)
    elements.append(svg_text(48, 972, "D1-D9: same 30-row subset. F2-F9: same 100-row development set; F6-F9 have no archived auto_review.py summary and are shown as NA.", size=12, fill="#374151"))
    elements.append(svg_text(48, 994, "The auto-review rubric changed during early iterations and no rubric hash was stored. Compare the series as historical development logs, not as a fixed-validator benchmark.", size=12, fill="#374151"))
    elements.append(svg_text(48, 1016, "P0 was 0 in every archived Qwen review. Manual or Claude-authored follow-up reviews are excluded from these curves.", size=12, fill="#374151"))
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    raw_findings: list[dict[str, object]] = []

    for checkpoint in CHECKPOINTS:
        result_path = ROOT / checkpoint.result_relpath if checkpoint.result_relpath else None
        review_path = ROOT / checkpoint.review_relpath if checkpoint.review_relpath else None
        result_exists = bool(result_path and result_path.exists())
        review_exists = bool(review_path and review_path.exists())
        result_data = parse_result(result_path) if result_exists and result_path else {}
        review_data = parse_review(review_path) if review_exists and review_path else {}

        review_samples = int(review_data["review_samples"]) if review_data else None
        if review_samples is not None and review_samples != checkpoint.samples_expected:
            raise ValueError(f"Unexpected sample count for {checkpoint.label}: {review_samples}")
        if result_data and int(result_data["result_sample_count"]) != checkpoint.samples_expected:
            raise ValueError(
                f"Unexpected result sample count for {checkpoint.label}: "
                f"{result_data['result_sample_count']}"
            )

        findings = list(review_data.get("findings", []))
        for finding in findings:
            raw_findings.append({
                "trajectory_index": checkpoint.trajectory_index,
                "label": checkpoint.label,
                "cohort": checkpoint.cohort,
                "samples": checkpoint.samples_expected,
                **finding,
            })

        p0 = int(review_data["review_p0"]) if review_data else None
        p1 = int(review_data["review_p1"]) if review_data else None
        p2 = int(review_data["review_p2"]) if review_data else None
        extraction_p1 = sum(1 for finding in findings if finding["severity"] == "P1" and finding["section"] == "extraction")
        letter_p1 = sum(1 for finding in findings if finding["severity"] == "P1" and finding["section"] == "letter")
        row_sets_match = ""
        if result_data and review_data:
            row_sets_match = int(result_data["result_row_ids"] == review_data["review_row_ids"])
            if not row_sets_match:
                raise ValueError(f"Result/review ROW mismatch at {checkpoint.label}")
        main_rows.append({
            "trajectory_index": checkpoint.trajectory_index,
            "label": checkpoint.label,
            "phase_interpretation": "ChatGPT rubric-assisted initial PDAC adaptation" if checkpoint.trajectory_index == 1 else "LLM-review-guided refinement checkpoint",
            "cohort": checkpoint.cohort,
            "iteration_within_cohort": checkpoint.iteration,
            "samples_expected": checkpoint.samples_expected,
            "result_file": checkpoint.result_relpath or "",
            "result_file_exists": int(result_exists),
            "result_sha256": sha256_file(result_path) if result_exists and result_path else "",
            "review_file": checkpoint.review_relpath or "",
            "auto_review_available": int(review_exists),
            "review_sha256": sha256_file(review_path) if review_exists and review_path else "",
            "review_git_commit": git_last_commit(review_path) if review_exists and review_path else "",
            "reviewer_declared": review_data.get("reviewer_declared", ""),
            "review_generated": review_data.get("review_generated", ""),
            "auto_review_samples": review_samples if review_samples is not None else "",
            "auto_clean_samples": review_data.get("review_clean", ""),
            "auto_p0": p0 if p0 is not None else "",
            "auto_p1": p1 if p1 is not None else "",
            "auto_p2": p2 if p2 is not None else "",
            "auto_total_flags": review_data.get("review_total_flags", ""),
            "auto_p0_per_100_samples": per_100(p0, review_samples),
            "auto_p1_per_100_samples": per_100(p1, review_samples),
            "auto_p2_per_100_samples": per_100(p2, review_samples),
            "auto_total_flags_per_100_samples": per_100(
                int(review_data["review_total_flags"]) if review_data else None,
                review_samples,
            ),
            "auto_extraction_p1": extraction_p1 if review_data else "",
            "auto_letter_p1": letter_p1 if review_data else "",
            "auto_extraction_p1_per_100_samples": per_100(extraction_p1, review_samples) if review_data else None,
            "auto_letter_p1_per_100_samples": per_100(letter_p1, review_samples) if review_data else None,
            "review_row_count": review_data.get("review_row_count", ""),
            "review_row_ids": review_data.get("review_row_ids", ""),
            "review_result_rows_match": row_sets_match,
            "secondary_review_files": ";".join(checkpoint.secondary_review_relpaths),
            "secondary_review_source": (
                "Claude/manual-style follow-up; not a physician review"
                if checkpoint.secondary_review_relpaths else ""
            ),
            "judge_rubric_hash_recorded": 0 if review_exists else "",
            "comparability_note": (
                "No archived auto_review.py summary; automated counts are NA"
                if not review_exists else
                "Historical Qwen auto-review; rubric hash not stored and auto_review.py changed during development"
            ),
            **result_data,
        })

        if review_data:
            for severity in ("P0", "P1", "P2"):
                for aggregation, key in (("domain", "domain_category"), ("mechanism", "mechanism_category")):
                    counts = Counter(
                        str(finding[key]) for finding in findings if finding["severity"] == severity
                    )
                    for category, count in sorted(counts.items()):
                        category_rows.append({
                            "trajectory_index": checkpoint.trajectory_index,
                            "label": checkpoint.label,
                            "cohort": checkpoint.cohort,
                            "samples": checkpoint.samples_expected,
                            "severity": severity,
                            "aggregation": aggregation,
                            "category": category,
                            "flag_count": count,
                            "flags_per_100_samples": count * 100.0 / checkpoint.samples_expected,
                        })

    main_fields = [
        "trajectory_index", "label", "phase_interpretation", "cohort", "iteration_within_cohort",
        "samples_expected", "result_file", "result_file_exists", "result_sha256",
        "result_source_config", "result_run_started", "result_sample_count", "result_row_ids",
        "result_schema_parse_ok", "result_schema_key_count", "result_schema_signature",
        "result_schema_keys", "review_file", "auto_review_available", "review_sha256",
        "review_git_commit", "reviewer_declared", "review_generated", "auto_review_samples",
        "auto_clean_samples", "auto_p0", "auto_p1", "auto_p2", "auto_total_flags",
        "auto_p0_per_100_samples", "auto_p1_per_100_samples", "auto_p2_per_100_samples",
        "auto_total_flags_per_100_samples", "auto_extraction_p1", "auto_letter_p1",
        "auto_extraction_p1_per_100_samples", "auto_letter_p1_per_100_samples",
        "review_row_count", "review_row_ids", "review_result_rows_match",
        "secondary_review_files", "secondary_review_source", "judge_rubric_hash_recorded",
        "comparability_note",
    ]

    d_review_sets = {row["review_row_ids"] for row in main_rows[:9] if row["review_row_ids"]}
    if len(d_review_sets) != 1:
        raise ValueError("The D1-D9 review ROW sets are not identical")
    full_result_sets = {row["result_row_ids"] for row in main_rows[9:] if row["result_row_ids"]}
    if len(full_result_sets) != 1:
        raise ValueError("The F2-F9 result ROW sets are not identical")
    schema_signatures = {
        row.get("result_schema_signature", "")
        for row in main_rows
        if row.get("result_schema_signature", "")
    }
    if len(schema_signatures) != 1:
        raise ValueError("Available result files do not share one output schema")
    write_csv(OUT_DIR / "pdac_trajectory.csv", main_rows, main_fields)
    write_csv(
        OUT_DIR / "pdac_flag_categories.csv",
        category_rows,
        ["trajectory_index", "label", "cohort", "samples", "severity", "aggregation",
         "category", "flag_count", "flags_per_100_samples"],
    )
    write_csv(
        OUT_DIR / "pdac_flag_details.csv",
        raw_findings,
        ["trajectory_index", "label", "cohort", "samples", "row_id", "severity",
         "section", "field", "domain_category", "mechanism_category", "issue"],
    )
    write_svg(OUT_DIR / "pdac_trajectory.svg", main_rows, category_rows)

    print(f"Wrote {len(main_rows)} checkpoints")
    print(f"Archived automated reviews: {sum(int(row['auto_review_available']) for row in main_rows)}")
    print(f"Parsed flagged findings: {len(raw_findings)}")


if __name__ == "__main__":
    main()
