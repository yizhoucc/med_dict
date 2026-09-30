#!/usr/bin/env python3
"""Rebuild the PDAC extraction-only development trajectory.

The script reads archived Qwen auto_review.py reports. It does not call a
model and does not reinterpret the clinical correctness of any finding.
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
    review_relpath: str


CHECKPOINTS = [
    Checkpoint(1, "30-sample development subset", 1, "D1", 30,
               "results/v32_pdac_30_results.txt", "results/v32_pdac_30_review.md"),
    Checkpoint(2, "30-sample development subset", 2, "D2", 30,
               None, "results/v32_pdac_30_iter2_review.md"),
    Checkpoint(3, "30-sample development subset", 3, "D3", 30,
               None, "results/v32_pdac_30_iter3_review.md"),
    Checkpoint(4, "30-sample development subset", 4, "D4", 30,
               "results/v32_pdac_30_iter4_results.txt", "results/v32_pdac_30_iter4_review.md"),
    Checkpoint(5, "30-sample development subset", 5, "D5", 30,
               "results/v32_pdac_30_iter5_results.txt", "results/v32_pdac_30_iter5_review.md"),
    Checkpoint(6, "30-sample development subset", 6, "D6", 30,
               "results/v32_pdac_30_iter6_results.txt", "results/v32_pdac_30_iter6_review.md"),
    Checkpoint(7, "30-sample development subset", 7, "D7", 30,
               "results/v32_pdac_30_iter7_results.txt", "results/v32_pdac_30_iter7_review.md"),
    Checkpoint(8, "30-sample development subset", 8, "D8", 30,
               None, "results/v32_pdac_30_iter8_review.md"),
    Checkpoint(9, "30-sample development subset", 9, "D9", 30,
               None, "results/v32_pdac_30_iter9_review.md"),
    Checkpoint(10, "100-sample development set", 2, "F2", 100,
               "results/v32_pdac_full_iter2_results.txt", "results/v32_pdac_full_iter2_review.md"),
    Checkpoint(11, "100-sample development set", 3, "F3", 100,
               "results/v32_pdac_full_iter3_results.txt", "results/v32_pdac_full_iter3_review.md"),
    Checkpoint(12, "100-sample development set", 4, "F4", 100,
               "results/v32_pdac_full_iter4_results.txt", "results/v32_pdac_full_iter4_review.md"),
]


CATEGORIES = [
    ("diagnosis_stage_metastasis", "Diagnosis / stage / metastasis"),
    ("active_anticancer_medications", "Active anticancer medications"),
    ("treatment_changes_and_plans", "Treatment changes / therapy plan"),
    ("treatment_goal", "Treatment goal"),
    ("response_assessment", "Response assessment"),
    ("laboratory_results", "Laboratory results"),
    ("clinical_findings", "Clinical findings"),
    ("visit_context", "Visit context"),
    ("follow_up_and_other_plans", "Follow-up and other plans"),
    ("other_extraction", "Other extraction"),
]
CATEGORY_LABELS = dict(CATEGORIES)


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


def clinical_category(field: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", field.lower()).strip("_")
    if any(token in normalized for token in (
        "cancer_diagnosis", "type_of_cancer", "stage_of_cancer", "metastasis",
    )):
        return "diagnosis_stage_metastasis"
    if "current_medications" in normalized or "current_meds" in normalized:
        return "active_anticancer_medications"
    if any(token in normalized for token in (
        "treatment_changes", "recent_changes", "medication_plan", "therapy_plan",
        "radiotherapy_plan", "surgery_plan", "procedure_plan",
    )):
        return "treatment_changes_and_plans"
    if "treatment_goals" in normalized or "goals_of_treatment" in normalized:
        return "treatment_goal"
    if "response_assessment" in normalized:
        return "response_assessment"
    if "lab_results" in normalized or "lab_summary" in normalized:
        return "laboratory_results"
    if "clinical_findings" in normalized or normalized == "findings":
        return "clinical_findings"
    if "reason_for_visit" in normalized or normalized in {
        "patient_type", "second_opinion", "in_person", "summary",
    }:
        return "visit_context"
    if any(token in normalized for token in (
        "imaging_plan", "lab_plan", "genetic_testing_plan", "referral",
        "follow_up", "advance_care",
    )):
        return "follow_up_and_other_plans"
    return "other_extraction"


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
        })

    detail_counts = Counter(str(finding["severity"]) for finding in findings)
    for severity in ("P0", "P1", "P2"):
        if detail_counts[severity] != summary[severity.lower()]:
            raise ValueError(
                f"Summary/detail mismatch in {path}: {severity} "
                f"summary={summary[severity.lower()]}, detail={detail_counts[severity]}"
            )

    rows = sorted({int(value) for value in re.findall(r"^## ROW (\d+)\b", text, re.MULTILINE)})
    extraction_p1 = [
        finding for finding in findings
        if finding["severity"] == "P1" and finding["section"] == "extraction"
    ]
    for finding in extraction_p1:
        category = clinical_category(str(finding["field"]))
        finding["category"] = category
        finding["category_label"] = CATEGORY_LABELS[category]
    return {
        "review_generated": generated_match.group(1).strip() if generated_match else "",
        "reviewer_declared": reviewer_match.group(1).strip() if reviewer_match else "",
        "review_samples": summary["samples"],
        "review_row_ids": ";".join(map(str, rows)),
        "extraction_p1": extraction_p1,
    }


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


def svg_text(x: float, y: float, value: object, *, size: int = 14,
             weight: str = "normal", anchor: str = "start",
             fill: str = "#202124", rotate: int | None = None) -> str:
    transform = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="{fill}"{transform}>{svg_escape(value)}</text>'
    )


def draw_trajectory(
    elements: list[str], x0: float, y0: float, width: float, height: float,
    rows: list[dict[str, object]],
) -> None:
    left, right, top, bottom = 78, 25, 48, 66
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom
    labels = [str(row["label"]) for row in rows]
    values = [float(row["extraction_p1_per_100_samples"]) for row in rows]
    ymax = math.ceil(max(values) / 20) * 20
    x_positions = [px0 + (px1 - px0) * i / (len(rows) - 1) for i in range(len(rows))]
    y_position = lambda value: py1 - value / ymax * (py1 - py0)

    elements.append(svg_text(x0 + 8, y0 + 25, "A. Extraction P1 flags across PDAC development", size=18, weight="bold"))
    for tick in range(7):
        value = ymax * tick / 6
        y = y_position(value)
        elements.append(f'<line x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}" stroke="#E5E7EB" stroke-width="1"/>')
        elements.append(svg_text(px0 - 10, y + 4, f"{value:.0f}", size=11, anchor="end", fill="#5F6368"))

    d_points = [(x_positions[i], y_position(values[i])) for i in range(9)]
    f_points = [(x_positions[i], y_position(values[i])) for i in range(9, len(rows))]
    elements.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in d_points) + '" fill="none" stroke="#B91C1C" stroke-width="3"/>')
    elements.append('<polyline points="' + " ".join(f"{x:.1f},{y:.1f}" for x, y in f_points) + '" fill="none" stroke="#2563EB" stroke-width="3"/>')
    elements.append(f'<line x1="{d_points[-1][0]:.1f}" y1="{d_points[-1][1]:.1f}" x2="{f_points[0][0]:.1f}" y2="{f_points[0][1]:.1f}" stroke="#6B7280" stroke-width="2.5" stroke-dasharray="7 6"/>')
    divider_x = (x_positions[8] + x_positions[9]) / 2
    elements.append(f'<line x1="{divider_x:.1f}" y1="{py0}" x2="{divider_x:.1f}" y2="{py1}" stroke="#9CA3AF" stroke-width="1.5" stroke-dasharray="4 5"/>')
    elements.append(svg_text(divider_x + 8, py0 + 15, "Development set expanded", size=11, fill="#4B5563"))
    elements.append(svg_text(divider_x + 8, py0 + 31, "30 to 100 notes; case mix changed", size=11, fill="#4B5563"))

    for i, (x, y) in enumerate(d_points + f_points):
        color = "#B91C1C" if i < 9 else "#2563EB"
        elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.3" fill="{color}"/>')
        elements.append(svg_text(x, y - 9, f"{values[i]:.1f}", size=10, anchor="middle", fill=color))
        elements.append(svg_text(x, py1 + 22, labels[i], size=11, anchor="middle", fill="#374151"))

    elements.append(svg_text((px0 + x_positions[8]) / 2, py1 + 47, "Fixed 30-note subset", size=12, anchor="middle", fill="#6B7280"))
    elements.append(svg_text((x_positions[9] + px1) / 2, py1 + 47, "100-note development set", size=12, anchor="middle", fill="#6B7280"))
    elements.append(svg_text(x0 + 19, (py0 + py1) / 2, "Flags per 100 samples", size=12, anchor="middle", fill="#4B5563", rotate=-90))
    elements.append(f'<line x1="{px1-230}" y1="{y0+18}" x2="{px1-202}" y2="{y0+18}" stroke="#B91C1C" stroke-width="3"/>')
    elements.append(svg_text(px1 - 195, y0 + 22, "30-note subset", size=11, fill="#374151"))
    elements.append(f'<line x1="{px1-110}" y1="{y0+18}" x2="{px1-82}" y2="{y0+18}" stroke="#2563EB" stroke-width="3"/>')
    elements.append(svg_text(px1 - 75, y0 + 22, "100-note set", size=11, fill="#374151"))


def draw_category_heatmap(
    elements: list[str], x0: float, y0: float, width: float, height: float,
    main_rows: list[dict[str, object]], category_rows: list[dict[str, object]],
) -> None:
    left, right, top, bottom = 245, 25, 48, 64
    px0, px1 = x0 + left, x0 + width - right
    py0, py1 = y0 + top, y0 + height - bottom
    labels = [str(row["label"]) for row in main_rows]
    lookup = {
        (int(row["trajectory_index"]), str(row["category"])): float(row["flags_per_100_samples"])
        for row in category_rows
    }
    values = list(lookup.values())
    vmax = max(values) if values else 1.0
    cell_w = (px1 - px0) / len(main_rows)
    cell_h = (py1 - py0) / len(CATEGORIES)

    elements.append(svg_text(x0 + 8, y0 + 25, "B. Extraction P1 flags by clinical category", size=18, weight="bold"))
    for row_index, (category, label) in enumerate(CATEGORIES):
        y = py0 + row_index * cell_h
        elements.append(svg_text(px0 - 10, y + cell_h * 0.67, label, size=11, anchor="end", fill="#374151"))
        for col_index, main_row in enumerate(main_rows):
            value = lookup[(int(main_row["trajectory_index"]), category)]
            ratio = value / vmax if vmax else 0
            red = int(244 - 145 * ratio)
            green = int(247 - 120 * ratio)
            blue = int(250 - 55 * ratio)
            fill = f"#{red:02x}{green:02x}{blue:02x}"
            x = px0 + col_index * cell_w
            elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" fill="{fill}" stroke="#FFFFFF" stroke-width="1"/>')
            elements.append(svg_text(x + cell_w / 2, y + cell_h * 0.67, f"{value:.1f}", size=9, anchor="middle", fill="#111827"))

    divider_x = px0 + 9 * cell_w
    elements.append(f'<line x1="{divider_x:.1f}" y1="{py0}" x2="{divider_x:.1f}" y2="{py1}" stroke="#111827" stroke-width="2"/>')
    for i, label in enumerate(labels):
        x = px0 + (i + 0.5) * cell_w
        elements.append(svg_text(x, py1 + 22, label, size=11, anchor="middle", fill="#374151"))
    elements.append(svg_text((px0 + divider_x) / 2, py1 + 47, "Fixed 30-note subset", size=12, anchor="middle", fill="#6B7280"))
    elements.append(svg_text((divider_x + px1) / 2, py1 + 47, "100-note development set", size=12, anchor="middle", fill="#6B7280"))


def write_svg(path: Path, main_rows: list[dict[str, object]], category_rows: list[dict[str, object]]) -> None:
    width, height = 1500, 1060
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#FFFFFF"/>',
        svg_text(48, 38, "PDAC extraction development trajectory", size=24, weight="bold"),
        svg_text(48, 64, "Archived Qwen automated-review P1 flags, normalized per 100 samples", size=14, fill="#4B5563"),
    ]
    draw_trajectory(elements, 40, 85, 1420, 415, main_rows)
    draw_category_heatmap(elements, 40, 515, 1420, 420, main_rows, category_rows)
    elements.append(svg_text(48, 974, "The dashed segment marks expansion from the fixed 30-note subset to the 100-note development set. Case mix changed at this boundary.", size=12, fill="#374151"))
    elements.append(svg_text(48, 997, "Counts describe issues flagged by the historical automated reviewer. They are not clinician-confirmed error rates.", size=12, fill="#374151"))
    elements.append(svg_text(48, 1020, "The reviewer rubric changed during early D-stage iterations and no exact prompt hash was stored.", size=12, fill="#374151"))
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_rows: list[dict[str, object]] = []
    category_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []

    for checkpoint in CHECKPOINTS:
        review_path = ROOT / checkpoint.review_relpath
        if not review_path.exists():
            raise FileNotFoundError(review_path)
        review = parse_review(review_path)
        review_samples = int(review["review_samples"])
        if review_samples != checkpoint.samples_expected:
            raise ValueError(f"Unexpected review sample count at {checkpoint.label}: {review_samples}")

        result_path = ROOT / checkpoint.result_relpath if checkpoint.result_relpath else None
        result_exists = bool(result_path and result_path.exists())
        result = parse_result(result_path) if result_exists and result_path else {}
        if result and int(result["result_sample_count"]) != checkpoint.samples_expected:
            raise ValueError(f"Unexpected result sample count at {checkpoint.label}")
        if result and result["result_row_ids"] != review["review_row_ids"]:
            raise ValueError(f"Result/review ROW mismatch at {checkpoint.label}")

        extraction_p1 = list(review["extraction_p1"])
        counts = Counter(str(finding["category"]) for finding in extraction_p1)
        if sum(counts.values()) != len(extraction_p1):
            raise ValueError(f"Category total mismatch at {checkpoint.label}")
        for category, category_label in CATEGORIES:
            count = counts.get(category, 0)
            category_rows.append({
                "trajectory_index": checkpoint.trajectory_index,
                "label": checkpoint.label,
                "cohort": checkpoint.cohort,
                "samples": checkpoint.samples_expected,
                "category": category,
                "category_label": category_label,
                "flag_count": count,
                "flags_per_100_samples": count * 100.0 / checkpoint.samples_expected,
            })
        for finding in extraction_p1:
            detail_rows.append({
                "trajectory_index": checkpoint.trajectory_index,
                "label": checkpoint.label,
                "cohort": checkpoint.cohort,
                "samples": checkpoint.samples_expected,
                **finding,
            })

        main_rows.append({
            "trajectory_index": checkpoint.trajectory_index,
            "label": checkpoint.label,
            "phase_interpretation": (
                "ChatGPT rubric-assisted initial PDAC adaptation"
                if checkpoint.trajectory_index == 1
                else "LLM-review-guided refinement checkpoint"
            ),
            "cohort": checkpoint.cohort,
            "iteration_within_cohort": checkpoint.iteration,
            "samples": checkpoint.samples_expected,
            "extraction_p1_flags": len(extraction_p1),
            "extraction_p1_per_100_samples": len(extraction_p1) * 100.0 / checkpoint.samples_expected,
            "review_file": checkpoint.review_relpath,
            "review_sha256": sha256_file(review_path),
            "review_git_commit": git_last_commit(review_path),
            "reviewer_declared": review["reviewer_declared"],
            "review_generated": review["review_generated"],
            "review_row_ids": review["review_row_ids"],
            "result_file": checkpoint.result_relpath or "",
            "result_file_exists": int(result_exists),
            "result_sha256": sha256_file(result_path) if result_exists and result_path else "",
            "judge_rubric_hash_recorded": 0,
            "comparability_note": (
                "Same 30-note subset; historical reviewer rubric changed during D-stage development"
                if checkpoint.trajectory_index <= 9
                else "Same 100-note set and no later committed auto_review.py change after the pre-F2 rubric update"
            ),
            **result,
        })

    d_review_sets = {row["review_row_ids"] for row in main_rows[:9]}
    if len(d_review_sets) != 1:
        raise ValueError("D1-D9 do not share one review ROW set")
    full_review_sets = {row["review_row_ids"] for row in main_rows[9:]}
    if len(full_review_sets) != 1:
        raise ValueError("F2-F4 do not share one review ROW set")
    schema_signatures = {
        row.get("result_schema_signature", "")
        for row in main_rows
        if row.get("result_schema_signature", "")
    }
    if len(schema_signatures) != 1:
        raise ValueError("Available result files do not share one output schema")

    write_csv(
        OUT_DIR / "pdac_trajectory.csv",
        main_rows,
        [
            "trajectory_index", "label", "phase_interpretation", "cohort",
            "iteration_within_cohort", "samples", "extraction_p1_flags",
            "extraction_p1_per_100_samples", "review_file", "review_sha256",
            "review_git_commit", "reviewer_declared", "review_generated",
            "review_row_ids", "result_file", "result_file_exists", "result_sha256",
            "result_source_config", "result_run_started", "result_sample_count",
            "result_row_ids", "result_schema_parse_ok", "result_schema_key_count",
            "result_schema_signature", "result_schema_keys",
            "judge_rubric_hash_recorded", "comparability_note",
        ],
    )
    write_csv(
        OUT_DIR / "pdac_flag_categories.csv",
        category_rows,
        [
            "trajectory_index", "label", "cohort", "samples", "category",
            "category_label", "flag_count", "flags_per_100_samples",
        ],
    )
    write_csv(
        OUT_DIR / "pdac_flag_details.csv",
        detail_rows,
        [
            "trajectory_index", "label", "cohort", "samples", "row_id",
            "field", "category", "category_label", "issue",
        ],
    )
    write_svg(OUT_DIR / "pdac_trajectory.svg", main_rows, category_rows)

    print(f"Wrote {len(main_rows)} extraction checkpoints")
    print(f"Parsed extraction P1 findings: {len(detail_rows)}")
    print(f"Predefined clinical categories per checkpoint: {len(CATEGORIES)}")


if __name__ == "__main__":
    main()
