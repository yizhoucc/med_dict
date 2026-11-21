#!/usr/bin/env python3
"""Build the auditable Word-review text source from the archived physician TXT."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PHYSICIAN_ARCHIVE = ROOT / "WORKSHOP_PAPER_EDITABLE.physician-original.txt"
WORD_SOURCE_JSON = ROOT / "WORKSHOP_PAPER_WORD_SOURCE.json"
WORD_SOURCE_TXT = ROOT / "WORKSHOP_PAPER_WORD_SOURCE.txt"
LEDGER = ROOT / "WORD_SOURCE_DIFF_LEDGER.md"
MANIFEST = ROOT / "SHA256SUMS"
BUILDER = ROOT / "build_word_source.py"

EXPECTED_SOURCE_SHA256 = "81f4fccf08b608aed229408919bfb90bddc9c7f68a0fb056e74a6f953eaedcc4"
LOCKED_PROJECT_TEXT_SHA256 = "f662c3a769a14ff1ffc5fef083ba6e69758599fe30143fb118abe7cfe0d7b754"
LOCKED_PROJECT_COMMIT = "98569a744d2cd40d43419aa1b6c21565b5be1835"
LOCKED_PROJECT_VERSION = "v2.3.0"


@dataclass(frozen=True)
class EditRule:
    before: str
    after: str
    expected_count: int
    reason: str


EDIT_RULES = (
    EditRule("concerns[4]", "concerns [4]", 1, "Add the missing space before citation [4]."),
    EditRule("extraction[2]", "extraction [2]", 1, "Add the missing space before citation [2]."),
    EditRule("sinlge prompt", "single prompt", 1, "Correct the spelling error in the baseline description."),
    EditRule("in single prompt", "in a single prompt", 1, "Add the missing article."),
    EditRule("single-prompt", "single prompt", 3, "Use the physician's preferred unhyphenated term throughout."),
    EditRule("status.Huang", "status. Huang", 1, "Add the missing sentence space."),
    EditRule("have showed", "have shown", 1, "Correct the verb form."),
    EditRule("completeness.[9,10].", "completeness [9,10].", 1, "Correct citation spacing and duplicate punctuation."),
    EditRule("deterministric", "deterministic", 1, "Correct the spelling error."),
    EditRule("post-proccessing", "post processing", 1, "Correct the spelling and use the requested unhyphenated term."),
    EditRule("Verification and postprocessing order", "Verification and post processing order", 1, "Use the requested spaced term in the Supplement heading."),
    EditRule("rules, this suggests", "rules. This suggests", 1, "Repair the comma splice."),
    EditRule(
        "hormone status, which are used to classify breast cancer subtypes but are not applicable to PDAC",
        "hormone status, which is used to classify breast cancer subtypes but is not applicable to PDAC",
        1,
        "Correct subject-verb agreement.",
    ),
    EditRule("In all the 1,040 evaluations", "Across all 1,040 evaluations", 1, "Correct the sentence opening."),
    EditRule("86.21%), the pancreatic cancer estimate", "86.21%). The pancreatic cancer estimate", 1, "Repair the comma splice between the two estimates."),
)


INTRODUCTION_REPLACEMENT = (
    "LLMs can extract information from clinical notes, but their outputs can vary. "
    "They may omit details, confuse past findings with current findings, or add information that is not supported by the original note [4,5]. "
    "In the CORAL benchmark, the commercial models outperformed the open model evaluated in that study [4]. "
    "Sending protected health information (PHI) to externally hosted services raises privacy and governance concerns [2]. "
    "Local deployment can keep PHI within the institution, but it does not by itself ensure accurate extraction. "
    "Although manual review can identify extraction errors, reviewing every output is impractical at scale. "
    "These challenges motivate an LLM-based extraction workflow that improves reliability, preserves data privacy, and reduces manual review."
)

LLM_JUDGE_REPLACEMENT = (
    "Prior studies have examined LLM as a judge evaluation in clinical generative AI, "
    "with agreement varying by rubric and setting [9,10]."
)

ABLATION_REPLACEMENT = (
    "The C versus D comparison did not show a significant directional preference (19 versus 16; P = .736). "
    "Condition D added deterministic inline processing and the full POST hook layer to Condition C, "
    "so this cumulative comparison cannot isolate the effect of any single component. "
    "Some breast cancer rules concerned ER, PR, and HER2 status and were not applicable to PDAC."
)

ANNOTATIONS = {
    16: {
        "id": "AN001",
        "kind": "proposed_replacement",
        "text": f'(Proposed replacement: "{INTRODUCTION_REPLACEMENT}")',
        "reason": "Keep the physician's revised flow while limiting the CORAL comparison to that benchmark and assigning the privacy claim to reference 2.",
    },
    167: {
        "id": "AN002",
        "kind": "proposed_replacement",
        "text": f'(Proposed replacement: "{LLM_JUDGE_REPLACEMENT}")',
        "reason": "Avoid a shared high-concordance claim that references 9 and 10 do not support across all rubrics and settings.",
    },
    171: {
        "id": "AN003",
        "kind": "proposed_replacement",
        "text": f'(Proposed replacement: "{ABLATION_REPLACEMENT}")',
        "reason": "Describe the cumulative C versus D comparison without attributing the result to one component or calling HER2 hormone status.",
    },
    205: {
        "id": "AN004",
        "kind": "reference_recommendation",
        "text": '(Reference recommendation: References 13 and 14 are not cited in the current text and could be removed. Reference 15 could be retained if VALID is cited after "expert reference standards.")',
        "reason": "Keep all physician references in this review source while recording the current citation mismatch.",
    },
}

TOP_HEADINGS = {
    "INTRODUCTION",
    "METHODS",
    "RESULTS",
    "DISCUSSION",
    "REFERENCES",
    "SUPPLEMENTARY MATERIAL",
}
FRONT_HEADINGS = {"TITLE", "AUTHORS", "ABSTRACT"}
SUBHEADINGS = {
    "Study Design",
    "Dataset",
    "Baseline model and model development",
    "Inference Harness",
    "Breast cancer development with physician review",
    "PDAC adaptation with feedback from language models",
    "Clinical evaluation data collection and scoring rubric",
    "Ablation study",
    "Statistical analysis",
    "Sensitivity and physician agreement",
    "Performance across Clinical Extraction Fields",
    "Technical ablation",
    "Supplementary Methods",
    "Data access and holdout definition",
    "Assessment and Plan identification",
    "Verification and post processing order",
    "Insufficient source information",
    "Model roles during PDAC adaptation",
    "Physician evaluation interface",
    "Ablation study judge",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def classify_block(text: str, start_line: int, section: str | None) -> tuple[str, int | None]:
    first = text.splitlines()[0]
    if first in FRONT_HEADINGS:
        return "front_heading", 1
    if first in TOP_HEADINGS:
        return "section_heading", 1
    if first in SUBHEADINGS:
        return "subheading", 2
    if first == "[TODO]":
        return "placeholder", None
    if first.startswith("[Insert "):
        return "figure_placeholder", None
    if re.match(r"^(?:Supplementary )?Figure (?:\d+|S\d+)\.", first):
        return "figure_legend", None
    if re.match(r"^(?:Supplementary )?Table (?:\d+|S\d+)\.", first):
        return "table", None
    if section == "REFERENCES" and re.match(r"^\d+\. ", first):
        return "reference", None
    if re.match(r"^\d+\. Gate \d+ ", first):
        return "ordered_item", None
    return "paragraph", None


def parse_blocks(lines: list[str]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    start: int | None = None
    current: list[str] = []
    section: str | None = None

    def flush(end_line: int) -> None:
        nonlocal start, current, section
        if start is None:
            return
        if len(current) == 2 and current[0] in FRONT_HEADINGS and current[1] == "[TODO]":
            blocks.append(
                {
                    "id": f"B{len(blocks) + 1:03d}",
                    "kind": "front_heading",
                    "origin": "physician_verbatim",
                    "source_line_start": start,
                    "source_line_end": start,
                    "text": current[0],
                    "level": 1,
                }
            )
            blocks.append(
                {
                    "id": f"B{len(blocks) + 1:03d}",
                    "kind": "placeholder",
                    "origin": "physician_verbatim",
                    "source_line_start": start + 1,
                    "source_line_end": start + 1,
                    "text": current[1],
                }
            )
            start = None
            current = []
            return
        text = "\n".join(current)
        kind, level = classify_block(text, start, section)
        block: dict[str, object] = {
            "id": f"B{len(blocks) + 1:03d}",
            "kind": kind,
            "origin": "physician_copyedited" if text != "\n".join(original_lines[start - 1 : end_line]) else "physician_verbatim",
            "source_line_start": start,
            "source_line_end": end_line,
            "text": text,
        }
        if level is not None:
            block["level"] = level
        if kind == "table":
            table_lines = text.splitlines()
            block["caption"] = table_lines[0]
            block["rows"] = [row.split("\t") for row in table_lines[1:]]
        blocks.append(block)
        if text in TOP_HEADINGS:
            section = text
        if end_line in ANNOTATIONS:
            annotation = ANNOTATIONS[end_line]
            blocks.append(
                {
                    "id": annotation["id"],
                    "kind": annotation["kind"],
                    "origin": "editorial_annotation",
                    "anchor_source_line": end_line,
                    "text": annotation["text"],
                    "reason": annotation["reason"],
                }
            )
        start = None
        current = []

    for number, value in enumerate(lines, 1):
        if value == "":
            flush(number - 1)
            continue
        if start is None:
            start = number
        current.append(value)
    flush(len(lines))
    return blocks


source_bytes = PHYSICIAN_ARCHIVE.read_bytes()
if sha256_bytes(source_bytes) != EXPECTED_SOURCE_SHA256:
    raise AssertionError("The archived physician source hash changed")
source_text = source_bytes.decode("utf-8")
original_lines = source_text.splitlines()

edit_records: list[dict[str, object]] = []
edited_text = source_text
edit_number = 1
for rule in EDIT_RULES:
    offsets = [match.start() for match in re.finditer(re.escape(rule.before), edited_text)]
    if len(offsets) != rule.expected_count:
        raise AssertionError(
            f"Expected {rule.expected_count} occurrence(s) of {rule.before!r}, found {len(offsets)}"
        )
    for offset in offsets:
        edit_records.append(
            {
                "id": f"CE{edit_number:03d}",
                "source_line": line_number(edited_text, offset),
                "before": rule.before,
                "after": rule.after,
                "reason": rule.reason,
            }
        )
        edit_number += 1
    edited_text = edited_text.replace(rule.before, rule.after)

edited_lines = edited_text.splitlines()
for index, value in enumerate(edited_lines):
    stripped = value.rstrip()
    if stripped != value:
        edit_records.append(
            {
                "id": f"CE{edit_number:03d}",
                "source_line": index + 1,
                "before": value[len(stripped) :],
                "after": "",
                "reason": "Remove trailing whitespace.",
            }
        )
        edit_number += 1
        edited_lines[index] = stripped

if len(edit_records) != 18:
    raise AssertionError(f"Expected 18 direct copyedit occurrences, found {len(edit_records)}")

blocks = parse_blocks(edited_lines)
linear_text = "\n\n".join(str(block["text"]) for block in blocks) + "\n"

tables = [block for block in blocks if block["kind"] == "table"]
figure_placeholders = [block for block in blocks if block["kind"] == "figure_placeholder"]
figure_legends = [block for block in blocks if block["kind"] == "figure_legend"]
references = [block for block in blocks if block["kind"] == "reference"]
annotations = [block for block in blocks if block["origin"] == "editorial_annotation"]
todo_blocks = [block for block in blocks if block["text"] == "[TODO]"]

if len(tables) != 8:
    raise AssertionError(f"Expected 8 tables, found {len(tables)}")
if len(figure_placeholders) != 6 or len(figure_legends) != 6:
    raise AssertionError("Expected 6 figure placeholders and 6 figure legends")
if len(references) != 15:
    raise AssertionError(f"Expected 15 references, found {len(references)}")
if len(annotations) != 4:
    raise AssertionError(f"Expected 4 editorial annotations, found {len(annotations)}")
if len(todo_blocks) != 3:
    raise AssertionError(f"Expected 3 front-matter TODO blocks, found {len(todo_blocks)}")

original_numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:e[-+]?\d+)?%?|κ|10⁻⁵", source_text)
edited_numbers = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)*(?:e[-+]?\d+)?%?|κ|10⁻⁵", edited_text)
if original_numbers != edited_numbers:
    raise AssertionError("A numeric token changed during direct copyediting")

outline = [
    {"text": block["text"], "kind": block["kind"], "source_line": block.get("source_line_start")}
    for block in blocks
    if block["kind"] in {"front_heading", "section_heading", "subheading"}
]

payload = {
    "schema_version": 1,
    "purpose": "Auditable text authority for the physician Word review draft",
    "authority": {
        "physician_archive": PHYSICIAN_ARCHIVE.name,
        "physician_sha256": EXPECTED_SOURCE_SHA256,
        "physician_size_bytes": len(source_bytes),
        "encoding": "UTF-8 without BOM",
        "locked_project_version": LOCKED_PROJECT_VERSION,
        "locked_project_commit": LOCKED_PROJECT_COMMIT,
        "locked_project_text_sha256": LOCKED_PROJECT_TEXT_SHA256,
        "policy": "Physician prose is authoritative. Only enumerated copyedits are applied directly. Content concerns remain as separate editorial annotations.",
    },
    "counts": {
        "blocks": len(blocks),
        "physician_content_blocks": len(blocks) - len(annotations),
        "direct_copyedit_occurrences": len(edit_records),
        "editorial_annotations": len(annotations),
        "tables": len(tables),
        "figure_placeholders": len(figure_placeholders),
        "figure_legends": len(figure_legends),
        "references": len(references),
        "front_matter_todos": len(todo_blocks),
    },
    "humanizer_review": {
        "scope": "Proposed replacement text only",
        "draft_audit_final_completed": True,
        "audit": [
            "The introduction replacement keeps the physician's order and removes the unsupported broad model comparison.",
            "The LLM judge replacement uses the author-specified cautious evidence boundary.",
            "The ablation replacement reports the observed comparison before stating what the cumulative design cannot isolate.",
            "The proposed replacements contain no em dash or en dash and add no promotional or filler language.",
        ],
    },
    "outline": outline,
    "direct_copyedits": edit_records,
    "editorial_annotations": [
        {key: value for key, value in block.items() if key != "origin"} for block in annotations
    ],
    "linear_text_sha256": sha256_bytes(linear_text.encode("utf-8")),
    "blocks": blocks,
}

WORD_SOURCE_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
WORD_SOURCE_TXT.write_text(linear_text, encoding="utf-8")

diff_lines = list(
    difflib.unified_diff(
        source_text.splitlines(),
        edited_text.splitlines(),
        fromfile=PHYSICIAN_ARCHIVE.name,
        tofile=WORD_SOURCE_TXT.name + " (annotations excluded)",
        lineterm="",
    )
)

ledger_lines = [
    "# Word source diff ledger",
    "",
    "## Authority",
    "",
    f"- Physician archive SHA-256: `{EXPECTED_SOURCE_SHA256}`",
    f"- Locked project boundary: `{LOCKED_PROJECT_VERSION}` at `{LOCKED_PROJECT_COMMIT}`",
    f"- Locked project TXT SHA-256: `{LOCKED_PROJECT_TEXT_SHA256}`",
    "- The physician archive is byte-for-byte unchanged. The Word source applies only the copyedits below and inserts four separate editorial annotations.",
    "",
    "## Direct copyedits",
    "",
    "| ID | Source line | Before | After | Reason |",
    "|---|---:|---|---|---|",
]
for record in edit_records:
    before = str(record["before"]).replace("|", "\\|").replace("\n", "\\n") or "[trailing whitespace]"
    after = str(record["after"]).replace("|", "\\|").replace("\n", "\\n") or "[removed]"
    ledger_lines.append(
        f'| {record["id"]} | {record["source_line"]} | `{before}` | `{after}` | {record["reason"]} |'
    )

ledger_lines.extend(
    [
        "",
        "## Editorial annotations",
        "",
        "The physician sentence or paragraph remains in place after direct copyediting. Each item below follows it as a separate parenthetical note.",
        "",
        "| ID | Anchor line | Type | Text |",
        "|---|---:|---|---|",
    ]
)
for block in annotations:
    annotation_text = str(block["text"]).replace("|", "\\|")
    ledger_lines.append(
        f'| {block["id"]} | {block["anchor_source_line"]} | {block["kind"]} | {annotation_text} |'
    )

ledger_lines.extend(
    [
        "",
        "## Preservation checks",
        "",
        f"- Physician content blocks retained: {len(blocks) - len(annotations)}",
        f"- Direct copyedit occurrences: {len(edit_records)}",
        f"- Separate editorial annotations: {len(annotations)}",
        f"- Tables retained: {len(tables)}",
        f"- Figure placeholders retained: {len(figure_placeholders)}",
        f"- Figure legends retained: {len(figure_legends)}",
        f"- References retained: {len(references)}",
        f"- TITLE, AUTHORS, and ABSTRACT TODO blocks retained: {len(todo_blocks)}",
        "- All numeric tokens are identical before and after direct copyediting.",
        "- Table structure, row order, figure order, section order, result values, and Supplement content are unchanged except for the explicitly listed single prompt terminology edits.",
        "",
        "## Unified diff of direct copyedits",
        "",
        "Editorial annotations are excluded from this diff so that every direct change to physician text is visible.",
        "",
        "```diff",
        *diff_lines,
        "```",
        "",
    ]
)
LEDGER.write_text("\n".join(ledger_lines), encoding="utf-8")

manifest_paths = [PHYSICIAN_ARCHIVE, WORD_SOURCE_JSON, WORD_SOURCE_TXT, LEDGER, BUILDER]
MANIFEST.write_text(
    "".join(f"{sha256_file(path)}  {path.name}\n" for path in manifest_paths),
    encoding="utf-8",
)

print(json.dumps(payload["counts"], sort_keys=True))
print(f"word_source_json_sha256={sha256_file(WORD_SOURCE_JSON)}")
print(f"word_source_txt_sha256={sha256_file(WORD_SOURCE_TXT)}")
