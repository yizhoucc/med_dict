#!/usr/bin/env python3
"""
baseline_extraction.py — Extract clinical info with NO pipeline, NO gates, NO hooks.
Just raw note → one model call with one prompt → extraction.

The preferred ``--matched`` mode uses the same output fields and operational
field definitions as the pipeline evaluation while deliberately retaining the
single-pass baseline design.  The historical ``--json`` and free-text modes
remain available so previously generated results stay reproducible.

Usage:
    python3 baseline_extraction.py data/CORAL/.../breastca_annotated.csv --output results/baseline_extract_breast.txt
    python3 baseline_extraction.py data/CORAL/.../pdac_annotated.csv --output results/baseline_extract_pdac.txt --indices 0,1,2
    python3 baseline_extraction.py data/CORAL/.../breastca_annotated.csv --output results/baseline_extract_breast_matched.txt --matched --cancer-type breast
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml

BASELINE_PROMPT_FREETEXT = """Read the following oncology clinical note and extract the key clinical information.

Include:
- Reason for visit
- Cancer diagnosis (type, stage, metastasis)
- Lab results
- Clinical findings
- Current medications
- Treatment changes
- Treatment goals
- Response to treatment
- Plan: medications, therapy, radiation, procedures, imaging, labs, genetic testing, referrals, follow-up, advance care planning

Clinical Note:
{note_text}

Extract the information now."""

BASELINE_PROMPT_JSON = """Read the following oncology clinical note and extract the key clinical information.

Respond ONLY with a single JSON object using the exact schema below. Do not include any text before or after the JSON.

{{
  "Reason_for_Visit": {{
    "Patient type": "New patient or Follow up",
    "summary": "A specific summary including cancer type, stage if known, and purpose of this visit."
  }},
  "Cancer_Diagnosis": {{
    "Type_of_Cancer": "Specific cancer type with histologic subtype and receptor statuses when available",
    "Stage_of_Cancer": "AJCC stage or description",
    "Distant Metastasis": "Yes (to where), No, or Not sure"
  }},
  "Lab_Results": {{
    "lab_summary": "Specific lab values with numbers, or No labs in note"
  }},
  "Clinical_Findings": {{
    "findings": "Key clinical and imaging findings"
  }},
  "Current_Medications": {{
    "current_meds": "Currently active medications"
  }},
  "Treatment_Changes": {{
    "recent_changes": "Recent treatment changes",
    "supportive_meds": "Supportive medications"
  }},
  "Treatment_Goals": {{
    "goals_of_treatment": "curative, palliative, adjuvant, risk reduction, surveillance, or symptom management",
    "goals_description": "Physician's explicit treatment intent language if stated"
  }},
  "Response_Assessment": {{
    "response_assessment": "How the cancer is currently responding to treatment"
  }},
  "Medication_Plan": {{
    "medication_plan": "Planned medication changes"
  }},
  "Therapy_plan": {{
    "therapy_plan": "Planned therapy"
  }},
  "radiotherapy_plan": {{
    "radiotherapy_plan": "Planned radiation therapy or None"
  }},
  "Procedure_Plan": {{
    "procedure_plan": "Planned procedures or None"
  }},
  "Imaging_Plan": {{
    "imaging_plan": "Planned imaging or None"
  }},
  "Lab_Plan": {{
    "lab_plan": "Planned labs or None"
  }},
  "Genetic_Testing_Plan": {{
    "genetic_testing_plan": "Planned genetic testing or None"
  }},
  "Genetic_Testing_Results": {{
    "genetic_testing_results": "Completed genetic/molecular test results or No genetic testing results in note"
  }},
  "Referral": {{
    "Nutrition": "Nutrition referral or None",
    "Genetics": "Genetics referral or None",
    "Specialty": "Specialty referrals or None",
    "Others": "Other referrals or None"
  }},
  "follow_up_next_visit": {{
    "Next clinic visit": "Timing and purpose of next visit"
  }},
  "Advance_care_planning": {{
    "Advance care": "Advance care planning details or Not discussed"
  }}
}}

Clinical Note:
{note_text}

Respond with the JSON object now."""


DEFAULT_MATCHED_CONTRACT = Path(__file__).resolve().parent / "prompts" / "matched_baseline_contract.yaml"


def infer_cancer_type(csv_path):
    """Infer the cancer-specific contract from an unambiguous dataset name."""
    name = Path(csv_path).name.lower()
    if "pdac" in name or "pancre" in name:
        return "pdac"
    if "breast" in name:
        return "breast"
    raise ValueError(
        "Could not infer cancer type from the CSV filename; pass --cancer-type breast or pdac."
    )


def load_matched_contract(path, cancer_type):
    """Load and validate the auditable matched-baseline task contract."""
    contract_path = Path(path)
    raw = contract_path.read_bytes()
    contract = yaml.safe_load(raw)
    if cancer_type not in contract.get("cancer_fields", {}):
        raise ValueError(f"Contract {contract_path} has no cancer_fields entry for {cancer_type!r}")
    if not isinstance(contract.get("output_schema"), dict):
        raise ValueError(f"Contract {contract_path} is missing output_schema")
    return contract, hashlib.sha256(raw).hexdigest()


def build_matched_prompt(contract, cancer_type, note_text):
    """Build one prompt for one baseline call; no staged prompting or repair is used."""
    field_definitions = dict(contract.get("common_fields", {}))
    field_definitions.update(contract["cancer_fields"][cancer_type])
    definitions = "\n".join(
        f"- {field_path}: {description}" for field_path, description in field_definitions.items()
    )
    schema = json.dumps(contract["output_schema"], indent=2, ensure_ascii=False)
    return (
        f"{contract['instructions'].strip()}\n\n"
        f"CANCER-SPECIFIC TASK: {cancer_type}\n\n"
        "FIELD DEFINITIONS (use these exact meanings):\n"
        f"{definitions}\n\n"
        "Return ONLY one JSON object with exactly this nested schema. "
        "Use an empty string when the note contains no supported value; do not omit keys.\n"
        f"{schema}\n\n"
        "CLINICAL NOTE:\n"
        f"{note_text}\n\n"
        "Return the JSON object now."
    )


def missing_schema_paths(expected, actual, prefix=""):
    """Return nested key paths missing from a generated JSON object."""
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return [prefix.rstrip(".") or "<root>"]
    missing = []
    for key, expected_value in expected.items():
        path = f"{prefix}{key}"
        if key not in actual:
            missing.append(path)
        elif isinstance(expected_value, dict):
            missing.extend(missing_schema_paths(expected_value, actual[key], path + "."))
    return missing


def main():
    parser = argparse.ArgumentParser(description="Baseline extraction (no pipeline)")
    parser.add_argument("csv_path", help="Path to CSV with note_text column")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--indices", help="Comma-separated row indices (default: all)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--json", action="store_true", help="Use the historical JSON prompt")
    mode_group.add_argument(
        "--matched",
        action="store_true",
        help="Use the matched single-prompt contract (recommended for PL-vs-BL ablation)",
    )
    parser.add_argument("--cancer-type", choices=("breast", "pdac"), help="Required by --matched unless inferable from CSV filename")
    parser.add_argument("--contract", default=str(DEFAULT_MATCHED_CONTRACT), help="Matched-baseline contract YAML")
    args = parser.parse_args()

    from vllm_pipeline.vllm_client import VLLMClient

    client = VLLMClient(base_url=args.base_url, model_name=args.model)
    if not client.health_check():
        print("ERROR: vLLM server not reachable")
        sys.exit(1)
    print(f"Connected to vLLM at {args.base_url}")

    with open(args.csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    note_col = header.index('note_text')
    coral_col = header.index('coral_idx') if 'coral_idx' in header else 0

    if args.indices:
        indices = [int(x) for x in args.indices.split(',')]
    else:
        indices = list(range(len(rows)))

    print(f"Loaded {len(rows)} rows, processing {len(indices)}")

    cancer_type = None
    contract = None
    contract_hash = None
    if args.matched:
        cancer_type = args.cancer_type or infer_cancer_type(args.csv_path)
        contract, contract_hash = load_matched_contract(args.contract, cancer_type)

    prompt_template = BASELINE_PROMPT_JSON if args.json else BASELINE_PROMPT_FREETEXT
    structured_output = args.json or args.matched
    gen_config = {"max_new_tokens": 4096 if structured_output else 2048, "do_sample": False}
    total_start = time.time()

    json_ok = 0
    json_fail = 0
    json_errors = []

    with open(args.output, 'w') as out:
        mode = "matched-json" if args.matched else ("legacy-json" if args.json else "legacy-free-text")
        out.write(f"BASELINE EXTRACTION — No pipeline, single prompt, no post-processing ({mode})\n")
        out.write(f"Model: {args.model}\n")
        out.write(f"Source: {args.csv_path}\n")
        if args.matched:
            out.write(f"Cancer type: {cancer_type}\n")
            out.write(f"Task contract: {args.contract}\n")
            out.write(f"Task contract SHA256: {contract_hash}\n")
        out.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        out.write(f"{'='*60}\n\n")

        for idx in indices:
            row = rows[idx]
            note_text = row[note_col]
            coral_id = row[coral_col] if coral_col < len(row) else str(idx)

            print(f"ROW {idx+1} (coral={coral_id}, {len(note_text)} chars)...", end=" ", flush=True)
            t0 = time.time()

            if args.matched:
                prompt = build_matched_prompt(contract, cancer_type, note_text)
            else:
                prompt = prompt_template.format(note_text=note_text)
            extraction = client.chat_generate(prompt, gen_config)

            elapsed = time.time() - t0

            json_valid = False
            json_err_msg = ""
            if structured_output:
                text = extraction.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    text = text.rsplit("```", 1)[0]
                try:
                    parsed = json.loads(text)
                    if args.matched:
                        missing = missing_schema_paths(contract["output_schema"], parsed)
                    else:
                        expected_keys = {"Reason_for_Visit", "Cancer_Diagnosis", "Lab_Results", "Treatment_Goals", "Referral"}
                        found_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
                        missing = sorted(expected_keys - found_keys)
                    if not missing:
                        json_valid = True
                        json_ok += 1
                    else:
                        json_err_msg = f"JSON parsed but missing keys: {missing[:5]}"
                        json_fail += 1
                except json.JSONDecodeError as e:
                    json_err_msg = f"JSON parse error at pos {e.pos}: {str(e)[:60]}"
                    json_fail += 1
                except Exception as e:
                    json_err_msg = f"Error: {str(e)[:60]}"
                    json_fail += 1

                status = "JSON_OK" if json_valid else f"JSON_FAIL ({json_err_msg})"
                print(f"{elapsed:.1f}s, {len(extraction)} chars, {status}")
                if not json_valid:
                    json_errors.append((idx+1, coral_id, json_err_msg))
            else:
                print(f"{elapsed:.1f}s, {len(extraction)} chars")

            out.write(f"{'='*60}\n")
            out.write(f"ROW {idx+1} (coral_idx={coral_id})\n")
            out.write(f"{'='*60}\n\n")
            out.write(extraction)
            out.write(f"\n\n--- STATS ---\n")
            out.write(f"Time: {elapsed:.1f}s | Output: {len(extraction)} chars")
            if structured_output:
                out.write(f" | JSON: {'OK' if json_valid else 'FAIL - ' + json_err_msg}")
            out.write(f"\n\n\n")

    total = time.time() - total_start
    print(f"\nDone! {len(indices)} samples in {total:.0f}s ({total/60:.1f}min)")
    print(f"Output: {args.output}")

    if structured_output:
        print(f"\n--- JSON VALIDATION SUMMARY ---")
        print(f"  Valid:   {json_ok}/{len(indices)} ({100*json_ok/len(indices):.0f}%)")
        print(f"  Invalid: {json_fail}/{len(indices)} ({100*json_fail/len(indices):.0f}%)")
        if json_errors:
            print(f"  Failures:")
            for row_n, coral, err in json_errors:
                print(f"    ROW {row_n} (coral={coral}): {err}")


if __name__ == "__main__":
    main()
