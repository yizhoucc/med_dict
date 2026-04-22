#!/usr/bin/env python3
"""Compare HF vs vLLM extraction results field by field, sample by sample."""

import json
import re
import sys

def parse_results(filepath):
    """Parse results.txt into {row_num: {field: value}} dict."""
    with open(filepath) as f:
        text = f.read()

    # Split by ROW markers
    rows = {}
    row_splits = re.split(r'RESULTS FOR ROW (\d+)\n=+', text)
    # row_splits: ['header', '1', 'content1', '2', 'content2', ...]

    for i in range(1, len(row_splits), 2):
        row_num = int(row_splits[i])
        content = row_splits[i + 1]

        # Extract keypoints JSON
        kp_match = re.search(r'--- Column: keypoints ---\n(.+?)(?=\n--- Column:|$)', content, re.DOTALL)
        if not kp_match:
            continue

        kp_text = kp_match.group(1).strip()
        try:
            kp = json.loads(kp_text)
        except json.JSONDecodeError:
            print(f"  WARNING: Could not parse keypoints for ROW {row_num}")
            continue

        # Flatten nested dict
        flat = {}
        for section, values in kp.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat[k] = str(v).strip() if v else ""
            else:
                flat[section] = str(values).strip() if values else ""

        rows[row_num] = flat

    return rows

def normalize(val):
    """Normalize a value for comparison."""
    if not val:
        return ""
    val = str(val).strip()
    # Normalize whitespace
    val = re.sub(r'\s+', ' ', val)
    return val

# 11 key extraction fields to compare
KEY_FIELDS = [
    "Type_of_Cancer",
    "Stage_of_Cancer",
    "Distant Metastasis",
    "response_assessment",
    "current_meds",
    "goals_of_treatment",
    "therapy_plan",
    "imaging_plan",
    "lab_plan",
    "genetic_testing_plan",
    "Medication_Plan",  # sometimes "medication_plan"
]

# Fields where empty = bad
SHOULD_NOT_BE_EMPTY = [
    "Type_of_Cancer",
    "Stage_of_Cancer",
    "response_assessment",
    "goals_of_treatment",
]

def compare(hf_file, vllm_file):
    print("Parsing HF results...")
    hf = parse_results(hf_file)
    print(f"  Found {len(hf)} rows")

    print("Parsing vLLM results...")
    vllm = parse_results(vllm_file)
    print(f"  Found {len(vllm)} rows")

    # Find matched rows
    matched = sorted(set(hf.keys()) & set(vllm.keys()))
    print(f"\nMatched rows: {len(matched)}")

    # Field-level stats
    field_stats = {f: {"hf_wins": 0, "vllm_wins": 0, "tie": 0, "hf_empty": 0, "vllm_empty": 0} for f in KEY_FIELDS}

    # Collect all differences for detailed output
    all_diffs = []

    for row in matched:
        hf_kp = hf[row]
        vllm_kp = vllm[row]

        for field in KEY_FIELDS:
            # Handle case variations
            hf_val = normalize(hf_kp.get(field, "") or hf_kp.get(field.lower(), "") or hf_kp.get(field.replace(" ", "_"), ""))
            vllm_val = normalize(vllm_kp.get(field, "") or vllm_kp.get(field.lower(), "") or vllm_kp.get(field.replace(" ", "_"), ""))

            # Also try medication_plan vs Medication_Plan
            if field == "Medication_Plan" and not hf_val:
                hf_val = normalize(hf_kp.get("medication_plan", ""))
            if field == "Medication_Plan" and not vllm_val:
                vllm_val = normalize(vllm_kp.get("medication_plan", ""))

            if hf_val == vllm_val:
                field_stats[field]["tie"] += 1
                continue

            # Check empties
            hf_empty = not hf_val or hf_val.lower() in ["", "none", "none planned.", "not mentioned", "not mentioned in note", "no", "n/a"]
            vllm_empty = not vllm_val or vllm_val.lower() in ["", "none", "none planned.", "not mentioned", "not mentioned in note", "no", "n/a"]

            if hf_empty:
                field_stats[field]["hf_empty"] += 1
            if vllm_empty:
                field_stats[field]["vllm_empty"] += 1

            # Determine who's better based on content length/detail (rough heuristic)
            diff_entry = {
                "row": row,
                "field": field,
                "hf": hf_val[:200],
                "vllm": vllm_val[:200],
                "hf_empty": hf_empty,
                "vllm_empty": vllm_empty,
            }

            # Clear winner: one has content, other is empty
            if vllm_empty and not hf_empty:
                diff_entry["winner"] = "HF"
                field_stats[field]["hf_wins"] += 1
            elif hf_empty and not vllm_empty:
                diff_entry["winner"] = "vLLM"
                field_stats[field]["vllm_wins"] += 1
            else:
                # Both have content - need manual review
                diff_entry["winner"] = "DIFF"
                # Rough heuristic: longer = more detailed (not always right but useful for sorting)
                if len(vllm_val) > len(hf_val) * 1.2:
                    diff_entry["likely"] = "vLLM (more detail)"
                elif len(hf_val) > len(vllm_val) * 1.2:
                    diff_entry["likely"] = "HF (more detail)"
                else:
                    diff_entry["likely"] = "similar length"

            all_diffs.append(diff_entry)

    # Print summary
    print("\n" + "="*80)
    print("FIELD-LEVEL SUMMARY (clear wins only — empty vs content)")
    print("="*80)
    print(f"{'Field':<25} {'HF wins':>8} {'vLLM wins':>9} {'Tie':>5} {'DIFF':>5} {'HF empty':>9} {'vLLM empty':>10}")
    print("-"*80)

    total_hf = 0
    total_vllm = 0
    total_diff = 0

    for field in KEY_FIELDS:
        s = field_stats[field]
        diff_count = sum(1 for d in all_diffs if d["field"] == field and d["winner"] == "DIFF")
        print(f"{field:<25} {s['hf_wins']:>8} {s['vllm_wins']:>9} {s['tie']:>5} {diff_count:>5} {s['hf_empty']:>9} {s['vllm_empty']:>10}")
        total_hf += s["hf_wins"]
        total_vllm += s["vllm_wins"]
        total_diff += diff_count

    print("-"*80)
    print(f"{'TOTAL':<25} {total_hf:>8} {total_vllm:>9} {'':>5} {total_diff:>5}")

    # Print all HF wins (vLLM needs to fix these)
    hf_wins = [d for d in all_diffs if d["winner"] == "HF"]
    print(f"\n\n{'='*80}")
    print(f"vLLM LOSSES — {len(hf_wins)} cases where HF has content but vLLM is empty/none")
    print(f"{'='*80}")
    for d in sorted(hf_wins, key=lambda x: (x["field"], x["row"])):
        print(f"\nROW {d['row']} | {d['field']}")
        print(f"  HF:   {d['hf']}")
        print(f"  vLLM: {d['vllm']}")

    # Print all DIFF cases (need manual review)
    diff_cases = [d for d in all_diffs if d["winner"] == "DIFF"]
    print(f"\n\n{'='*80}")
    print(f"CONTENT DIFFERENCES — {len(diff_cases)} cases where both have content but differ")
    print(f"{'='*80}")
    for d in sorted(diff_cases, key=lambda x: (x["field"], x["row"])):
        likely = d.get("likely", "")
        print(f"\nROW {d['row']} | {d['field']} [{likely}]")
        print(f"  HF:   {d['hf']}")
        print(f"  vLLM: {d['vllm']}")

if __name__ == "__main__":
    hf_file = "results/v31_full_20260413_221315/results.txt"
    vllm_file = "results/v31_vllm_iter5e_results.txt"
    compare(hf_file, vllm_file)
