#!/usr/bin/env python3
"""
Analyze attribution results:
1. Overall attribution rate
2. Per-field attribution success
3. Cross-reference with known extraction errors from tracking.md
4. Analyze attribution quality patterns
"""

import json
import sys
from collections import defaultdict


def load_attribution(path):
    """Load attribution JSON results."""
    with open(path) as f:
        return json.load(f)


def load_progress(path):
    """Load progress JSON to get keypoints."""
    with open(path) as f:
        return json.load(f)


# Known extraction errors from tracking.md (Qwen 32B, latest versions)
# Format: coral_idx -> {field_name: "error description"}
# These are the P1/P2 errors from v13 (rows 36-45) and v14 (rows 47-82)
KNOWN_ERRORS = {
    # v13 rows 36-45 (P2 only, P1=0)
    175: {"therapy_plan": "P2: contains valtrex (non-cancer drug)"},
    176: {"therapy_plan": "P2: missing 'dd' (dose-dense)"},
    177: {"recent_changes": "P2: empty (LLM randomness)",
          "Specialty": "P2: missed Gyn Onc referral"},
    178: {"imaging_plan": "P2: missed echocardiogram"},
    179: {"supportive_meds": "P2: ondansetron may not be cancer-related",
          "Others": "P2: PT referral missed"},
    180: {"Patient type": "P2: should be Follow up",
          "Stage_of_Cancer": "P2: empty (LLM randomness)"},
    181: {"Type_of_Cancer": "P2: missing HER2- (redacted)"},
    182: {"Lab_Plan": "P2: missed 'draw'"},
    # v14 rows 47-82 (P1=2 before v14a fix)
    189: {"assessment_plan_text": "P1: A/P regex missed Impression/Plan format"},
    # Llama 8B 100-row review (rows 0-99, coral_idx 140-239)
    # Only including the most severe errors (P0/P1 level)
    160: {"Patient type": "P1: should be New patient (referred for consultation)",
          "Stage_of_Cancer": "P1: empty instead of Stage 0/Tis"},
    161: {"goals_of_treatment": "P1: was 'palliative' in 8B, regressed in early Qwen"},
    162: {"Patient type": "P1: should be New patient"},
}

# Fields that are commonly skip-worthy (not informative for attribution)
SKIP_FIELDS = {
    'assessment_plan_text',  # raw text, not a structured extraction
}


def analyze_results(attribution_path, progress_path):
    """Main analysis function."""
    attribution = load_attribution(attribution_path)
    progress = load_progress(progress_path)
    results = progress.get('results', {})

    # Overall stats
    total_fields = 0
    attributed_fields = 0
    skipped_fields = 0
    field_stats = defaultdict(lambda: {"total": 0, "attributed": 0, "quotes": []})
    per_row_stats = {}

    for row_key, row_data in results.items():
        keypoints = row_data.get('keypoints', {})
        coral_idx = row_data.get('coral_idx', '?')
        row_attribution = attribution.get(row_key, {})

        row_total = 0
        row_attributed = 0

        for prompt_name, fields in keypoints.items():
            if not isinstance(fields, dict):
                continue
            for field_name, field_value in fields.items():
                if field_name in SKIP_FIELDS:
                    skipped_fields += 1
                    continue

                # Check if value is meaningful
                val_str = _flatten_value(field_value)
                if _is_skip_value(val_str):
                    skipped_fields += 1
                    continue

                total_fields += 1
                row_total += 1
                field_stats[field_name]["total"] += 1

                if field_name in row_attribution:
                    attributed_fields += 1
                    row_attributed += 1
                    field_stats[field_name]["attributed"] += 1
                    for q in row_attribution[field_name]:
                        field_stats[field_name]["quotes"].append({
                            "row": row_key,
                            "coral_idx": coral_idx,
                            "value": val_str[:100],
                            "quote": q[:150]
                        })

        per_row_stats[row_key] = {
            "coral_idx": coral_idx,
            "total": row_total,
            "attributed": row_attributed,
            "rate": row_attributed / row_total if row_total > 0 else 0,
        }

    # Print results
    print("=" * 70)
    print("ATTRIBUTION ANALYSIS RESULTS")
    print("=" * 70)

    print(f"\n## Overall Statistics")
    print(f"  Total rows: {len(results)}")
    print(f"  Total attributable fields: {total_fields}")
    print(f"  Attributed fields: {attributed_fields}")
    print(f"  Attribution rate: {attributed_fields/total_fields*100:.1f}%")
    print(f"  Skipped fields (empty/non-informative): {skipped_fields}")

    # Per-row stats
    print(f"\n## Per-Row Attribution Rates")
    print(f"  {'Row':<6} {'coral_idx':<12} {'Attributed':<12} {'Total':<8} {'Rate':<8}")
    print(f"  {'-'*46}")

    rates = []
    for row_key in sorted(per_row_stats.keys(), key=lambda x: int(x)):
        s = per_row_stats[row_key]
        rates.append(s['rate'])
        marker = " ⚠️" if s['rate'] < 0.7 else ""
        print(f"  {row_key:<6} {s['coral_idx']:<12} {s['attributed']:<12} {s['total']:<8} {s['rate']:.1%}{marker}")

    if rates:
        avg_rate = sum(rates) / len(rates)
        min_rate = min(rates)
        max_rate = max(rates)
        print(f"\n  Average rate: {avg_rate:.1%}")
        print(f"  Min rate: {min_rate:.1%}")
        print(f"  Max rate: {max_rate:.1%}")

    # Per-field stats
    print(f"\n## Per-Field Attribution Rates")
    print(f"  {'Field':<35} {'Attributed':<12} {'Total':<8} {'Rate':<8}")
    print(f"  {'-'*63}")

    sorted_fields = sorted(field_stats.items(), key=lambda x: x[1]["attributed"]/max(x[1]["total"],1))
    for field_name, stats in sorted_fields:
        rate = stats["attributed"] / stats["total"] if stats["total"] > 0 else 0
        marker = " ⚠️" if rate < 0.7 else ""
        print(f"  {field_name:<35} {stats['attributed']:<12} {stats['total']:<8} {rate:.1%}{marker}")

    # Fields with lowest attribution rates
    print(f"\n## Fields with Lowest Attribution Rates (<70%)")
    for field_name, stats in sorted_fields:
        rate = stats["attributed"] / stats["total"] if stats["total"] > 0 else 0
        if rate < 0.7 and stats["total"] >= 3:
            print(f"\n  ### {field_name} ({stats['attributed']}/{stats['total']} = {rate:.1%})")
            # Show a few unattributed examples
            # (Would need to cross-reference with progress to find unattributed)

    # Cross-reference with known errors
    print(f"\n## Cross-Reference with Known Extraction Errors")
    known_error_fields = 0
    known_error_attributed = 0
    known_error_details = []

    for row_key, row_data in results.items():
        coral_idx = row_data.get('coral_idx', '?')
        if coral_idx in KNOWN_ERRORS:
            row_attribution = attribution.get(row_key, {})
            for field_name, error_desc in KNOWN_ERRORS[coral_idx].items():
                known_error_fields += 1
                if field_name in row_attribution:
                    known_error_attributed += 1
                    quotes = row_attribution[field_name]
                    known_error_details.append({
                        "row": row_key,
                        "coral_idx": coral_idx,
                        "field": field_name,
                        "error": error_desc,
                        "quote": quotes[0][:150] if quotes else "N/A",
                        "attributed": True,
                    })
                else:
                    known_error_details.append({
                        "row": row_key,
                        "coral_idx": coral_idx,
                        "field": field_name,
                        "error": error_desc,
                        "quote": "N/A",
                        "attributed": False,
                    })

    print(f"  Known error fields found: {known_error_fields}")
    print(f"  Known error fields attributed: {known_error_attributed}")
    if known_error_fields > 0:
        print(f"  Attribution rate for known errors: {known_error_attributed/known_error_fields*100:.1f}%")

    if known_error_details:
        print(f"\n  ### Known Error Attribution Details")
        for d in known_error_details:
            status = "ATTRIBUTED" if d['attributed'] else "NOT ATTRIBUTED"
            print(f"\n  Row {d['row']} (coral_idx={d['coral_idx']}) — {d['field']}")
            print(f"    Error: {d['error']}")
            print(f"    Status: {status}")
            if d['attributed']:
                print(f"    Quote: \"{d['quote']}\"")


# ── helpers (copied from source_attribution.py) ────────────────

_SKIP_EXACT = {
    '', 'none', 'n/a', '[]', '{}',
}

_SKIP_CONTAINS = (
    'the note does not',
    'not provided in the note',
    'not specified in',
    'not mentioned in',
    'not discussed during',
    'not documented',
    'cannot be determined',
    'unable to determine',
    'no data available',
)


def _is_skip_value(value_str):
    val = value_str.strip()
    if len(val) == 0:
        return True
    val_lower = val.lower().rstrip('.').rstrip()
    if val_lower in _SKIP_EXACT:
        return True
    for phrase in _SKIP_CONTAINS:
        if phrase in val_lower:
            return True
    return False


def _flatten_value(field_value):
    if isinstance(field_value, list):
        parts = []
        for item in field_value:
            if isinstance(item, dict):
                parts.extend(str(v) for v in item.values() if v)
            else:
                parts.append(str(item))
        return ' '.join(parts)
    elif isinstance(field_value, dict):
        return ' '.join(str(v) for v in field_value.values() if v)
    return str(field_value)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <attribution.json> <progress.json>")
        sys.exit(1)

    analyze_results(sys.argv[1], sys.argv[2])
