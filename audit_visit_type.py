#!/usr/bin/env python3
"""Audit POST-VISIT-TYPE corrections in v17 results."""

import re
import json

TARGET_ROWS = [2, 7, 8, 9, 49, 60, 62, 63, 79, 82, 85]
RESULTS_FILE = "/Users/yizhoucc/repo/med_dict/results/v17_verify_20260318_184026/results.txt"

def extract_row_data(file_path, row_num):
    """Extract note_text and keypoints for a specific row."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Find the row section
    pattern = rf"RESULTS FOR ROW {row_num}\n={60,}\n\n--- Column: coral_idx ---\n(\d+)\n\n--- Column: note_text ---\n\"(.*?)\"\n\n--- Column: assessment_and_plan ---\n.*?\n\n--- Column: keypoints ---\n(\{{.*?\n\}})"

    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None, None, None

    coral_idx = match.group(1)
    note_text = match.group(2)
    keypoints_str = match.group(3)

    try:
        keypoints = json.loads(keypoints_str)
    except json.JSONDecodeError:
        keypoints = None

    return coral_idx, note_text, keypoints

def check_visit_type_evidence(note_text):
    """Check for evidence of in-person vs televisit."""
    note_lower = note_text.lower()

    # In-person evidence
    in_person_patterns = [
        r'face-to-face',
        r'in[\s\-]person',
        r'saw\s+(?:her|him|patient|pt)\s+in\s+clinic',
        r'patient\s+in\s+clinic',
        r'came\s+in\s+to\s+clinic',
        r'physical\s+exam',
        r'examined',
    ]

    # Telehealth evidence
    telehealth_patterns = [
        r'televisit',
        r'tele[\s\-]?visit',
        r'video\s+visit',
        r'video\s+consult',
        r'telehealth',
        r'tele[\s\-]?health',
        r'telephone\s+visit',
        r'phone\s+visit',
    ]

    in_person_matches = []
    for pattern in in_person_patterns:
        matches = re.finditer(pattern, note_lower)
        for m in matches:
            # Get context (50 chars before and after)
            start = max(0, m.start() - 50)
            end = min(len(note_text), m.end() + 50)
            context = note_text[start:end].replace('\n', ' ')
            in_person_matches.append(context)

    telehealth_matches = []
    for pattern in telehealth_patterns:
        matches = re.finditer(pattern, note_lower)
        for m in matches:
            start = max(0, m.start() - 50)
            end = min(len(note_text), m.end() + 50)
            context = note_text[start:end].replace('\n', ' ')
            telehealth_matches.append(context)

    return in_person_matches, telehealth_matches

def main():
    print("| Row | Coral Idx | Original Value | In-Person Evidence | Telehealth Evidence | Correction OK? | Notes |")
    print("|-----|-----------|----------------|-------------------|---------------------|----------------|-------|")

    for row_num in TARGET_ROWS:
        coral_idx, note_text, keypoints = extract_row_data(RESULTS_FILE, row_num)

        if note_text is None:
            print(f"| {row_num} | N/A | ERROR: Row not found | | | | |")
            continue

        # Get the visit type from keypoints
        visit_type = "UNKNOWN"
        if keypoints and "Reason_for_Visit" in keypoints:
            visit_type = keypoints["Reason_for_Visit"].get("in-person", "UNKNOWN")

        # Check for evidence
        in_person_ev, telehealth_ev = check_visit_type_evidence(note_text)

        # Determine if correction was correct
        has_in_person = len(in_person_ev) > 0
        has_telehealth = len(telehealth_ev) > 0

        if has_in_person and not has_telehealth:
            verdict = "✓ Yes"
        elif has_telehealth and not has_in_person:
            verdict = "✗ No (False Positive)"
        elif has_in_person and has_telehealth:
            verdict = "⚠ Uncertain (Mixed)"
        else:
            verdict = "⚠ No Evidence"

        # Format evidence
        in_person_str = f"{len(in_person_ev)} matches" if in_person_ev else "None"
        telehealth_str = f"{len(telehealth_ev)} matches" if telehealth_ev else "None"

        # Show first evidence snippet
        note = ""
        if in_person_ev:
            note = f"First: '{in_person_ev[0][:60]}...'"
        elif telehealth_ev:
            note = f"First: '{telehealth_ev[0][:60]}...'"

        print(f"| {row_num} | {coral_idx} | {visit_type} | {in_person_str} | {telehealth_str} | {verdict} | {note} |")

    # Print detailed evidence for inspection
    print("\n\n=== DETAILED EVIDENCE ===\n")
    for row_num in TARGET_ROWS:
        coral_idx, note_text, keypoints = extract_row_data(RESULTS_FILE, row_num)
        if note_text is None:
            continue

        in_person_ev, telehealth_ev = check_visit_type_evidence(note_text)

        print(f"\n--- Row {row_num} (coral_idx: {coral_idx}) ---")

        if in_person_ev:
            print(f"\nIN-PERSON evidence ({len(in_person_ev)} matches):")
            for i, ev in enumerate(in_person_ev[:3], 1):  # Show first 3
                print(f"  {i}. ...{ev}...")

        if telehealth_ev:
            print(f"\nTELEHEALTH evidence ({len(telehealth_ev)} matches):")
            for i, ev in enumerate(telehealth_ev[:3], 1):
                print(f"  {i}. ...{ev}...")

        if not in_person_ev and not telehealth_ev:
            print("\nNo clear evidence found.")

if __name__ == "__main__":
    main()
