#!/usr/bin/env python3
"""Audit POST-VISIT-TYPE corrections in v17 results."""

import re

TARGET_ROWS = [2, 7, 8, 9, 49, 60, 62, 63, 79, 82, 85]
RESULTS_FILE = "/Users/yizhoucc/repo/med_dict/results/v17_verify_20260318_184026/results.txt"

def extract_row_section(content, row_num):
    """Extract the entire section for a specific row."""
    # Match from "RESULTS FOR ROW N" to the next "RESULTS FOR ROW"
    pattern = rf"RESULTS FOR ROW {row_num}\n={'='*60}\n\n(.*?)(?=\nRESULTS FOR ROW \d+\n|$)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1)
    return None

def extract_note_text(section):
    """Extract note_text from a row section."""
    pattern = r'--- Column: note_text ---\n"(.*?)"\n\n--- Column: assessment'
    match = re.search(pattern, section, re.DOTALL)
    if match:
        return match.group(1)
    return None

def extract_visit_type(section):
    """Extract the in-person field from keypoints."""
    pattern = r'"in-person":\s*"([^"]+)"'
    match = re.search(pattern, section)
    if match:
        return match.group(1)
    return "UNKNOWN"

def check_evidence(note_text):
    """Check for in-person and telehealth evidence."""
    if not note_text:
        return [], []

    note_lower = note_text.lower()

    # In-person patterns
    in_person_patterns = [
        (r'face-to-face', 'face-to-face'),
        (r'saw\s+(?:her|him|patient|pt|them)\s+in\s+clinic', 'saw in clinic'),
        (r'patient\s+(?:was\s+)?(?:seen\s+)?in\s+clinic', 'in clinic'),
        (r'came\s+(?:in\s+)?to\s+clinic', 'came to clinic'),
        (r'presents?\s+(?:to|in)\s+clinic', 'presents to clinic'),
    ]

    # Telehealth patterns
    telehealth_patterns = [
        (r'televisit', 'televisit'),
        (r'tele[\s\-]?visit', 'tele-visit'),
        (r'video\s+visit', 'video visit'),
        (r'video\s+consult(?:ation)?', 'video consult'),
        (r'telehealth', 'telehealth'),
        (r'tele[\s\-]?health', 'tele-health'),
        (r'telephone\s+visit', 'telephone visit'),
        (r'phone\s+visit', 'phone visit'),
        (r'virtual\s+visit', 'virtual visit'),
    ]

    in_person_ev = []
    for pattern, label in in_person_patterns:
        for m in re.finditer(pattern, note_lower):
            start = max(0, m.start() - 60)
            end = min(len(note_text), m.end() + 60)
            context = note_text[start:end].replace('\n', ' ')
            in_person_ev.append((label, context))

    telehealth_ev = []
    for pattern, label in telehealth_patterns:
        for m in re.finditer(pattern, note_lower):
            start = max(0, m.start() - 60)
            end = min(len(note_text), m.end() + 60)
            context = note_text[start:end].replace('\n', ' ')
            telehealth_ev.append((label, context))

    return in_person_ev, telehealth_ev

def main():
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    print("| Row | Visit Type | In-Person | Telehealth | Verdict | First Evidence |")
    print("|-----|------------|-----------|------------|---------|----------------|")

    results = []
    for row_num in TARGET_ROWS:
        section = extract_row_section(content, row_num)
        if not section:
            print(f"| {row_num} | ERROR | - | - | Row not found | - |")
            continue

        note_text = extract_note_text(section)
        visit_type = extract_visit_type(section)
        in_person_ev, telehealth_ev = check_evidence(note_text)

        # Determine verdict
        has_in_person = len(in_person_ev) > 0
        has_telehealth = len(telehealth_ev) > 0

        if has_in_person and not has_telehealth:
            verdict = "✓ CORRECT"
        elif has_telehealth and not has_in_person:
            verdict = "✗ FALSE POSITIVE"
        elif has_in_person and has_telehealth:
            verdict = "⚠ MIXED"
        else:
            verdict = "⚠ NO EVIDENCE"

        # Format counts
        in_p_count = len(in_person_ev)
        tele_count = len(telehealth_ev)

        # First evidence
        first_ev = ""
        if in_person_ev:
            first_ev = f"{in_person_ev[0][0]}"
        elif telehealth_ev:
            first_ev = f"{telehealth_ev[0][0]}"

        print(f"| {row_num} | {visit_type} | {in_p_count} | {tele_count} | {verdict} | {first_ev} |")
        results.append((row_num, visit_type, in_person_ev, telehealth_ev, verdict))

    # Detailed evidence
    print("\n\n=== DETAILED EVIDENCE ===\n")
    for row_num, visit_type, in_person_ev, telehealth_ev, verdict in results:
        print(f"\n{'='*80}")
        print(f"Row {row_num} - Visit Type: {visit_type} - {verdict}")
        print('='*80)

        if in_person_ev:
            print(f"\nIN-PERSON EVIDENCE ({len(in_person_ev)} matches):")
            for i, (label, context) in enumerate(in_person_ev, 1):
                print(f"  {i}. [{label}] ...{context}...")

        if telehealth_ev:
            print(f"\nTELEHEALTH EVIDENCE ({len(telehealth_ev)} matches):")
            for i, (label, context) in enumerate(telehealth_ev, 1):
                print(f"  {i}. [{label}] ...{context}...")

        if not in_person_ev and not telehealth_ev:
            print("\nNo clear evidence found for visit type.")

if __name__ == "__main__":
    main()
