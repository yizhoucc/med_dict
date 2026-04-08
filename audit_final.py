#!/usr/bin/env python3
"""Final audit of POST-VISIT-TYPE corrections"""

import re

RESULTS_FILE = "/Users/yizhoucc/repo/med_dict/results/v17_verify_20260318_184026/results.txt"

# Rows that triggered the POST-VISIT-TYPE rule (from user's request)
TARGET_ROWS = [2, 7, 8, 9, 49, 60, 62, 63, 79, 82, 85]

# Known row positions (from find_row_lines.py)
ROW_POSITIONS = {
    2: 601,
    7: 1199,
    8: 1335,
    9: 1474,
    49: 4613,
    63: 5743,
    82: 7157,
    85: 7589,
}

def get_row_section(content_lines, start_line, row_num):
    """Extract section for a specific row."""
    # Find the next row's start line
    next_row_num = row_num + 1
    while next_row_num < 200:
        next_line_pattern = f"RESULTS FOR ROW {next_row_num}"
        for i, line in enumerate(content_lines[start_line - 1:], start_line):
            if next_line_pattern in line:
                return content_lines[start_line - 1:i]
        next_row_num += 1

    # If no next row found, take next 200 lines
    return content_lines[start_line - 1:start_line + 199]

def check_evidence(section_text):
    """Check for in-person and telehealth evidence."""
    lower_text = section_text.lower()

    # In-person patterns
    in_person_patterns = [
        r'face-to-face',
        r'\bsaw\s+(?:her|him|patient|pt|them)\s+in\s+clinic',
        r'patient\s+(?:was\s+)?in\s+clinic',
        r'came\s+(?:in\s+)?to\s+(?:the\s+)?clinic',
    ]

    # Telehealth patterns
    telehealth_patterns = [
        r'televisit',
        r'tele[\s\-]visit',
        r'video\s+visit',
        r'video\s+consult',
        r'telehealth',
        r'tele[\s\-]health',
        r'telephone\s+visit',
        r'phone\s+visit',
        r'virtual\s+visit',
    ]

    in_person_matches = []
    for pattern in in_person_patterns:
        for m in re.finditer(pattern, lower_text):
            start = max(0, m.start() - 60)
            end = min(len(section_text), m.end() + 60)
            context = section_text[start:end].replace('\n', ' ').strip()
            in_person_matches.append(context)

    telehealth_matches = []
    for pattern in telehealth_patterns:
        for m in re.finditer(pattern, lower_text):
            start = max(0, m.start() - 60)
            end = min(len(section_text), m.end() + 60)
            context = section_text[start:end].replace('\n', ' ').strip()
            telehealth_matches.append(context)

    return in_person_matches, telehealth_matches

def main():
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        content_lines = f.readlines()

    print("POST-VISIT-TYPE RULE AUDIT REPORT")
    print("=" * 80)
    print(f"\nRule: Correct 'Televisit' → 'in-person' when note contains 'face-to-face'\n")
    print("| Row | In-Person | Telehealth | Verdict | Note |")
    print("|-----|-----------|------------|---------|------|")

    results = []

    for row in TARGET_ROWS:
        if row not in ROW_POSITIONS:
            print(f"| {row} | - | - | SKIPPED | Row not in results |")
            continue

        start_line = ROW_POSITIONS[row]
        section_lines = get_row_section(content_lines, start_line, row)
        section_text = ''.join(section_lines)

        in_person_ev, telehealth_ev = check_evidence(section_text)

        in_p_count = len(in_person_ev)
        tele_count = len(telehealth_ev)

        # Determine verdict
        if in_p_count > 0 and tele_count == 0:
            verdict = "✓ CORRECT"
        elif tele_count > 0 and in_p_count == 0:
            verdict = "✗ FALSE POS"
        elif in_p_count > 0 and tele_count > 0:
            verdict = "⚠ MIXED"
        else:
            verdict = "⚠ NO EVID"

        # Note
        note = ""
        if in_person_ev:
            snippet = in_person_ev[0][:50] + "..." if len(in_person_ev[0]) > 50 else in_person_ev[0]
            note = snippet
        elif telehealth_ev:
            snippet = telehealth_ev[0][:50] + "..." if len(telehealth_ev[0]) > 50 else telehealth_ev[0]
            note = snippet

        print(f"| {row} | {in_p_count} | {tele_count} | {verdict} | {note[:40]} |")
        results.append((row, in_person_ev, telehealth_ev, verdict))

    # Detailed evidence
    print("\n\n" + "=" * 80)
    print("DETAILED EVIDENCE")
    print("=" * 80)

    for row, in_person_ev, telehealth_ev, verdict in results:
        print(f"\nRow {row} - {verdict}")
        print("-" * 80)

        if in_person_ev:
            print(f"\nIN-PERSON EVIDENCE ({len(in_person_ev)} matches):")
            for i, context in enumerate(in_person_ev[:3], 1):
                print(f"  {i}. ...{context}...")

        if telehealth_ev:
            print(f"\nTELEHEALTH EVIDENCE ({len(telehealth_ev)} matches):")
            for i, context in enumerate(telehealth_ev[:3], 1):
                print(f"  {i}. ...{context}...")

        if not in_person_ev and not telehealth_ev:
            print("\nNo clear evidence found.")

    # Summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    correct = sum(1 for _, _, _, v in results if v == "✓ CORRECT")
    false_pos = sum(1 for _, _, _, v in results if v == "✗ FALSE POS")
    mixed = sum(1 for _, _, _, v in results if v == "⚠ MIXED")
    no_evid = sum(1 for _, _, _, v in results if v == "⚠ NO EVID")
    skipped = len(TARGET_ROWS) - len(results)

    print(f"\nTotal rows audited: {len(results)}")
    print(f"✓ Correct corrections: {correct}")
    print(f"✗ False positives: {false_pos}")
    print(f"⚠ Mixed evidence: {mixed}")
    print(f"⚠ No evidence: {no_evid}")
    print(f"Skipped (not in results): {skipped}")

    accuracy = (correct / len(results) * 100) if results else 0
    print(f"\nAccuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    main()
