#!/usr/bin/env python3
"""Detailed audit focusing on billing template vs real evidence"""

import re
import json

RESULTS_FILE = "/Users/yizhoucc/repo/med_dict/results/v17_verify_20260318_184026/results.txt"
TARGET_ROWS = [2, 7, 8, 9, 49, 60, 62, 63, 79, 82, 85]

ROW_POSITIONS = {
    2: 601, 7: 1199, 8: 1335, 9: 1474, 49: 4613,
    63: 5743, 82: 7157, 85: 7589,
}

def get_row_section(content_lines, start_line, row_num):
    """Extract section for a specific row."""
    for i in range(start_line, min(start_line + 500, len(content_lines))):
        if i > start_line and "RESULTS FOR ROW " in content_lines[i]:
            return content_lines[start_line - 1:i]
    return content_lines[start_line - 1:start_line + 199]

def extract_visit_type_and_attribution(section_text):
    """Extract visit type from keypoints and attribution evidence."""
    # Get visit type
    match = re.search(r'"in-person":\s*"([^"]+)"', section_text)
    visit_type = match.group(1) if match else "UNKNOWN"

    # Get attribution evidence
    attr_pattern = r'"in-person":\s*\[(.*?)\]'
    attr_match = re.search(attr_pattern, section_text, re.DOTALL)
    attribution = []
    if attr_match:
        attr_content = attr_match.group(1)
        # Extract quoted strings
        attribution = re.findall(r'"([^"]+)"', attr_content)

    return visit_type, attribution

def classify_evidence(text):
    """Classify a piece of evidence."""
    lower = text.lower()

    # Billing template phrases (ambiguous)
    billing_templates = [
        r'time includes face-to-face time',
        r'this time includes face-to-face',
        r'spent.*face-to-face',
        r'minutes face-to-face with the patient',
    ]

    # Clear in-person evidence
    clear_in_person = [
        r'i performed a face-to-face encounter',
        r'shared visit.*face-to-face',
        r'saw.*in clinic',
        r'patient in clinic',
    ]

    # Telehealth evidence
    telehealth_patterns = [
        r'televisit',
        r'video visit',
        r'telehealth tools',
        r'video.*connection',
        r'zoom connection',
        r'real-time telehealth',
    ]

    for pattern in telehealth_patterns:
        if re.search(pattern, lower):
            return "TELEHEALTH"

    for pattern in clear_in_person:
        if re.search(pattern, lower):
            return "IN-PERSON (clear)"

    for pattern in billing_templates:
        if re.search(pattern, lower):
            return "BILLING TEMPLATE (ambiguous)"

    return "UNCLEAR"

def check_note_text(section_text):
    """Check note_text for telehealth indicators."""
    # Extract note_text
    pattern = r'--- Column: note_text ---\n"(.*?)"\n\n--- Column: assessment'
    match = re.search(pattern, section_text, re.DOTALL)
    if not match:
        return []

    note = match.group(1)
    lower = note.lower()

    indicators = []

    # Check for chief complaint "Video Visit"
    if re.search(r'video visit.*follow-up', lower, re.DOTALL):
        indicators.append("Chief Complaint: Video Visit")

    # Check for telehealth statement
    if re.search(r'i performed this.*using real-time telehealth tools', lower):
        indicators.append("Explicit: 'I performed this evaluation using real-time telehealth tools'")

    if re.search(r'televisit', lower):
        indicators.append("Mentions 'televisit'")

    if re.search(r'live video.*connection', lower):
        indicators.append("Mentions 'live video connection'")

    if re.search(r'zoom connection', lower):
        indicators.append("Mentions 'Zoom connection'")

    return indicators

def main():
    with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
        content_lines = f.readlines()

    print("DETAILED POST-VISIT-TYPE AUDIT")
    print("=" * 100)
    print("\n| Row | Visit Type | Attribution Evidence | Classification | Note Text Indicators |")
    print("|-----|------------|---------------------|----------------|----------------------|")

    detailed_results = []

    for row in TARGET_ROWS:
        if row not in ROW_POSITIONS:
            print(f"| {row} | SKIPPED | - | - | Row not in results |")
            continue

        start_line = ROW_POSITIONS[row]
        section_lines = get_row_section(content_lines, start_line, row)
        section_text = ''.join(section_lines)

        visit_type, attribution = extract_visit_type_and_attribution(section_text)
        telehealth_indicators = check_note_text(section_text)

        # Classify attribution evidence
        if attribution:
            evidence_class = classify_evidence(attribution[0])
            evidence_text = attribution[0][:60] + "..." if len(attribution[0]) > 60 else attribution[0]
        else:
            evidence_class = "NO ATTRIBUTION"
            evidence_text = "-"

        indicators_str = "; ".join(telehealth_indicators) if telehealth_indicators else "None"

        # Determine verdict
        if telehealth_indicators and "TEMPLATE" in evidence_class:
            verdict = "✗ FALSE POS (telehealth + billing template)"
        elif telehealth_indicators:
            verdict = "✗ FALSE POS (clear telehealth)"
        elif "clear" in evidence_class.lower():
            verdict = "✓ CORRECT"
        elif "TEMPLATE" in evidence_class:
            verdict = "⚠ AMBIGUOUS (billing template only)"
        else:
            verdict = "⚠ UNCLEAR"

        print(f"| {row} | {visit_type} | {evidence_class} | {verdict} | {indicators_str[:40]} |")
        detailed_results.append((row, visit_type, attribution, evidence_class, telehealth_indicators, verdict))

    # Detailed breakdown
    print("\n\n" + "=" * 100)
    print("DETAILED BREAKDOWN")
    print("=" * 100)

    for row, visit_type, attribution, evidence_class, telehealth_indicators, verdict in detailed_results:
        print(f"\n{'='*100}")
        print(f"Row {row} - {verdict}")
        print('='*100)

        print(f"\nVisit Type: {visit_type}")

        if attribution:
            print(f"\nAttribution Evidence ({len(attribution)} items):")
            for i, attr in enumerate(attribution[:3], 1):
                print(f"  {i}. {attr}")
                print(f"     Classification: {classify_evidence(attr)}")

        if telehealth_indicators:
            print(f"\nTelehealth Indicators in Note Text:")
            for ind in telehealth_indicators:
                print(f"  - {ind}")

    # Summary
    print("\n\n" + "=" * 100)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 100)

    correct = sum(1 for r in detailed_results if "CORRECT" in r[5])
    false_pos = sum(1 for r in detailed_results if "FALSE POS" in r[5])
    ambiguous = sum(1 for r in detailed_results if "AMBIGUOUS" in r[5])

    print(f"\nTotal rows audited: {len(detailed_results)}")
    print(f"✓ Correct: {correct}")
    print(f"✗ False Positives: {false_pos}")
    print(f"⚠ Ambiguous: {ambiguous}")

    print("\nRECOMMENDATION:")
    print("The POST-VISIT-TYPE rule has HIGH FALSE POSITIVE rate.")
    print("\nProblem: 'face-to-face' appears in billing template text even for telehealth visits.")
    print("Example: 'This time includes face-to-face time with the patient' is boilerplate.")
    print("\nSuggested fix:")
    print("1. EXCLUDE matches where the note explicitly states telehealth/video visit")
    print("2. REQUIRE 'face-to-face' to appear in a SPECIFIC context (e.g., 'I performed a face-to-face encounter')")
    print("3. IGNORE billing template phrases like 'time includes face-to-face time'")

if __name__ == "__main__":
    main()
