#!/bin/bash
# Audit POST-VISIT-TYPE corrections

RESULT_FILE="/Users/yizhoucc/repo/med_dict/results/v17_verify_20260318_184026/results.txt"

# Get line numbers for each row
declare -A ROW_LINES=(
  [2]=601
  [7]=1199
  [8]=1335
  [9]=1474
  [49]=5730
  [60]=7262
  [62]=7584
  [63]=7730
  [79]=9710
  [82]=10049
  [85]=10386
)

# Find next row for each
declare -A ROW_END=(
  [2]=764
  [7]=1335
  [8]=1474
  [9]=1607
  [49]=5886
  [60]=7418
  [62]=7730
  [63]=7876
  [79]=9863
  [82]=10195
  [85]=10531
)

echo "| Row | Face-to-Face | Televisit/Video | Verdict | Key Evidence |"
echo "|-----|--------------|-----------------|---------|--------------|"

for row in 2 7 8 9 49 60 62 63 79 82 85; do
    start=${ROW_LINES[$row]}
    end=${ROW_END[$row]}

    if [ -z "$start" ]; then
        echo "| $row | - | - | ERROR | Row line not found |"
        continue
    fi

    # Extract the section
    section=$(sed -n "${start},${end}p" "$RESULT_FILE")

    # Check for face-to-face evidence
    f2f_count=$(echo "$section" | grep -i "face-to-face" | wc -l | tr -d ' ')

    # Check for televisit evidence
    tele_count=$(echo "$section" | grep -iE "televisit|tele-visit|video.*visit|telehealth|tele-health|telephone.*visit|phone.*visit|virtual.*visit" | wc -l | tr -d ' ')

    # Get first evidence
    evidence=""
    if [ "$f2f_count" -gt 0 ]; then
        evidence=$(echo "$section" | grep -i "face-to-face" -m 1 -o -C 20 | tr '\n' ' ' | cut -c 1-50)
    elif [ "$tele_count" -gt 0 ]; then
        evidence=$(echo "$section" | grep -iE "televisit|video.*visit" -m 1 -o -C 20 | tr '\n' ' ' | cut -c 1-50)
    fi

    # Determine verdict
    if [ "$f2f_count" -gt 0 ] && [ "$tele_count" -eq 0 ]; then
        verdict="✓ CORRECT"
    elif [ "$tele_count" -gt 0 ] && [ "$f2f_count" -eq 0 ]; then
        verdict="✗ FALSE POS"
    elif [ "$f2f_count" -gt 0 ] && [ "$tele_count" -gt 0 ]; then
        verdict="⚠ MIXED"
    else
        verdict="⚠ NO EVID"
    fi

    echo "| $row | $f2f_count | $tele_count | $verdict | ${evidence:0:40}... |"
done
