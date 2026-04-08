#!/usr/bin/env python3
"""Find line numbers for each row in results.txt"""

TARGET_ROWS = [2, 7, 8, 9, 49, 60, 62, 63, 79, 82, 85]
RESULTS_FILE = "/Users/yizhoucc/repo/med_dict/results/v17_verify_20260318_184026/results.txt"

with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
    lines = f.readlines()

row_positions = {}

for i, line in enumerate(lines, 1):
    if line.startswith("RESULTS FOR ROW "):
        row_num = int(line.split("RESULTS FOR ROW ")[1].strip())
        row_positions[row_num] = i

print("Row number to line number mapping:")
for row in TARGET_ROWS:
    if row in row_positions:
        print(f"Row {row}: line {row_positions[row]}")
    else:
        print(f"Row {row}: NOT FOUND")

print("\nAll rows found:")
sorted_rows = sorted(row_positions.keys())
for row in sorted_rows:
    print(f"  Row {row}: line {row_positions[row]}")
