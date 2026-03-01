"""Parse results.txt into individual JSON files per row."""
import json
import os
import re
import sys


def parse_results(results_path, output_dir):
    with open(results_path, "r") as f:
        content = f.read()

    # Split by row headers
    row_pattern = r"={60}\nRESULTS FOR ROW (\d+)\n={60}"
    parts = re.split(row_pattern, content)

    # parts[0] = header/prompts, then alternating: row_num, row_content
    rows = {}
    for i in range(1, len(parts), 2):
        row_num = int(parts[i])
        row_content = parts[i + 1]
        rows[row_num] = row_content

    os.makedirs(output_dir, exist_ok=True)

    for row_num, content in sorted(rows.items()):
        row_data = {}

        # Extract columns
        col_pattern = r"--- Column: (\w+) ---\n"
        cols = list(re.finditer(col_pattern, content))

        for j, match in enumerate(cols):
            col_name = match.group(1)
            start = match.end()
            end = cols[j + 1].start() if j + 1 < len(cols) else len(content)
            value = content[start:end].strip()

            if col_name == "keypoints":
                # Parse the JSON block
                # Find the JSON object
                brace_start = value.find("{")
                if brace_start >= 0:
                    # Find matching closing brace
                    depth = 0
                    for k in range(brace_start, len(value)):
                        if value[k] == "{":
                            depth += 1
                        elif value[k] == "}":
                            depth -= 1
                            if depth == 0:
                                json_str = value[brace_start : k + 1]
                                break
                    try:
                        row_data[col_name] = json.loads(json_str)
                    except json.JSONDecodeError:
                        row_data[col_name] = json_str
                else:
                    row_data[col_name] = value
            else:
                # Remove surrounding quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                row_data[col_name] = value

        out_path = os.path.join(output_dir, f"row_{row_num:02d}.json")
        with open(out_path, "w") as f:
            json.dump(row_data, f, indent=2, ensure_ascii=False)
        print(f"Row {row_num} -> {out_path}")

    print(f"\nParsed {len(rows)} rows total.")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(
        base, "results", "default_20260301_084320", "results.txt"
    )
    output_dir = os.path.join(base, "results", "default_20260301_084320", "parsed")
    parse_results(results_path, output_dir)
