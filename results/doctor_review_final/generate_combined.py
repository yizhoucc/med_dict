"""Generate COMBINED_REVIEW.md and COMBINED_REVIEW.html from individual sample files.

Reads full clinical notes from CORAL CSV (not from sample files which may be truncated).
Extracts letters from each condition's sample files.

Usage:
    python3 generate_combined.py
"""
import csv, os, re, markdown

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', '..', 'data', 'CORAL',
    'coral-expert-curated-medical-oncology-reports-to-advance-language-model-inference-1.0',
    'coral', 'annotated')

BREAST_CSV = os.path.join(DATA, 'breastca_annotated.csv')
PDAC_CSV = os.path.join(DATA, 'pdac_annotated.csv')

def read_file(path):
    with open(path) as f:
        return f.read()

def load_csv_notes(csv_path):
    with open(csv_path) as f:
        return [r['note_text'] for r in csv.DictReader(f)]

def extract_letter(content):
    match = re.search(r'## Patient Letter[^\n]*\n', content)
    if not match:
        return ""
    start = match.end()
    end_match = re.search(r'\n---\s*\n', content[start:])
    if end_match:
        return content[start:start + end_match.start()].strip()
    rating_idx = content.find("## Rating", start)
    if rating_idx != -1:
        return content[start:rating_idx].strip()
    return content[start:].strip()

def extract_cancer_type(content):
    match = re.search(r'\*\*Cancer Type:\*\*\s*(.*)', content)
    return match.group(1).strip() if match else ""

RATING_TEMPLATE = """### Rating

| Dimension | Score (1-5) | Comments |
|-----------|------------|----------|
| Accurate |  |  |
| Hallucination-free |  |  |
| Comprehensible |  |  |
| Concise |  |  |
| Useful |  |  |

**Fabricated info?** Yes / No

**Missing critical info?** Yes / No

**Harmful content?** Yes / No

**Comments:**

"""

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Patient Letter Evaluation — Combined Review</title>
<style>
  body {{ font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11pt;
         line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 40px; color: #222; }}
  h1 {{ font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 8px; margin-top: 50px; }}
  h2 {{ font-size: 15pt; margin-top: 30px; color: #333; }}
  h3 {{ font-size: 12pt; margin-top: 20px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0; }}
  th, td {{ border: 1px solid #999; padding: 8px 12px; text-align: left; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  pre, code {{ font-family: 'Courier New', monospace; font-size: 9pt; }}
  pre {{ background: #f8f8f8; padding: 14px; border: 1px solid #ddd;
         white-space: pre-wrap; word-wrap: break-word; overflow-wrap: break-word; }}
  hr {{ border: none; border-top: 1px solid #ccc; margin: 24px 0; }}
  strong {{ color: #111; }}
  @media print {{
    body {{ font-size: 9pt; padding: 0; }}
    h1 {{ page-break-before: always; font-size: 16pt; }}
    h1:first-of-type {{ page-break-before: avoid; }}
    pre {{ font-size: 7.5pt; }}
    table {{ font-size: 9pt; }}
  }}
</style>
</head><body>
{body}
</body></html>"""


def build_markdown():
    breast_notes = load_csv_notes(BREAST_CSV)
    pdac_notes = load_csv_notes(PDAC_CSV)
    output = []

    # Rubric
    rubric = read_file(os.path.join(BASE, 'EVALUATION_RUBRIC.md'))
    output.append(rubric.rstrip())
    output.append("\n\n---\n\n")

    # Breast cancer (3 systems)
    output.append("# Part I: Breast Cancer (20 Samples × 3 Systems)\n\n")
    for i in range(20):
        num = f"{i+1:02d}"
        pipeline = read_file(os.path.join(BASE, 'breast_pipeline', f'sample_{num}.md'))
        baseline = read_file(os.path.join(BASE, 'breast_baseline', f'sample_{num}.md'))
        chatgpt = read_file(os.path.join(BASE, 'breast_chatgptbaseline', f'sample_{num}.md'))

        cancer_type = extract_cancer_type(pipeline)
        full_note = breast_notes[i]

        output.append(f"# Breast Cancer — Sample {i+1}\n\n")
        if cancer_type:
            output.append(f"**Cancer Type:** {cancer_type}\n\n")
        output.append("## Original Clinical Note\n\n```\n" + full_note + "\n```\n\n")

        for label, letter_text in [("A", extract_letter(chatgpt)),
                                    ("B", extract_letter(baseline)),
                                    ("C", extract_letter(pipeline))]:
            output.append(f"---\n\n## Letter {label}\n\n")
            output.append(letter_text + "\n\n")
            output.append(RATING_TEMPLATE)
        output.append("---\n\n")

    # PDAC (2 systems)
    output.append("# Part II: Pancreatic Cancer (20 Samples × 2 Systems)\n\n")
    for i in range(20):
        num = f"{i+1:02d}"
        pipeline = read_file(os.path.join(BASE, 'pdac_pipeline', f'sample_{num}.md'))
        baseline = read_file(os.path.join(BASE, 'pdac_baseline', f'sample_{num}.md'))

        cancer_type = extract_cancer_type(pipeline)
        full_note = pdac_notes[i]

        output.append(f"# Pancreatic Cancer — Sample {i+1}\n\n")
        if cancer_type:
            output.append(f"**Cancer Type:** {cancer_type}\n\n")
        output.append("## Original Clinical Note\n\n```\n" + full_note + "\n```\n\n")

        for label, letter_text in [("A", extract_letter(baseline)),
                                    ("B", extract_letter(pipeline))]:
            output.append(f"---\n\n## Letter {label}\n\n")
            output.append(letter_text + "\n\n")
            output.append(RATING_TEMPLATE)
        output.append("---\n\n")

    return "".join(output)


def main():
    md_text = build_markdown()

    # Write .md
    md_path = os.path.join(BASE, 'COMBINED_REVIEW.md')
    with open(md_path, 'w') as f:
        f.write(md_text)
    print(f"MD:   {md_path} ({len(md_text)//1024} KB)")

    # Write .html
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    html_full = HTML_TEMPLATE.format(body=html_body)
    html_path = os.path.join(BASE, 'COMBINED_REVIEW.html')
    with open(html_path, 'w') as f:
        f.write(html_full)
    print(f"HTML: {html_path} ({len(html_full)//1024} KB)")


if __name__ == '__main__':
    main()
