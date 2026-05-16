"""Generate review HTML docs from individual sample files.

Outputs:
  BREAST_REVIEW.html — 20 breast samples × 3 systems (ChatGPT / Baseline / Pipeline)
  PDAC_REVIEW.html   — 20 PDAC samples × 3 systems (ChatGPT / Baseline / Pipeline)

Reads full clinical notes from CORAL CSV (not from sample files which may be truncated).

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

HTML_TEMPLATE = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>{title}</title>
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


def build_breast_md():
    breast_notes = load_csv_notes(BREAST_CSV)
    output = []

    # Rubric
    rubric = read_file(os.path.join(BASE, 'EVALUATION_RUBRIC.md'))
    output.append(rubric.rstrip())
    output.append("\n\n---\n\n")

    for i in range(20):
        num = f"{i+1:02d}"
        pipeline = read_file(os.path.join(BASE, 'breast_pipeline', f'sample_{num}.md'))
        baseline = read_file(os.path.join(BASE, 'breast_baseline', f'sample_{num}.md'))
        chatgpt = read_file(os.path.join(BASE, 'breast_chatgptbaseline', f'sample_{num}.md'))

        full_note = breast_notes[i]

        output.append(f"# Breast Cancer — Sample {i+1}\n\n")
        output.append("## Original Clinical Note\n\n```\n" + full_note + "\n```\n\n")

        for label, letter_text in [("A", extract_letter(chatgpt)),
                                    ("B", extract_letter(baseline)),
                                    ("C", extract_letter(pipeline))]:
            output.append(f"---\n\n## Letter {label}\n\n")
            output.append(letter_text + "\n\n")
        output.append("---\n\n")

    return "".join(output)


def build_pdac_md():
    pdac_notes = load_csv_notes(PDAC_CSV)
    output = []

    # Rubric
    rubric = read_file(os.path.join(BASE, 'EVALUATION_RUBRIC.md'))
    output.append(rubric.rstrip())
    output.append("\n\n---\n\n")

    for i in range(20):
        num = f"{i+1:02d}"
        pipeline = read_file(os.path.join(BASE, 'pdac_pipeline', f'sample_{num}.md'))
        baseline = read_file(os.path.join(BASE, 'pdac_baseline', f'sample_{num}.md'))
        chatgpt = read_file(os.path.join(BASE, 'pdac_chatgptbaseline', f'sample_{num}.md'))

        full_note = pdac_notes[i]

        output.append(f"# Pancreatic Cancer — Sample {i+1}\n\n")
        output.append("## Original Clinical Note\n\n```\n" + full_note + "\n```\n\n")

        for label, letter_text in [("A", extract_letter(chatgpt)),
                                    ("B", extract_letter(baseline)),
                                    ("C", extract_letter(pipeline))]:
            output.append(f"---\n\n## Letter {label}\n\n")
            output.append(letter_text + "\n\n")
        output.append("---\n\n")

    return "".join(output)


def write_html(md_text, html_path, title):
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code'])
    html_full = HTML_TEMPLATE.format(title=title, body=html_body)
    with open(html_path, 'w') as f:
        f.write(html_full)
    return len(html_full)


def main():
    breast_md = build_breast_md()
    pdac_md = build_pdac_md()

    breast_html_path = os.path.join(BASE, 'BREAST_REVIEW.html')
    pdac_html_path = os.path.join(BASE, 'PDAC_REVIEW.html')

    sz1 = write_html(breast_md, breast_html_path,
                     "Breast Cancer — Patient Letter Review (20 Samples × 3 Systems)")
    sz2 = write_html(pdac_md, pdac_html_path,
                     "Pancreatic Cancer — Patient Letter Review (20 Samples × 3 Systems)")

    print(f"Breast: {breast_html_path} ({sz1//1024} KB)")
    print(f"PDAC:   {pdac_html_path} ({sz2//1024} KB)")


if __name__ == '__main__':
    main()
