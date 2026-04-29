#!/usr/bin/env python3
"""
auto_review.py — Automated review of pipeline results using Qwen via vLLM.

Step 2 of the iteration workflow:
  1. Generate: python3 run.py exp/xxx.yaml
  2. Review:   python3 auto_review.py results/xxx_results.txt
  3. Fix:      Human/Claude reads review doc, edits prompts/hooks

Usage:
    python3 auto_review.py results/v32_pdac_test_results.txt
    python3 auto_review.py results/v32_pdac_test_results.txt --output results/v32_pdac_review.md
    python3 auto_review.py results/v32_pdac_test_results.txt --base-url http://localhost:8000/v1
"""

import argparse
import json
import os
import re
import sys
import time

from vllm_pipeline.vllm_client import VLLMClient


# ── Prompt templates ─────────────────────────────────────────────────────────

EXTRACTION_REVIEW_PROMPT = """You are a medical oncology quality reviewer. Your job is to review the accuracy of structured data extracted from a clinical note.

ORIGINAL CLINICAL NOTE:
{note_text}

EXTRACTED DATA (keypoints):
{keypoints_json}

IMPORTANT EXTRACTION RULES (you must understand these before reviewing):
- current_meds: ONLY cancer-related medications (chemo, targeted therapy, immunotherapy, hormonal therapy, bone agents). General medications (blood pressure, diabetes, vitamins, pain meds, anxiety meds) are CORRECTLY EXCLUDED. Do NOT flag missing non-cancer drugs.
- supportive_meds: ONLY cancer-treatment-related supportive care (anti-nausea, growth factors, cancer pain meds). General prescriptions are correctly excluded.
- If drug names are [REDACTED] in the note, empty current_meds is acceptable — do NOT flag as P1.
- goals_of_treatment: Valid values are curative, palliative, adjuvant, risk reduction, surveillance, symptom management. For patients on surveillance with no active treatment, "surveillance" is correct. "Adjuvant" is wrong for patients who completed adjuvant therapy years ago.
- Stage_of_Cancer: Only flag if the extraction INVENTS a stage not in the note. Reasonable inference from tumor data is acceptable (P2 at most).
- For patients with RESECTED prior metastasis and current NED (no evidence of disease), saying "now metastatic Stage IV" is misleading — the current status should reflect NED.

REVIEW EACH FIELD against the original note. For each field, check:
1. FAITHFUL: Is the value supported by the note? Any fabrication/hallucination?
2. COMPLETE: Did the extraction miss important CANCER-RELATED information?
3. CORRECT TEMPORAL: Are current/past/future correctly distinguished?
4. CORRECT CLASSIFICATION: For categorical fields (Patient type, goals, etc.), is the category right?

Severity calibration — be strict about these definitions:
- P0: Hallucination — value contains information NOT in the note (fabricated facts, invented numbers)
- P1: Major error — ONLY for: wrong clinical direction (curative↔palliative), factual contradiction with the note, critical cancer-related omission that changes clinical meaning. Missing non-cancer medications is NOT P1.
- P2: Minor issue — incomplete but not wrong, slightly imprecise wording, could be better, missing minor details

Output a JSON object with this structure:
{{
  "findings": [
    {{
      "field": "field_name",
      "severity": "P0|P1|P2",
      "issue": "brief description of the problem",
      "note_evidence": "quote from note that shows the issue"
    }}
  ],
  "clean_fields": ["list of fields with no issues"],
  "summary": "1-2 sentence overall assessment"
}}

If all fields are correct, return empty findings list and summary "All fields clean."
Output ONLY the JSON, no other text."""

LETTER_REVIEW_PROMPT = """You are reviewing a patient letter generated from clinical note data. The letter should be accurate, complete, and written at an 8th-grade reading level.

ORIGINAL CLINICAL NOTE (abbreviated):
{note_text}

EXTRACTED DATA used to generate the letter:
{keypoints_json}

GENERATED LETTER:
{letter_text}

IMPORTANT CONTEXT for letter review:
- The letter is written FOR PATIENTS at an 8th-grade reading level. Simplification is INTENTIONAL.
- Using "pancreatic cancer" instead of "pancreatic ductal adenocarcinoma" is acceptable simplification for follow-up patients. Only flag if it's a NEW patient who needs to understand their specific diagnosis.
- Using "neuroendocrine tumor" instead of "nonfunctioning pancreatic neuroendocrine tumor" is fine.
- The letter should use "you/your" voice, NOT "he/she/the patient" voice.
- Dosing details (mg, mg/m2) should NOT be in the letter — but if the letter says "reduced to" or "increased to" without the number, that IS a problem (incomplete sentence).
- Missing [source:] tags are NOT a problem — they are stripped in post-processing.

Review the letter for:
1. ACCURACY: Every statement must be supported by the keypoints AND note. Flag fabrication.
2. COMPLETENESS: Important clinical info should not be omitted (minor details can be skipped).
3. READABILITY: Simple language (8th-grade level). Flag unexplained jargon ONLY if it is truly incomprehensible to a layperson. These terms do NOT need explanation and should NOT be flagged: "CT scan", "MRI", "PET scan", "chemotherapy", "tumor", "cancer", "adenocarcinoma", "CA 19-9", "CEA", "biopsy", "metastatic", "lymph node", "liver", "pancreas", "stable disease", "progression", "dose reduction", "AST", "ALT", "blood tests", "white blood cell", "red blood cell", "hemoglobin", "platelet". CRITICAL: Unexplained medical jargon is ALWAYS P2, NEVER P1. Do NOT classify any jargon issue as P1 regardless of how technical the term is. P1 is reserved for factual errors, harmful content, garbled text, and wrong voice — NOT readability issues. Terms like "hepatic", "osseous", "adenocarcinoma", "Mediport", "standard of care", "fine-needle aspiration", brand drug names — all P2 at most. An "incomplete sentence" that is actually just simplified (e.g., "remaining cancer" instead of "residual adenocarcinoma") is NOT incomplete — it's acceptable simplification (P2 at most).
4. VOICE: Must use "you/your", never "he/she/the patient".
5. GARBLED TEXT: Flag any obvious text corruption, missing words, or incomplete sentences.
6. NO HARM: Nothing misleading that could cause wrong patient actions.

Severity calibration:
- P0: Letter says something factually wrong or fabricated
- P1: ONLY for: garbled/unreadable text, wrong voice ("he" instead of "you"), factually misleading statement, incomplete sentence with missing critical info (e.g., "dose reduced to" with no number)
- P2: Minor readability issue, unexplained jargon, style problem, slightly imprecise wording

Output a JSON object:
{{
  "findings": [
    {{
      "severity": "P0|P1|P2",
      "sentence": "the problematic sentence from the letter",
      "issue": "what's wrong",
      "suggestion": "how to fix it"
    }}
  ],
  "summary": "1-2 sentence overall assessment"
}}

If the letter is clean, return empty findings and summary "Letter is clean."
Output ONLY the JSON, no other text."""


# ── Parse results.txt ────────────────────────────────────────────────────────

def parse_results_file(filepath):
    """Parse results.txt into a list of sample dicts."""
    with open(filepath, "r") as f:
        content = f.read()

    # Split by ROW boundaries
    parts = re.split(r"={60}\nRESULTS FOR ROW (\d+)\n={60}", content)
    # parts[0] = header, then alternating (row_num, content)

    samples = []
    for i in range(1, len(parts), 2):
        row_num = int(parts[i])
        data = parts[i + 1]

        sample = {"row": row_num}

        # Extract note_text
        m = re.search(r"--- Column: note_text ---\n(.*?)\n\n--- Column:", data, re.DOTALL)
        if m:
            sample["note_text"] = m.group(1).strip().strip('"')

        # Extract assessment_and_plan
        m = re.search(r"--- Column: assessment_and_plan ---\n(.*?)\n\n--- Column:", data, re.DOTALL)
        if m:
            sample["assessment_and_plan"] = m.group(1).strip().strip('"')

        # Extract keypoints
        m = re.search(r"--- Column: keypoints ---\n(.*?)\n\n--- Column:", data, re.DOTALL)
        if m:
            try:
                sample["keypoints"] = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                sample["keypoints"] = {}

        # Extract letter
        m = re.search(r"--- Column: letter ---\n(.*?)\n\n--- Column:", data, re.DOTALL)
        if m:
            sample["letter"] = m.group(1).strip().strip('"').replace("\\n", "\n")

        samples.append(sample)

    return samples


# ── Review logic ─────────────────────────────────────────────────────────────

def truncate_note(note_text, max_chars=12000):
    """Truncate long notes, keeping beginning and end (A/P is usually at the end)."""
    if len(note_text) <= max_chars:
        return note_text
    half = max_chars // 2
    return note_text[:half] + "\n\n[... middle of note truncated for review ...]\n\n" + note_text[-half:]


def call_review(client, prompt, max_retries=2):
    """Call Qwen for review, parse JSON response."""
    gen_config = {"max_new_tokens": 2048, "do_sample": False}

    for attempt in range(max_retries + 1):
        try:
            response = client.chat_generate(prompt, gen_config)
            # Try to extract JSON from response
            # Strip markdown code fences if present
            cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            return json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries:
                print(f"    Retry {attempt + 1} (parse error: {e})")
                continue
            return {"findings": [], "summary": f"Review failed: {e}", "error": True}


def review_sample(client, sample):
    """Review one sample's extraction and letter."""
    note = truncate_note(sample.get("note_text", ""))
    keypoints = sample.get("keypoints", {})
    letter = sample.get("letter", "")
    kp_json = json.dumps(keypoints, indent=2, ensure_ascii=False)

    # 1. Extraction review
    ext_prompt = EXTRACTION_REVIEW_PROMPT.format(
        note_text=note,
        keypoints_json=kp_json,
    )
    print(f"    Reviewing extraction...", end=" ", flush=True)
    t0 = time.time()
    ext_review = call_review(client, ext_prompt)
    print(f"{time.time() - t0:.1f}s")

    # 2. Letter review (only if letter exists)
    letter_review = {"findings": [], "summary": "No letter to review."}
    if letter and len(letter) > 50:
        let_prompt = LETTER_REVIEW_PROMPT.format(
            note_text=note,
            keypoints_json=kp_json,
            letter_text=letter,
        )
        print(f"    Reviewing letter...", end=" ", flush=True)
        t0 = time.time()
        letter_review = call_review(client, let_prompt)
        print(f"{time.time() - t0:.1f}s")

    return ext_review, letter_review


# ── Output ───────────────────────────────────────────────────────────────────

def write_review_doc(samples, ext_reviews, letter_reviews, output_path, source_file):
    """Write structured review markdown."""
    # Count totals
    total_p0 = total_p1 = total_p2 = 0
    for er, lr in zip(ext_reviews, letter_reviews):
        for f in er.get("findings", []) + lr.get("findings", []):
            sev = f.get("severity", "")
            if sev == "P0":
                total_p0 += 1
            elif sev == "P1":
                total_p1 += 1
            elif sev == "P2":
                total_p2 += 1

    clean_count = sum(
        1 for er, lr in zip(ext_reviews, letter_reviews)
        if not er.get("findings") and not lr.get("findings")
    )

    with open(output_path, "w") as f:
        f.write(f"# Auto Review: {os.path.basename(source_file)}\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"Reviewer: Qwen2.5-32B-Instruct-AWQ (auto_review.py)\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Samples**: {len(samples)}\n")
        f.write(f"- **Clean**: {clean_count}/{len(samples)}\n")
        f.write(f"- **P0** (hallucination): {total_p0}\n")
        f.write(f"- **P1** (major error): {total_p1}\n")
        f.write(f"- **P2** (minor issue): {total_p2}\n\n")

        if total_p0 + total_p1 > 0:
            f.write("### Critical Issues\n\n")
            for sample, er, lr in zip(samples, ext_reviews, letter_reviews):
                for finding in er.get("findings", []) + lr.get("findings", []):
                    if finding.get("severity") in ("P0", "P1"):
                        f.write(f"- **ROW {sample['row']}** [{finding['severity']}]: {finding.get('issue', 'N/A')}\n")
            f.write("\n")

        f.write("---\n\n")

        # Per-sample details
        for sample, er, lr in zip(samples, ext_reviews, letter_reviews):
            row = sample["row"]
            # Quick type/stage summary from keypoints
            kp = sample.get("keypoints", {})
            cancer_type = kp.get("Cancer_Diagnosis", {}).get("Type_of_Cancer", "N/A")
            stage = kp.get("Cancer_Diagnosis", {}).get("Stage_of_Cancer", "N/A")

            ext_findings = er.get("findings", [])
            let_findings = lr.get("findings", [])
            is_clean = not ext_findings and not let_findings
            status = "✅ CLEAN" if is_clean else "⚠️ ISSUES"

            f.write(f"## ROW {row} — {status}\n\n")
            f.write(f"**Type**: {cancer_type[:80]}\n")
            f.write(f"**Stage**: {stage[:80]}\n\n")

            # Extraction findings
            if ext_findings:
                f.write("### Extraction\n\n")
                f.write("| Severity | Field | Issue | Note Evidence |\n")
                f.write("|----------|-------|-------|---------------|\n")
                for finding in ext_findings:
                    sev = finding.get("severity", "?")
                    field = finding.get("field", "?")
                    issue = finding.get("issue", "").replace("|", "\\|").replace("\n", " ")
                    evidence = finding.get("note_evidence", "").replace("|", "\\|").replace("\n", " ")[:100]
                    f.write(f"| {sev} | {field} | {issue} | {evidence} |\n")
                f.write("\n")
            else:
                f.write("### Extraction: ✅ All fields clean\n\n")

            f.write(f"*Extraction summary*: {er.get('summary', 'N/A')}\n\n")

            # Letter findings
            if let_findings:
                f.write("### Letter\n\n")
                f.write("| Severity | Issue | Sentence |\n")
                f.write("|----------|-------|----------|\n")
                for finding in let_findings:
                    sev = finding.get("severity", "?")
                    issue = finding.get("issue", "").replace("|", "\\|").replace("\n", " ")
                    sent = finding.get("sentence", "").replace("|", "\\|").replace("\n", " ")[:80]
                    f.write(f"| {sev} | {issue} | {sent} |\n")
                f.write("\n")
            else:
                f.write("### Letter: ✅ Clean\n\n")

            f.write(f"*Letter summary*: {lr.get('summary', 'N/A')}\n\n")
            f.write("---\n\n")

    print(f"\nReview doc written to: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-review pipeline results using Qwen via vLLM")
    parser.add_argument("results_file", help="Path to results.txt from pipeline run")
    parser.add_argument("--output", "-o", help="Output review doc path (default: auto)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Model name")
    args = parser.parse_args()

    # Auto-generate output path
    if not args.output:
        base = os.path.splitext(args.results_file)[0]
        args.output = base.replace("_results", "_review") + ".md"

    # Connect to vLLM
    client = VLLMClient(base_url=args.base_url, model_name=args.model)
    if not client.health_check():
        print("ERROR: vLLM server not reachable at", args.base_url)
        sys.exit(1)
    print(f"Connected to vLLM at {args.base_url}")

    # Parse results
    samples = parse_results_file(args.results_file)
    print(f"Parsed {len(samples)} samples from {args.results_file}")

    # Review each sample
    ext_reviews = []
    letter_reviews = []
    total_start = time.time()

    for i, sample in enumerate(samples):
        print(f"\nROW {sample['row']} ({i+1}/{len(samples)}):")
        er, lr = review_sample(client, sample)
        ext_reviews.append(er)
        letter_reviews.append(lr)

        # Quick status
        ext_issues = len(er.get("findings", []))
        let_issues = len(lr.get("findings", []))
        status = "CLEAN" if ext_issues + let_issues == 0 else f"ext={ext_issues} let={let_issues}"
        print(f"    → {status}")

    elapsed = time.time() - total_start
    print(f"\nTotal review time: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Write review doc
    write_review_doc(samples, ext_reviews, letter_reviews, args.output, args.results_file)


if __name__ == "__main__":
    main()
