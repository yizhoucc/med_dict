#!/usr/bin/env python3
"""
baseline_generate.py — Generate patient letters with NO pipeline, NO gates, NO hooks.
Just raw note → single prompt → letter. For baseline comparison in the research proposal.

Usage:
    python3 baseline_generate.py data/CORAL/.../breastca_annotated.csv --output results/baseline_breast.txt
    python3 baseline_generate.py data/CORAL/.../pdac_unannotated.csv --output results/baseline_pdac.txt --indices 0,1,2,...
"""

import argparse
import csv
import json
import os
import sys
import time

from vllm_pipeline.vllm_client import VLLMClient

BASELINE_PROMPT = """You are a medical communication specialist at a cancer center.
Your role is to translate complex oncology clinical notes into
clear, compassionate summary letters that patients can understand.

Read the following oncology clinical note and write a patient-friendly
summary letter.

Requirements:
- Write at or below an 8th-grade reading level. Use short sentences
  and common words.
- When a medical term must be used, immediately explain it in plain
  language.
- Include: diagnosis and stage, treatment plan, key test results,
  next steps.
- Do NOT add information not present in the original note.
- Do NOT provide specific medication dosages.
- Do NOT speculate about prognosis unless stated in the note.
- Remind the patient to discuss questions with their care team.
- Length: 250-350 words.
- Tone: warm, respectful, empowering.

Clinical Note:
{note_text}

Write the patient letter now."""


def main():
    parser = argparse.ArgumentParser(description="Generate baseline letters (no pipeline)")
    parser.add_argument("csv_path", help="Path to CSV with note_text column")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--indices", help="Comma-separated row indices (default: all)")
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    args = parser.parse_args()

    # Connect to vLLM
    client = VLLMClient(base_url=args.base_url, model_name=args.model)
    if not client.health_check():
        print("ERROR: vLLM server not reachable")
        sys.exit(1)
    print(f"Connected to vLLM at {args.base_url}")

    # Load data
    with open(args.csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    note_col = header.index('note_text')
    coral_col = header.index('coral_idx') if 'coral_idx' in header else 0

    if args.indices:
        indices = [int(x) for x in args.indices.split(',')]
    else:
        indices = list(range(len(rows)))

    print(f"Loaded {len(rows)} rows, processing {len(indices)}")

    # Generate
    gen_config = {"max_new_tokens": 1024, "do_sample": False}
    results = []
    total_start = time.time()

    with open(args.output, 'w') as out:
        out.write(f"BASELINE GENERATION — No pipeline, single prompt\n")
        out.write(f"Model: {args.model}\n")
        out.write(f"Source: {args.csv_path}\n")
        out.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
        out.write(f"Prompt: Standard baseline (see research proposal Section 5.3)\n")
        out.write(f"{'='*60}\n\n")

        for idx in indices:
            row = rows[idx]
            note_text = row[note_col]
            coral_id = row[coral_col] if coral_col < len(row) else str(idx)

            print(f"ROW {idx+1} (coral={coral_id}, {len(note_text)} chars)...", end=" ", flush=True)
            t0 = time.time()

            prompt = BASELINE_PROMPT.format(note_text=note_text)
            letter = client.chat_generate(prompt, gen_config)

            elapsed = time.time() - t0
            print(f"{elapsed:.1f}s, {len(letter)} chars")

            out.write(f"{'='*60}\n")
            out.write(f"ROW {idx+1} (coral_idx={coral_id})\n")
            out.write(f"{'='*60}\n\n")
            out.write(f"--- LETTER ---\n")
            out.write(letter)
            out.write(f"\n\n--- STATS ---\n")
            out.write(f"Generation time: {elapsed:.1f}s\n")
            out.write(f"Letter length: {len(letter)} chars\n")
            out.write(f"\n\n")

    total = time.time() - total_start
    print(f"\nDone! {len(indices)} letters in {total:.0f}s ({total/60:.1f}min)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
