#!/usr/bin/env python3
"""
Experiment runner for medical note extraction pipeline.

Usage:
    python run.py exp/default.yaml
    python run.py exp/default.yaml --resume results/default_20260228_103000/
"""

import argparse
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from datetime import datetime

import torch
import yaml
import pandas as pd

from transformers import BitsAndBytesConfig


def post_fix_letter(letter):
    """Apply POST letter fixes: voice, dose gaps, medication test. Returns (fixed_letter, changed)."""
    changed = False
    # POST-LETTER-FIX: replace "medication test(ing)" with "a test"
    if re.search(r'medication\s+test(ing)?', letter, re.IGNORECASE):
        letter = re.sub(r'(a\s+)?medication\s+test(ing)?', 'a test', letter, flags=re.IGNORECASE)
        print(f"  [POST-LETTER-FIX] Replaced 'medication test' with 'a test'")
        changed = True
    # POST-LETTER-VOICE: Fix third-person voice ("He/She/The patient" → "You")
    # NOTE: GRAMMAR runs AFTER VOICE because VOICE can create bad grammar
    # (e.g., "The patient was discussed" → "You was discussed")
    voice_fixes = [
        (r'\bHe responded\b', 'You responded'),
        (r'\bShe responded\b', 'You responded'),
        (r'\bHe tolerated\b', 'You tolerated'),
        (r'\bShe tolerated\b', 'You tolerated'),
        (r'\bHe is currently\b', 'You are currently'),
        (r'\bShe is currently\b', 'You are currently'),
        (r'\bHe was started\b', 'You were started'),
        (r'\bShe was started\b', 'You were started'),
        (r'\bHe will\b', 'You will'),
        (r'\bShe will\b', 'You will'),
        (r'\bHe has\b', 'You have'),
        (r'\bShe has\b', 'You have'),
        (r'\bHe had\b', 'You had'),
        (r'\bShe had\b', 'You had'),
        (r'\bI am concerned about exposing him\b', 'we are careful about exposing you'),
        (r'\bI am concerned about exposing her\b', 'we are careful about exposing you'),
        (r'\bhis residual\b', 'your residual'),
        (r'\bher residual\b', 'your residual'),
        (r'\bhis neuropathy\b', 'your neuropathy'),
        (r'\bher neuropathy\b', 'your neuropathy'),
        (r'\bhis treatment\b', 'your treatment'),
        (r'\bher treatment\b', 'your treatment'),
        (r'\bhis disease\b', 'your disease'),
        (r'\bher disease\b', 'your disease'),
        (r'\bhis condition\b', 'your condition'),
        (r'\bher condition\b', 'your condition'),
        (r'\bThe patient\b', 'You'),
        (r'\bthe patient\b', 'you'),
    ]
    for pattern, replacement in voice_fixes:
        if re.search(pattern, letter):
            letter = re.sub(pattern, replacement, letter)
            changed = True
    if changed:
        print(f"  [POST-LETTER-VOICE] Fixed third-person voice → second-person")
    # POST-LETTER-GRAMMAR: Fix grammar errors created by VOICE fix
    # "The patient was discussed" → VOICE → "You was discussed" → GRAMMAR → "You were discussed"
    old_gram = letter
    letter = re.sub(r'\bYou has\b', 'You have', letter)
    letter = re.sub(r'\byou has\b', 'you have', letter)
    letter = re.sub(r'\bYou was\b', 'You were', letter)
    letter = re.sub(r'\byou was\b', 'you were', letter)
    letter = re.sub(r'\bYou is\b', 'You are', letter)
    letter = re.sub(r'\byou is\b', 'you are', letter)
    if letter != old_gram:
        print(f"  [POST-LETTER-GRAMMAR] Fixed grammar errors")
        changed = True

    # POST-LETTER-DOSE-GAP: Fix incomplete dose sentences and chemo name artifacts
    dose_gap_patterns = [
        (r'was reduced\s+\.', 'was reduced.'),
        (r'reduced to\s*\.', 'reduced.'),
        (r'reduced to\s+for\b', 'reduced for'),
        (r'increased to\s+for\b', 'increased for'),
        (r'(?:increased?|patch)\s+to\s+and\b', 'and'),
        (r'increased?\s+to\s+and\b', 'increased, and'),
        (r'dose of your fentanyl\s+and\s+added', 'dose of your pain patch, and added'),
        (r'taking\s+(\w+)\s+\.', r'taking \1.'),
        (r'continue\s+(\w+)\s+\.', r'continue \1.'),
        (r'(?:reduced|increased)\s+dose of\s*\.', 'adjusted dose.'),
        (r'was dose-reduced\s+\.', 'was dose-reduced.'),
        (r'was dose-reduced\s*(?=\.\s|,\s)', 'was dose-reduced'),
        # Chemo regimen replacement artifacts
        (r'a chemotherapy regimen\s+regimen', 'a chemotherapy regimen'),
        (r'modified a chemotherapy regimen', 'a modified chemotherapy regimen'),
        (r'dose-modified a (?:chemotherapy )?(?:medication|regimen)', 'dose-modified chemotherapy'),
        (r'5-a medication bolus', 'one of the chemotherapy drugs'),
        (r'chemotherapy with a chemotherapy combination', 'chemotherapy'),
        (r'SOC\)?\s+chemotherapy with a chemotherapy', 'SOC) chemotherapy with a'),
        (r'\bSOC\b', 'standard'),
        (r'Bolus 5FU', 'one of the chemotherapy drugs'),
        # Collapse multiple "a medication" repetitions
        (r'a medication/a medication', 'your medications'),
        (r'a medication\s*/\s*a medication', 'your medications'),
        (r'between a medication/a medication', 'between treatment cycles'),
        (r'such as a medication\b', 'such as an alternative treatment'),
        # Doctor language leaks
        (r'regardless of the actual origin of (?:her|his|your|the) malignancy',
         'while we gather more information about your diagnosis'),
        # Collapse multiple "a medication" — 3+ in one sentence is unreadable
        # (handled below as a separate pass)
        # REDACTED garble fixes — [REDACTED] replaced with "a medication" creates bad compounds
        (r'a medication-paclitaxel', 'nab-paclitaxel (a chemotherapy drug)'),
        (r'unspecified agent-paclitaxel', 'nab-paclitaxel (a chemotherapy drug)'),
        (r'a medication-9', 'CA 19-9 (a tumor marker)'),
        (r'a medication-a medication', 'a treatment combination'),
        (r'medication-9 level', 'CA 19-9 level'),
    ]
    for pattern, replacement in dose_gap_patterns:
        old_letter = letter
        letter = re.sub(pattern, replacement, letter)
        if letter != old_letter:
            print(f"  [POST-LETTER-DOSE-GAP] Fixed incomplete dose sentence")
            changed = True
    # POST-LETTER-EMPTY-PLAN: If "What is the plan" section is empty, move content from treatment section
    plan_match = re.search(r'\*\*What is the plan going forward\?\*\*\s*\n(.*?)(?:Thank you|We understand)', letter, re.DOTALL)
    if plan_match:
        plan_content = plan_match.group(1).strip()
        if not plan_content or len(plan_content) < 10:
            # Plan is empty — check if treatment section has plan content
            treat_match = re.search(r'\*\*What treatment.*?\*\*\s*\n(.*?)\*\*What is the plan', letter, re.DOTALL)
            if treat_match:
                treat_content = treat_match.group(1).strip()
                if treat_content and len(treat_content) > 20:
                    # Copy last sentence(s) from treatment to plan as next steps
                    print(f"  [POST-LETTER-EMPTY-PLAN] Plan section was empty — added 'Please discuss next steps with your care team.'")
                    letter = letter.replace(
                        '**What is the plan going forward?**\n',
                        '**What is the plan going forward?**\nPlease discuss your next steps and treatment plan with your care team at your next visit.\n'
                    )
                    changed = True
    # POST-LETTER-MEDICATION-COLLAPSE: If "a medication" appears 3+ times, simplify
    med_count = len(re.findall(r'\ba medication\b', letter, re.IGNORECASE))
    if med_count >= 3:
        # Replace from 3rd occurrence onward with "your treatment"
        # Keep first 2 "a medication", replace rest
        def replace_nth(match, state={'count': 0}):
            state['count'] += 1
            if state['count'] <= 2:
                return match.group(0)
            return 'your treatment'
        state = {'count': 0}
        def replacer(m):
            state['count'] += 1
            return m.group(0) if state['count'] <= 2 else 'your treatment'
        letter = re.sub(r'\ba medication\b', replacer, letter, flags=re.IGNORECASE)
        print(f"  [POST-LETTER-MEDICATION-COLLAPSE] Collapsed {med_count} 'a medication' instances")
        changed = True
    return letter, changed


class LogTee(io.TextIOBase):
    """Tee stdout/stderr to both terminal and a log file.

    Filters out carriage-return progress bar lines so the log file
    only contains the final state of each progress bar, not every tick.
    """

    def __init__(self, log_path, original_stream):
        self._original = original_stream
        self._log_file = open(log_path, "a", encoding="utf-8")
        self._cr_buffer = ""  # buffer for \r-based progress lines

    def write(self, text):
        # Always write to terminal as-is
        self._original.write(text)

        # For the log file, filter progress bar updates:
        # Lines using \r without \n are progress bar ticks - buffer them
        # and only flush when a \n arrives (final state).
        if "\r" in text and "\n" not in text:
            # This is a progress bar update - just buffer the latest
            self._cr_buffer = text.rstrip("\r").split("\r")[-1]
            return len(text)

        if self._cr_buffer:
            # A \n arrived after \r-buffered content: write final state
            final_line = self._cr_buffer
            self._cr_buffer = ""
            # If text starts with \r, the new text replaces the buffer
            clean = text.lstrip("\r")
            if clean.strip():
                self._log_file.write(clean)
            else:
                self._log_file.write(final_line + "\n")
        else:
            self._log_file.write(text)

        self._log_file.flush()
        return len(text)

    def flush(self):
        self._original.flush()
        self._log_file.flush()

    def close(self):
        if self._cr_buffer:
            self._log_file.write(self._cr_buffer + "\n")
            self._cr_buffer = ""
        self._log_file.close()

    def fileno(self):
        return self._original.fileno()

    @property
    def encoding(self):
        return self._original.encoding

    def isatty(self):
        return self._original.isatty()


def setup_logging(run_dir):
    """Redirect stdout and stderr to both terminal and run_dir/run.log."""
    log_path = os.path.join(run_dir, "run.log")
    sys.stdout = LogTee(log_path, sys.__stdout__)
    sys.stderr = LogTee(log_path, sys.__stderr__)

from ult import (
    AutoModelForCausalLM,
    AutoTokenizer,
    ChatTemplate,
    build_base_cache,
    extract_and_verify,
    extract_and_verify_v2,
    load_medical_dictionary,
    load_oncology_whitelist,
    load_supportive_whitelist,
    load_lab_whitelist,
    load_genetic_tests,
    find_relevant_definitions,
    format_definitions_context,
    run_model,
    extract_with_tools,
    parse_tool_calls,
    execute_tool_calls,
    gc,
)
from source_attribution import attribute_row, get_attributable_fields
from letter_generation import generate_tagged_letter, parse_tagged_letter, post_check_letter, flesch_kincaid_grade, verify_letter_faithfulness


def load_config(yaml_path):
    """Load experiment config and resolve prompt files."""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Load prompt YAML files
    base_dir = os.path.dirname(yaml_path) or "."
    # Resolve prompt paths relative to project root (where run.py lives)
    project_root = os.path.dirname(os.path.abspath(__file__))

    prompt_files = config.get("prompts", {})
    config["_prompts"] = {}
    for key, prompt_path in prompt_files.items():
        # Try relative to project root first, then relative to yaml
        full_path = os.path.join(project_root, prompt_path)
        if not os.path.exists(full_path):
            full_path = os.path.join(base_dir, prompt_path)
        with open(full_path, "r") as f:
            config["_prompts"][key] = yaml.safe_load(f)

    return config


def config_hash(config):
    """Compute a hash of the config for checkpoint matching."""
    # Hash the serialized config (excluding runtime fields)
    hashable = {k: v for k, v in config.items() if not k.startswith("_")}
    content = json.dumps(hashable, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(config, results_base="results"):
    """Create a timestamped run directory."""
    name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(results_base, f"{name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config_snapshot(config, run_dir):
    """Save full config (with expanded prompts) to run directory."""
    snapshot = {}
    for k, v in config.items():
        if k == "_prompts":
            snapshot["prompts_content"] = v
        else:
            snapshot[k] = v
    snapshot["config_hash"] = config_hash(config)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False, allow_unicode=True)


def load_progress(run_dir):
    """Load progress from a previous run."""
    progress_path = os.path.join(run_dir, "progress.json")
    if os.path.exists(progress_path):
        with open(progress_path, "r") as f:
            return json.load(f)
    return None


def save_progress(run_dir, progress):
    """Save progress to disk."""
    progress_path = os.path.join(run_dir, "progress.json")
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def find_resumable_run(config, results_base="results"):
    """Search for an incomplete run with matching config hash."""
    chash = config_hash(config)
    if not os.path.exists(results_base):
        return None

    for entry in sorted(os.listdir(results_base), reverse=True):
        run_dir = os.path.join(results_base, entry)
        if not os.path.isdir(run_dir):
            continue
        progress = load_progress(run_dir)
        if progress and progress.get("config_hash") == chash:
            if not progress.get("completed", False):
                return run_dir
    return None


def write_results_header(results_path, config, run_time):
    """Write the header of results.txt matching notebook format."""
    source_name = os.path.basename(
        config.get("_yaml_path", "unknown.yaml")
    )
    with open(results_path, "w") as f:
        f.write(f"Source: {source_name} | Run started: {run_time}\n")
        f.write("=" * 60 + "\n")
        f.write("extraction_prompts\n")
        original_text = (
            str(config["_prompts"]["extraction"])
            .replace("\\n", "\n")
            .replace("\\'", "'")
            .replace('\\"', '"')
        )
        f.write(original_text + "\n")
        f.write("=" * 60 + "\n")

        f.write("plan_extraction_prompts\n")
        original_text = (
            str(config["_prompts"]["plan_extraction"])
            .replace("\\n", "\n")
            .replace("\\'", "'")
            .replace('\\"', '"')
        )
        f.write(original_text + "\n")
        f.write("=" * 60 + "\n")
        f.write("\n" * 5)


def rebuild_results_from_progress(results_path, progress, config, run_time):
    """Rebuild results.txt from progress.json data."""
    write_results_header(results_path, config, run_time)
    results = progress.get("results", {})
    # Write in order of completed indices
    for idx in sorted(progress.get("completed_indices", []), key=int):
        idx_str = str(idx)
        if idx_str in results:
            append_row_result(results_path, int(idx), results[idx_str])


def append_row_result(results_path, row_index, row_result):
    """Append a single row result to results.txt."""
    with open(results_path, "a") as f:
        f.write("\n" * 5)
        f.write(f"\n{'=' * 60}\n")
        f.write(f"RESULTS FOR ROW {row_index + 1}\n")
        f.write(f"{'=' * 60}\n")

        for col, value in row_result.items():
            f.write(f"\n--- Column: {col} ---\n")
            try:
                f.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            except (TypeError, ValueError):
                f.write(str(value) + "\n")


def regex_extract_assessment_plan(note_text):
    """Try to extract Assessment/Plan section using regex.

    Tested against CORAL breast cancer (89/100) and PDAC (96/100) datasets.
    Returns the extracted text or None if no match found.
    """
    # Patterns ordered from most specific to least specific.
    # Each covers real variations found in CORAL data.
    patterns = [
        # Combined A/P: "Assessment / Plan:", "Assessment and Plan:",
        # "ASSESSMENT & PLAN", "Assessment \Plan :", "ASSESSMENT/PLAN:"
        # Negative lookbehind prevents matching "of assessment and plan" in signatures
        r'(?<!of )(?:Assessment|ASSESSMENT)\s*(?:/|and|&|\\)\s*(?:Plan|PLAN|Recommendations|RECOMMENDATIONS)\s*:?',
        # Impression/Plan style: "Impression/Plan:", "Impression / Plan:"
        r'Impression\s*/\s*Plan\s*:',
        # PDAC style: "Impression and Recommendations:"
        r'Impression\s+and\s+Recommendations\s*:',
        # Separate sections: "Assessment:" (Plan: follows later in text)
        r'(?:Assessment|ASSESSMENT)\s*:',
        # Standalone "PLAN" header (preceded by whitespace, followed by uppercase)
        r'(?<=\s{3})PLAN\s+(?=[A-Z])',
    ]

    for pattern in patterns:
        flags = re.IGNORECASE if 'PLAN' not in pattern.split('(?=')[0] else 0
        match = re.search(pattern, note_text, re.IGNORECASE)
        if match:
            extracted = note_text[match.start():]
            if len(extracted) > 20:
                print(f"  Regex matched: '{match.group().strip()}'")
                return extracted

    return None


def truncate_repeated_text(text, min_block_size=100):
    """Detect and remove repeated blocks in model output.

    The model sometimes repeats the A/P text multiple times.
    Keep only the first occurrence.
    """
    if len(text) < min_block_size * 2:
        return text

    # Try to find a repeated block by checking if the first chunk
    # appears again later in the text
    for block_size in range(min_block_size, len(text) // 2 + 1, 50):
        first_block = text[:block_size]
        # Look for this block appearing again
        second_pos = text.find(first_block, block_size)
        if second_pos != -1:
            # Found repetition - keep everything up to the repeat
            return text[:second_pos].rstrip()

    return text


def extract_assessment_plan(note_text, model, tokenizer, config, chat_tmpl=None):
    """Extract assessment/plan section: regex first, LLM fallback with chat template."""
    if chat_tmpl is None:
        chat_tmpl = ChatTemplate("llama3")

    # --- Step A: Try regex extraction first ---
    print("  Trying regex extraction...")
    regex_result = regex_extract_assessment_plan(note_text)
    if regex_result is not None:
        print(f"  Regex extraction succeeded ({len(regex_result)} chars)")
        return regex_result

    print("  Regex failed, falling back to LLM extraction...")

    # --- Step B: LLM extraction with proper chat template ---
    max_retries = config.get("extraction", {}).get("max_retries", 3)
    gen_config_greedy = config["generation"]["assessment_plan"].copy()
    gen_config_greedy["eos_token_id"] = tokenizer.eos_token_id
    gen_config_retry = config["generation"]["retry"].copy()
    gen_config_retry["eos_token_id"] = tokenizer.eos_token_id

    for attempt in range(max_retries):
        if attempt == 0:
            current_prompt = chat_tmpl.system_user_assistant(
                "You are a medical text extraction tool. You extract sections from medical notes exactly as written. "
                "Return ONLY the extracted text, nothing else. No commentary, no 'here is the text', no repetition.",
                f"Here is a medical note:\n\n{note_text}\n\n"
                f"Extract the 'Assessment and Plan' or 'Assessment/Plan' section. "
                f"Return ONLY the original text from that section to the end of the note. "
                f"Do not modify, rephrase, or summarize. Do not repeat the text."
            )
            current_config = gen_config_greedy
        else:
            current_prompt = chat_tmpl.system_user_assistant(
                "You are a copy-paste tool. Return ONLY the requested section, exactly as written. "
                "No commentary. No repetition. Stop after the section ends.",
                f"SOURCE TEXT:\n{note_text}\n\n"
                f"Copy-paste the Assessment and Plan section exactly as written. "
                f"Do not summarize. Do not fix grammar. Do not repeat."
            )
            current_config = gen_config_retry

        print(f"  LLM Extraction Attempt {attempt + 1}...")
        candidate_text, _ = run_model(
            current_prompt, model, tokenizer, current_config
        )

        # --- Step C: Truncate repeated text ---
        candidate_text = truncate_repeated_text(candidate_text)

        # LLM sanity check
        sanity_check_prompt = chat_tmpl.system_user_assistant(
            "You are a strict text auditor. Your job is to verify data fidelity.",
            f"I have a SOURCE text and an EXTRACTED snippet. \n"
            f"Task: Verify if the EXTRACTED snippet appears in the SOURCE text.\n"
            f"Rules:\n"
            f"1. Ignore differences in whitespace (newlines, tabs, spaces).\n"
            f"2. Ignore minor formatting (bullet points vs dashes).\n"
            f"3. STRICTLY FAIL if the text is summarized, reworded, or contains new words.\n\n"
            f"--- SOURCE TEXT START ---\n{note_text}\n--- SOURCE TEXT END ---\n\n"
            f"--- EXTRACTED SNIPPET START ---\n{candidate_text}\n--- EXTRACTED SNIPPET END ---\n\n"
            f'Does the snippet match? Reply with exactly one JSON object: {{"match": true}} or {{"match": false}}.'
        )

        print("  Running LLM Sanity Check...")
        check_output, _ = run_model(
            sanity_check_prompt,
            model,
            tokenizer,
            {"max_new_tokens": 100, "do_sample": False},
        )

        is_pass = "true" in check_output.lower()
        if is_pass and len(candidate_text) > 20:
            print(f"  Success on attempt {attempt + 1} (Verified by LLM, {len(candidate_text)} chars)")
            return candidate_text

        print(f"  Attempt {attempt + 1} failed LLM sanity check. Retrying...")

    print("  All attempts failed. Skipping plan-specific extraction.")
    return None


# Two-phase extraction: Phase 1 (base facts) feeds context into Phase 2 (dependent reasoning)
PHASE1_KEYS = ["Reason_for_Visit", "Cancer_Diagnosis", "Lab_Results",
               "Clinical_Findings", "Current_Medications", "Treatment_Changes"]
PHASE2_KEYS = ["Treatment_Goals", "Response_Assessment"]


def _build_cross_context(keypoints):
    """Build context string from Phase 1 extraction results for Phase 2."""
    parts = ["CONTEXT from other extractions of this same note (use as reference):"]

    cancer = keypoints.get("Cancer_Diagnosis", {})
    if isinstance(cancer, dict):
        toc = cancer.get("Type_of_Cancer", "")
        stage = cancer.get("Stage_of_Cancer", "")
        met = cancer.get("Metastasis", "")
        if toc:
            parts.append(f"- Cancer type: {toc}")
        if stage:
            parts.append(f"- Stage: {stage}")
        if met:
            parts.append(f"- Metastasis: {met}")

    meds = keypoints.get("Current_Medications", {})
    if isinstance(meds, dict):
        cm = meds.get("current_meds", "")
        if cm:
            parts.append(f"- Current oncologic medications: {cm}")
        else:
            parts.append("- Current oncologic medications: none")

    findings = keypoints.get("Clinical_Findings", {})
    if isinstance(findings, dict):
        f = findings.get("findings", "")
        if f:
            parts.append(f"- Clinical findings: {f}")

    if len(parts) <= 1:
        return ""  # no useful context

    parts.append("\nUse this context to inform your extraction below.")
    return "\n".join(parts)


def _run_letter_only(config_path, progress_paths):
    """Generate patient letters from existing extraction results.

    Loads one or more progress.json files, merges them, builds KV cache
    per note, and generates letters without re-running extraction.

    Usage:
        python run.py exp/full_qwen.yaml --letter-only results/run1/progress.json results/run2/progress.json
    """
    config = load_config(config_path)
    letter_prompt_template = config["_prompts"]["letter_generation"]["patient_letter"]

    # Merge progress files
    merged = {}
    for path in progress_paths:
        with open(path, "r") as f:
            p = json.load(f)
        for idx, row_result in p.get("results", {}).items():
            merged[idx] = row_result
    print(f"Loaded {len(merged)} rows from {len(progress_paths)} progress file(s)")

    # Create output directory
    exp_name = config["experiment"]["name"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", f"letter_{exp_name}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(run_dir)
    results_path = os.path.join(run_dir, "results.txt")

    # Write header
    with open(results_path, "w") as f:
        f.write(f"Source: {config_path} --letter-only | Run started: {datetime.now()}\n")

    # Load model
    print("Loading model...")
    model_cfg = config["model"]
    hf_token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf.token")
    if os.path.exists(hf_token_path):
        with open(hf_token_path, "r") as f:
            hftoken = f.read().strip()
        os.environ["HF_HOME"] = model_cfg.get("cache_dir", os.environ.get("HF_HOME", ""))
        from huggingface_hub import login
        login(token=hftoken)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": model_cfg.get("device_map", "auto"),
        "cache_dir": model_cfg.get("cache_dir", None),
    }
    if model_cfg.get("quantization") == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=dtype
        )
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"], cache_dir=model_cfg.get("cache_dir", None)
    )
    model.eval()
    chat_tmpl = ChatTemplate(model_cfg.get("chat_template", "llama3"))

    keypoint_config = config["generation"]["keypoint"].copy()
    keypoint_config["eos_token_id"] = tokenizer.eos_token_id

    # Process each row
    global_start = time.time()
    output_results = {}
    print(f"\nGenerating letters for {len(merged)} rows...")

    for i, (idx, row_result) in enumerate(sorted(merged.items(), key=lambda x: int(x[0]))):
        row_start = time.time()
        note_text = row_result["note_text"]
        keypoints = row_result["keypoints"]
        attribution = row_result.get("attribution", {})
        print(f"\nRow {idx} ({i+1}/{len(merged)})...")

        # Build KV cache for this note
        model_note = note_text.replace("*****", "[REDACTED]")
        base_cache = build_base_cache(model_note, model, tokenizer, "", chat_tmpl=chat_tmpl)

        # Generate letter
        letter_gen_config = keypoint_config.copy()
        letter_gen_config["max_new_tokens"] = 768
        tagged_text = generate_tagged_letter(
            keypoints, model, tokenizer, chat_tmpl,
            letter_gen_config, base_cache, letter_prompt_template,
            note_text=note_text,
        )
        traceability = parse_tagged_letter(tagged_text, keypoints, attribution)
        letter = traceability.get("letter_text", "")
        letter, post_warnings = post_check_letter(letter)
        letter, _ = post_fix_letter(letter)
        traceability["letter_text"] = letter
        for w in post_warnings:
            print(f"  {w}")

        n_sentences = len(traceability.get("sentences", []))
        n_attributed = sum(
            1 for s in traceability.get("sentences", [])
            if s["source_fields"] != ["unattributed"] and s["source_fields"] != ["none"]
        )
        # Add note readability for comparison
        from letter_generation import flesch_kincaid_grade
        note_grade = flesch_kincaid_grade(note_text)
        if "metrics" in traceability:
            traceability["metrics"]["note_readability_grade"] = note_grade
        metrics = traceability.get("metrics", {})
        print(f"  [LETTER] {n_sentences} sentences, {n_attributed} attributed, "
              f"grade={metrics.get('readability_grade', '?')}/{note_grade} "
              f"coverage={metrics.get('field_coverage_pct', '?')}% "
              f"attr={metrics.get('attribution_complete_pct', '?')}% "
              f"({time.time() - row_start:.1f}s)")

        # Update row result
        row_result["letter"] = letter
        row_result["traceability"] = traceability
        output_results[idx] = row_result

        # Append to results.txt
        append_row_result(results_path, int(idx), row_result)

    # Save progress
    save_progress(run_dir, {
        "completed": True,
        "completed_indices": sorted(int(k) for k in output_results),
        "results": output_results,
    })

    total = time.time() - global_start
    print(f"\nLetter generation complete: {len(output_results)} rows in {total:.1f}s ({total/60:.1f}min)")
    print(f"Results in: {run_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Run medical extraction experiment")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a previous run directory to resume from",
    )
    parser.add_argument(
        "--letter-only",
        nargs="+",
        metavar="PROGRESS_JSON",
        default=None,
        help="Generate letters from existing progress.json files (skip extraction)",
    )
    args = parser.parse_args()

    # 0. Letter-only mode: generate letters from existing progress files
    if args.letter_only:
        _run_letter_only(args.config, args.letter_only)
        return

    # 1. Load config
    config = load_config(args.config)
    config["_yaml_path"] = args.config

    # 2. Set seed
    seed = config["experiment"].get("seed", 42)
    set_seed(seed)

    # 3. Determine run directory
    chash = config_hash(config)
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if args.resume:
        run_dir = args.resume.rstrip("/")
        progress = load_progress(run_dir)
        if progress is None:
            print(f"Error: No progress.json found in {run_dir}")
            return
        if progress.get("config_hash") != chash:
            print(
                f"Error: Config hash mismatch. "
                f"Expected {progress['config_hash']}, got {chash}"
            )
            return
        print(f"Resuming from {run_dir}")
        completed_indices = set(progress.get("completed_indices", []))
        saved_results = progress.get("results", {})
    else:
        # Auto-search for resumable run
        found_dir = find_resumable_run(config)
        if found_dir:
            print(f"Found incomplete run with matching config: {found_dir}")
            print("Auto-resuming...")
            run_dir = found_dir
            progress = load_progress(run_dir)
            completed_indices = set(progress.get("completed_indices", []))
            saved_results = progress.get("results", {})
        else:
            run_dir = create_run_dir(config)
            print(f"Starting new run: {run_dir}")
            completed_indices = set()
            saved_results = {}
            progress = None

    # 4. Save config snapshot
    save_config_snapshot(config, run_dir)

    # 4.5. Setup logging to run_dir/run.log
    setup_logging(run_dir)

    # 5. Load model
    print("Loading model...")
    model_cfg = config["model"]

    # Handle HF token
    hf_token_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "hf.token"
    )
    if os.path.exists(hf_token_path):
        with open(hf_token_path, "r") as f:
            hftoken = f.read().strip()
        os.environ["HF_HOME"] = model_cfg.get(
            "cache_dir", os.environ.get("HF_HOME", "")
        )
        from huggingface_hub import login
        login(token=hftoken)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

    # Quantization config
    quant_cfg = model_cfg.get("quantization")
    quantization_config = None
    if quant_cfg:
        quant_type = quant_cfg.get("type", "4bit")
        if quant_type == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type=quant_cfg.get("quant_type", "nf4"),
                bnb_4bit_use_double_quant=quant_cfg.get("double_quant", True),
            )
        elif quant_type == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Check for vLLM mode
    vllm_cfg = model_cfg.get("vllm")
    if vllm_cfg:
        from vllm_pipeline.vllm_client import VLLMClient
        base_url = vllm_cfg.get("base_url", "http://localhost:8000/v1")
        model = VLLMClient(base_url=base_url, model_name=model_cfg["name"])
        if not model.health_check():
            print("ERROR: vLLM server not reachable. Start it first.")
            sys.exit(1)
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        print(f"vLLM mode: connected to {base_url}")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
        load_kwargs = {
            "device_map": model_cfg.get("device_map", "auto"),
            "torch_dtype": torch_dtype,
        }
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **load_kwargs)
        print("Model loaded.")

    # 6. Load data
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["dataset_path"])
    row_indices = data_cfg.get("row_indices", None)
    if row_indices:
        df = df.iloc[row_indices]
        print(f"Data loaded: {len(df)} rows (indices {row_indices})")
    else:
        row_range = data_cfg.get("row_range", [0, len(df)])
        df = df.iloc[row_range[0] : row_range[1]]
        print(f"Data loaded: {len(df)} rows (range {row_range})")

    # 6.5. Determine cancer type
    cancer_type = data_cfg.get("cancer_type", None)
    if cancer_type is None:
        # Auto-detect from dataset path
        ds_path = data_cfg["dataset_path"].lower()
        if "breastca" in ds_path:
            cancer_type = "breast"
        elif "pdac" in ds_path:
            cancer_type = "pdac"
        else:
            cancer_type = "generic"
    config["_cancer_type"] = cancer_type
    print(f"Cancer type: {cancer_type}")

    # 7. Load prompts
    extraction_prompts = config["_prompts"]["extraction"]
    plan_extraction_prompts = config["_prompts"]["plan_extraction"]

    # 8. Generation configs
    keypoint_config = config["generation"]["keypoint"].copy()
    keypoint_config["eos_token_id"] = tokenizer.eos_token_id

    verify = config.get("extraction", {}).get("verify", True)

    # Select pipeline version
    pipeline = config.get("extraction", {}).get("pipeline", "v1")
    extract_fn = extract_and_verify_v2 if pipeline == "v2" else extract_and_verify
    gate_config = config.get("extraction", {}).get("gate_config", {})
    print(f"Using pipeline: {pipeline}")
    if gate_config:
        print(f"Gate config: {gate_config}")

    # Create chat template from config
    chat_template_name = config.get("model", {}).get("chat_template", "llama3")
    chat_tmpl = ChatTemplate(chat_template_name)
    print(f"Chat template: {chat_template_name}")

    # 9. Results file
    results_path = os.path.join(run_dir, "results.txt")
    if progress and completed_indices:
        # Rebuild results.txt from progress
        rebuild_results_from_progress(
            results_path, progress, config, run_time
        )
        print(f"Rebuilt results.txt from {len(completed_indices)} completed rows")
    else:
        write_results_header(results_path, config, run_time)

    # 10. Initialize progress
    current_progress = {
        "config_hash": chash,
        "seed": seed,
        "completed_indices": list(completed_indices),
        "results": saved_results,
        "completed": False,
    }

    # 11. Load medical dictionary
    med_dict = load_medical_dictionary()
    if med_dict:
        print(f"Medical dictionary loaded: {len(med_dict)} terms")

    # 11b. Load oncology drug whitelist
    whitelist = load_oncology_whitelist()
    print(f"Oncology whitelist loaded: {len(whitelist)} drugs")

    # 11c. Load supportive care drug whitelist
    supp_whitelist = load_supportive_whitelist()
    print(f"Supportive care whitelist loaded: {len(supp_whitelist)} drugs")

    # 11d. Load lab test whitelist
    lab_whitelist = load_lab_whitelist()
    print(f"Lab test whitelist loaded: {len(lab_whitelist)} terms")

    # 11e. Load genetic test keywords
    genetic_tests = load_genetic_tests()
    print(f"Genetic test keywords loaded: {len(genetic_tests)} terms")

    # 12. Main loop
    global_start = time.time()
    print(f"\nProcessing {len(df)} rows...")
    for index, row in df.iterrows():
        if index in completed_indices:
            print(f"Skipping row {index} (already completed)")
            continue

        row_start = time.time()
        print(f"\nProcessing row {index} ({list(df.index).index(index) + 1}/{len(df)})...")

        note_text = row["note_text"]

        # Pre-process: replace de-identification markers with explicit [REDACTED]
        # to prevent model from fabricating values for masked data (e.g., lab values)
        model_note = note_text.replace("*****", "[REDACTED]")

        # Find relevant medical term definitions for this note
        definitions = find_relevant_definitions(note_text, med_dict)
        defs_context = format_definitions_context(definitions)
        if definitions:
            print(f"  Injected {len(definitions)} term definitions: {[d[0] for d in definitions]}")

        # Extract assessment/plan with retries (use original for regex, model gets processed)
        ap_start = time.time()
        assessment_and_plan = extract_assessment_plan(
            note_text, model, tokenizer, config, chat_tmpl=chat_tmpl
        )
        print(f"  A/P extraction: {time.time() - ap_start:.1f}s")

        # Extract keypoints from full note (two-phase)
        ext_start = time.time()
        model_ap = assessment_and_plan.replace("*****", "[REDACTED]") if assessment_and_plan else None
        base_cache = build_base_cache(model_note, model, tokenizer, defs_context, chat_tmpl=chat_tmpl)
        fullnote_cache = base_cache  # save reference for Referral extraction later

        # Phase 1: Base fact extraction (independent prompts)
        phase1_prompts = {k: extraction_prompts[k] for k in PHASE1_KEYS if k in extraction_prompts}
        phase2_prompt_keys = [k for k in PHASE2_KEYS if k in extraction_prompts]

        # V26: enable tool calling for all phases if configured
        global_tool_ctx = {"full_note": note_text, "med_dict": med_dict} if config.get("extraction", {}).get("tool_calling", False) else None
        tool_instructions = ""

        # If tool calling enabled, append tool instructions to Phase 1/2 prompts too
        if global_tool_ctx:
            tool_instructions = (
                "\n\nTOOLS (optional — use ONLY if you need to find specific information in the note):\n"
                "- To search the note for specific content: SEARCH_NOTE(\"keyword\")\n"
                "- To look up a medical term definition: DEFINE(\"term\")\n"
                "If you have enough information, do NOT use tools — output the JSON directly."
            )
            phase1_prompts = {k: v + tool_instructions for k, v in phase1_prompts.items()}

        keypoints = extract_fn(
            phase1_prompts, model, tokenizer, keypoint_config, base_cache,
            verify=verify, chat_tmpl=chat_tmpl, oncology_whitelist=whitelist,
            gate_config=gate_config, supportive_whitelist=supp_whitelist,
            tool_context=global_tool_ctx,
        )
        print(f"  Phase 1 extraction ({len(phase1_prompts)} prompts): {time.time() - ext_start:.1f}s")

        # Phase 2: Dependent extraction with cross-prompt context
        if phase2_prompt_keys:
            phase2_start = time.time()
            cross_context = _build_cross_context(keypoints)
            if cross_context:
                print(f"  Cross-context injected ({len(cross_context)} chars)")

            phase2_prompts = {}
            for k in phase2_prompt_keys:
                base_prompt = extraction_prompts[k]
                if global_tool_ctx:
                    base_prompt = base_prompt + tool_instructions
                if cross_context:
                    phase2_prompts[k] = cross_context + "\n\n" + base_prompt
                else:
                    phase2_prompts[k] = base_prompt

            phase2_keypoints = extract_fn(
                phase2_prompts, model, tokenizer, keypoint_config, base_cache,
                verify=verify, chat_tmpl=chat_tmpl, oncology_whitelist=whitelist,
                gate_config=gate_config, supportive_whitelist=supp_whitelist,
                tool_context=global_tool_ctx,
            )
            keypoints.update(phase2_keypoints)
            print(f"  Phase 2 extraction ({len(phase2_prompts)} prompts): {time.time() - phase2_start:.1f}s")

        print(f"  Total extraction: {time.time() - ext_start:.1f}s")

        # Extract plan keypoints from assessment/plan section
        # Pop fields that need full note context, not just A/P [v14]
        referral_prompt = plan_extraction_prompts.pop("Referral", None)
        genetic_results_prompt = plan_extraction_prompts.pop("Genetic_Testing_Results", None)

        if assessment_and_plan is not None:
            plan_start = time.time()
            base_cache = build_base_cache(model_ap, model, tokenizer, defs_context, chat_tmpl=chat_tmpl)
            # V26: tool calling — append tool instructions to plan prompts
            plan_tool_ctx = None
            if config.get("extraction", {}).get("tool_calling", False):
                plan_tool_ctx = {"full_note": note_text, "med_dict": med_dict}
                tool_instructions = (
                    "\n\nTOOLS (optional — use ONLY if the A/P section lacks information you need):\n"
                    "- To find information elsewhere in the full note, write: SEARCH_NOTE(\"keyword\")\n"
                    "  Example: SEARCH_NOTE(\"current medications\") or SEARCH_NOTE(\"bone scan\")\n"
                    "- To look up a medical term, write: DEFINE(\"term\")\n"
                    "  Example: DEFINE(\"TCHP\") or DEFINE(\"peritoneal carcinomatosis\")\n"
                    "If you have enough information from the A/P, do NOT use tools — output the JSON directly.\n"
                    "Write any tool calls BEFORE your JSON output, each on its own line."
                )
                # Use a copy so original prompts aren't modified for next sample
                plan_extraction_prompts_with_tools = {k: v + tool_instructions for k, v in plan_extraction_prompts.items()}
            plan_prompts_to_use = plan_extraction_prompts_with_tools if plan_tool_ctx else plan_extraction_prompts
            plan_keypoints = extract_fn(
                plan_prompts_to_use,
                model,
                tokenizer,
                keypoint_config,
                base_cache,
                verify=verify,
                chat_tmpl=chat_tmpl,
                oncology_whitelist=whitelist,
                gate_config=gate_config,
                supportive_whitelist=supp_whitelist,
                tool_context=plan_tool_ctx,
            )
            keypoints.update(plan_keypoints)
            print(f"  Plan extraction prompts: {time.time() - plan_start:.1f}s")

        # Extract Referral from full note (not just A/P) [v14]
        if referral_prompt:
            ref_start = time.time()
            ref_prompt_to_use = referral_prompt + tool_instructions if global_tool_ctx else referral_prompt
            ref_keypoints = extract_fn(
                {"Referral": ref_prompt_to_use},
                model, tokenizer, keypoint_config, fullnote_cache,
                verify=verify, chat_tmpl=chat_tmpl, oncology_whitelist=whitelist,
                gate_config=gate_config, supportive_whitelist=supp_whitelist,
                tool_context=global_tool_ctx,
            )
            keypoints.update(ref_keypoints)
            print(f"  Referral extraction (full note): {time.time() - ref_start:.1f}s")
            # Restore the prompt for next iteration
            plan_extraction_prompts["Referral"] = referral_prompt

        if genetic_results_prompt:
            gr_start = time.time()
            gr_keypoints = extract_fn(
                {"Genetic_Testing_Results": genetic_results_prompt},
                model, tokenizer, keypoint_config, fullnote_cache,
                verify=verify, chat_tmpl=chat_tmpl, oncology_whitelist=whitelist,
                gate_config=gate_config, supportive_whitelist=supp_whitelist,
            )
            keypoints.update(gr_keypoints)
            print(f"  Genetic_Testing_Results (full note): {time.time() - gr_start:.1f}s")
            plan_extraction_prompts["Genetic_Testing_Results"] = genetic_results_prompt

            # POST-GENETIC-RESULTS-IHC: strip IHC receptor pathology (ER/PR/HER2/Ki67) that
            # contaminates genetic_testing_results. If no real molecular/genetic test remains,
            # clear to "No genetic testing results in note." [extraction-audit fix]
            gtr = keypoints.get("Genetic_Testing_Results", {})
            if isinstance(gtr, dict):
                gval = str(gtr.get("genetic_testing_results", "") or "")
                gval_low = gval.lower().strip().rstrip('.')
                _no_result = ("no genetic testing results in note", "none", "n/a", "")
                if gval_low not in _no_result:
                    GENETIC_KW = [
                        "brca", "atm", "palb2", "chek2", "mlh1", "msh2", "msh6", "pms2", "epcam",
                        "lynch", "kras", "tp53", "pik3ca", "braf", "ntrk", "esr1", "egfr", "alk", "ros1",
                        "cdkn2a", "smad4", "spink1", "msi", "microsatellite", "mmr", "mismatch repair",
                        "tmb", "tumor mutational", "oncotype", "mammaprint", "foundation", "guardant",
                        "tempus", "ucsf500", "ngs", "ctdna", "germline", "mutation", "variant",
                        "pathogenic", "vus", "hrd", "homologous recombination", "pd-l1", "pdl1",
                        "recurrence score", "non-secretor", "molecular profil", "sequencing", "gene panel",
                    ]
                    ihc_re = re.compile(
                        r'\b(er|pr|her2|her-2|ki-?67|fish|ihc|erbb2)\b|cent\s*17|sig/nuc|copy #|copy number|amplification',
                        re.IGNORECASE)
                    if not any(k in gval_low for k in GENETIC_KW):
                        gtr["genetic_testing_results"] = "No genetic testing results in note."
                        print(f"    [POST-GENETIC-RESULTS-IHC] Cleared non-genetic (IHC/pathology) value: '{gval[:60]}'")
                    else:
                        segs = re.split(r'[;\n]', gval)
                        kept = []
                        for s in segs:
                            s_strip = s.strip().rstrip('.,;')
                            if not s_strip:
                                continue
                            s_low = s_strip.lower()
                            s_genetic = any(k in s_low for k in GENETIC_KW)
                            if ihc_re.search(s_strip) and not s_genetic:
                                continue  # drop IHC-receptor-only segment
                            kept.append(s_strip)
                        if not kept:
                            gtr["genetic_testing_results"] = "No genetic testing results in note."
                            print(f"    [POST-GENETIC-RESULTS-IHC] All segments were IHC; cleared")
                        else:
                            new_val = "; ".join(kept)
                            if new_val.lower().rstrip('.') != gval_low:
                                gtr["genetic_testing_results"] = new_val if new_val.endswith(".") else new_val + "."
                                print(f"    [POST-GENETIC-RESULTS-IHC] Stripped IHC contamination → '{new_val[:60]}'")

        # Sanitize keypoints: convert any list values to strings [v32 compat fix]
        for section_key, section_val in keypoints.items():
            if isinstance(section_val, dict):
                for field_key, field_val in section_val.items():
                    if isinstance(field_val, list):
                        section_val[field_key] = "; ".join(str(v) for v in field_val)

        # Precompute note_lower for all POST checks [v17]
        note_lower = note_text.lower()

        # POST-VISIT-TYPE: validate in-person field against note text [v17]
        # NOTE: many billing templates include "face-to-face" even for telehealth visits,
        # so video_visit evidence takes priority over face_to_face billing language.
        rfv = keypoints.get("Reason_for_Visit", {})
        if isinstance(rfv, dict):
            in_person = rfv.get("in-person", "") or ""
            # Strong in-person evidence (exclude billing templates like "X minutes face-to-face")
            face_to_face = bool(re.search(
                r'face.to.face encounter|in.person visit|in person with|saw .* in clinic'
                r'|seen in .* clinic|presented to .* clinic',
                note_lower))
            video_visit = bool(re.search(
                r'video visit|telehealth|televisit|zoom|telephone visit|phone visit'
                r'|virtual visit|video connection|via video|by video',
                note_lower))
            # Video visit evidence takes priority (billing templates often have "face-to-face")
            if video_visit:
                if in_person.lower() not in ("televisit", "video visit", "telehealth"):
                    rfv["in-person"] = "Televisit"
                    print(f"    [POST-VISIT-TYPE] Corrected: → Televisit (video/telehealth in note)")
            elif face_to_face and 'televisit' in in_person.lower():
                rfv["in-person"] = "in-person"
                print(f"    [POST-VISIT-TYPE] Corrected: Televisit → in-person (face-to-face in note)")

        # POST-PATIENT-TYPE: validate Patient type values [v18]
        rfv = keypoints.get("Reason_for_Visit", {})
        if isinstance(rfv, dict):
            pt = rfv.get("Patient type", "") or ""
            pt_lower = pt.lower().strip()
            VALID_PATIENT_TYPES = ["new patient", "follow up", "follow-up", "followup"]
            if pt_lower and pt_lower not in VALID_PATIENT_TYPES:
                if "new" in pt_lower:
                    rfv["Patient type"] = "New patient"
                else:
                    rfv["Patient type"] = "Follow up"
                print(f"    [POST-PATIENT-TYPE] Corrected invalid '{pt}' → '{rfv['Patient type']}'")

        # POST-PATIENT-TYPE-CC: Cross-validate Patient type against Chief Complaint [v23]
        rfv = keypoints.get("Reason_for_Visit", {})
        if isinstance(rfv, dict):
            pt = rfv.get("Patient type", "") or ""
            pt_lower = pt.lower().strip()
            # Extract CC section from note
            cc_match = re.search(
                r'(?:chief\s+complaint|reason\s+for\s+visit|cc)\s*[:\s]*(.*?)(?:diagnosis|history|hpi|interim|subjective|\n\n)',
                note_text[:2000].lower(), re.DOTALL
            )
            if cc_match:
                cc_text = cc_match.group(1).strip()
                # Check for explicit follow-up in CC
                cc_has_followup = bool(re.search(r'\bfollow[\s-]?up\b|\bfup\b|\bf/u\b', cc_text))
                cc_has_new = bool(re.search(r'\bnew\s+patient\b|\bnew\s+consult\b|\bnew\s+patient\s+evaluation\b|\bconsultation\b|\b2nd\s+opinion\b|\bsecond\s+opinion\b', cc_text))
                if cc_has_followup and not cc_has_new and pt_lower == "new patient":
                    rfv["Patient type"] = "Follow up"
                    print(f"    [POST-PATIENT-TYPE-CC] CC says follow-up but model said 'New patient' → 'Follow up'")
                elif cc_has_new and not cc_has_followup and pt_lower in ("follow up", "follow-up", "followup"):
                    rfv["Patient type"] = "New patient"
                    print(f"    [POST-PATIENT-TYPE-CC] CC says new patient/consultation but model said '{pt}' → 'New patient'")

        # POST-REFERRAL: Search full note for referral patterns (plan extraction only sees A/P)
        referral = keypoints.get("Referral", {})
        if isinstance(referral, dict):
            # Search for "Ambulatory Referral to X", "refer to X", "Refer to X" in full note
            ref_patterns = re.findall(
                r'(?:ambulatory\s+)?referral\s+to\s+([^,.\n:;()\-–—]+)'
                r'|(?:I\s+will\s+|will\s+)?[Rr]efer\s+(?:her|him|the\s+patient|patient\s+)?to\s+([^,.\n:;()\-–—]+)',
                note_text, re.IGNORECASE
            )
            for groups in ref_patterns:
                match = (groups[0] or groups[1]).strip()
                if not match:
                    continue
                ml = match.lower()
                # Skip current department and non-referral items
                if any(skip in ml for skip in ['medical oncology', 'med onc', 'this clinic']):
                    continue
                # Categorize and add if missing
                if 'social work' in ml:
                    others = referral.get("Others", "None") or "None"
                    if 'social work' not in others.lower():
                        referral["Others"] = (others + ", Social work referral").lstrip("None, ").strip(", ")
                        print(f"    [POST-REFERRAL] found in full note: Social work")
                elif 'exercise' in ml and 'counseling' in ml:
                    others = referral.get("Others", "None") or "None"
                    if 'exercise' not in others.lower():
                        referral["Others"] = (others + ", Exercise counseling referral").lstrip("None, ").strip(", ")
                        print(f"    [POST-REFERRAL] found in full note: Exercise counseling")
                elif 'nutrition' in ml:
                    nutr = referral.get("Nutrition", "None") or "None"
                    if nutr == "None":
                        referral["Nutrition"] = "Nutrition referral"
                        print(f"    [POST-REFERRAL] found in full note: Nutrition")
                else:
                    # POST-SPECIALTY: Check if it's a specialty referral
                    SPECIALTY_KEYWORDS = [
                        'radiation oncology', 'rad onc',
                        'surgical oncology', 'surgery consult',
                        'gynecologic oncology', 'gyn onc', 'gynecologic',
                        'palliative', 'hospice',
                        'integrative medicine', 'complementary medicine',
                        'psychology', 'psychiatry', 'behavioral health',
                        'dental', 'cardiology', 'pulmonology', 'neurology',
                        'dermatology', 'urology', 'endocrinology',
                        'pain management', 'orthopedic', 'orthopaedic',
                        'plastic surgery', 'reconstructive',
                        'fertility', 'reproductive',
                    ]
                    if any(sk in ml for sk in SPECIALTY_KEYWORDS):
                        # Skip if match contains non-specialty text (note body leakage)
                        NON_SPECIALTY_PHRASES = [
                            'history of present', 'assessment', 'plan:',
                            'chief complaint', 'review of systems', 'physical exam',
                            'for consideration of', 'at the completion',
                            'will be made', 'to consider either',
                        ]
                        if any(ns in ml for ns in NON_SPECIALTY_PHRASES):
                            continue
                        # Skip if match contains imaging/order text leaking in
                        if re.search(r'\b(?:CT|MRI|PET|scan|order|lab)\b', match, re.IGNORECASE):
                            continue
                        spec = referral.get("Specialty", "None") or "None"
                        # Extract just the specialty name (first matching keyword phrase)
                        matched_specialty = None
                        for sk in SPECIALTY_KEYWORDS:
                            if sk in ml:
                                matched_specialty = sk.title()
                                break
                        match_short = matched_specialty if matched_specialty else match[:25].strip()
                        # Dedup: skip if this specialty is already present
                        if match_short.lower() in spec.lower():
                            continue
                        referral["Specialty"] = (spec + ", " + match_short).lstrip("None, ").strip(", ")
                        print(f"    [POST-SPECIALTY] found in full note: {match_short}")

        # POST-GENETICS: Remove mutation findings from Genetics referral field [B70, v17]
        # Prompt says "do NOT list genetic test RESULTS or known mutations" but model
        # sometimes still puts "BRCA1 mutation" etc. in the Genetics referral field.
        if isinstance(referral, dict):
            gen_val = referral.get("Genetics", "None") or "None"
            if gen_val != "None":
                gen_lower = gen_val.lower()
                # If it mentions mutation/carrier/positive/negative but NOT refer/consult → it's a finding, not a referral
                RESULT_WORDS = [
                    "mutation", "carrier", "positive", "pathogenic", "deleterious",
                    "negative", "no mutation", "no deleterious", "vus", "variant of uncertain",
                    "benign", "likely benign", "non-informative", "normal",
                    "variant", "wild type", "wild-type", "detected", "identified",
                ]
                REFERRAL_WORDS = [
                    "refer", "consult", "counseling", "counsel", "evaluation",
                    "recommend", "genetic testing", "send for",
                ]
                GENE_NAMES = ["brca1", "brca2", "palb2", "chek2", "atm", "tp53", "pten",
                              "cdh1", "stk11", "pms2", "mlh1", "msh2", "msh6", "ctnna1"]
                has_finding = any(w in gen_lower for w in RESULT_WORDS)
                has_referral = any(w in gen_lower for w in REFERRAL_WORDS)
                # v17: pure gene name without referral verb → it's a result, not a referral
                is_pure_gene = any(gn in gen_lower for gn in GENE_NAMES) and not has_referral
                # v18: result words take priority — if test results exist, it's completed, not a pending referral
                if has_finding:
                    referral["Genetics"] = "None"
                    print(f"    [POST-GENETICS] cleared completed test result: '{gen_val}'")
                elif is_pure_gene:
                    referral["Genetics"] = "None"
                    print(f"    [POST-GENETICS] cleared gene name (no referral verb): '{gen_val}'")

        # POST-OTHERS: Clean up Others field to only keep recognized referral types [B82]
        # Model sometimes puts lifestyle advice ("I recommend anti inflammatory diet...") in Others
        if isinstance(referral, dict):
            others_val = referral.get("Others", "None") or "None"
            if others_val != "None":
                OTHERS_REFERRAL_PATTERNS = [
                    "social work", "physical therapy", "occupational therapy",
                    "exercise counseling", "financial counseling", "home health",
                    "speech therapy", "lymphedema", "pt referral", "ot referral",
                ]
                items = [item.strip() for item in re.split(r'[,;.]', others_val) if item.strip()]
                kept = [item for item in items if any(p in item.lower() for p in OTHERS_REFERRAL_PATTERNS)]
                new_val = ", ".join(kept) if kept else "None"
                if new_val != others_val:
                    referral["Others"] = new_val
                    print(f"    [POST-OTHERS] cleaned: '{others_val[:80]}...' → '{new_val}'")

        # POST-NUTRITION: Clean up Nutrition referral false positives [B88]
        # Model sometimes puts general diet advice ("I recommend anti inflammatory diet...")
        # in Nutrition field. True nutrition referrals contain "refer/consult/follow up with nutritionist".
        if isinstance(referral, dict):
            nutr_val = referral.get("Nutrition", "None") or "None"
            if nutr_val != "None":
                nutr_lower = nutr_val.lower()
                has_referral_kw = any(w in nutr_lower for w in [
                    "refer", "consult", "follow up with nutri", "follow-up with nutri",
                    "nutritionist", "dietitian", "dietician", "nutrition referral",
                    "nutrition appointment", "nutrition on ",
                ])
                if not has_referral_kw:
                    referral["Nutrition"] = "None"
                    print(f"    [POST-NUTRITION] cleared diet advice (not a referral): '{nutr_val[:80]}'")

        # POST-REFERRAL-VALIDATE: verify LLM-extracted Specialty exists in note [v17]
        if isinstance(referral, dict):
            spec_val = referral.get("Specialty", "") or ""
            if spec_val and spec_val.lower() not in ("none", "none.", ""):
                VALIDATE_KEYWORDS = {
                    "radiation": ["radiation", "rad onc", "xrt", "radiotherapy"],
                    "surgical": ["surgical oncology", "surgeon", "surgery consult"],
                    "palliative": ["palliative", "hospice"],
                    "cardiology": ["cardiology", "cardiac", "cardio-onc"],
                    "genetics": ["genetic counseling", "genetics referral", "genetic testing referral"],
                }
                spec_lower = spec_val.lower()
                for category, keywords in VALIDATE_KEYWORDS.items():
                    if any(kw in spec_lower for kw in keywords):
                        # This specialty was extracted — verify it exists in note
                        found_in_note = any(kw in note_lower for kw in keywords)
                        if not found_in_note:
                            referral["Specialty"] = ""
                            print(f"    [POST-REFERRAL-VALIDATE] Removed '{spec_val}': not found in note")
                        break

        # POST-LAB: Remove imaging terms from Lab_Plan [B87]
        # Model sometimes confuses imaging (doppler, ultrasound) with lab tests.
        # Also remove "labs reviewed" type statements which describe past/current status, not future plans.
        lab = keypoints.get("Lab_Plan", {})
        if isinstance(lab, dict):
            lab_val = lab.get("lab_plan", "") or ""
            if isinstance(lab_val, dict):
                lab_val = "; ".join(f"{k}: {v}" for k, v in lab_val.items() if v)
            if isinstance(lab_val, list):
                lab_val = "; ".join(str(v) for v in lab_val)
            lab_val = str(lab_val) if lab_val else ""
            if lab_val and lab_val.lower() not in ("no labs planned.", "no labs planned", "none", ""):
                lab_lower = lab_val.lower()
                LAB_IMAGING_TERMS = [
                    "doppler", "ultrasound", "ct ", "ct,", "mri", "pet", "dexa",
                    "bone scan", "x-ray", "xray", "mammogram", "echocardiogram",
                    "echo ", "scan ",
                    # cardiac-function imaging abbreviations (pre-anthracycline workup) — these are
                    # imaging/cardiac studies, not labs. [fix#7, b8/b17 "pre-chemotherapy TTE"]
                    "tte", "echocardiography", "muga", " ekg", "ekg ", " ecg", "ecg ", "mugascan",
                ]
                has_imaging = any(t in lab_lower for t in LAB_IMAGING_TERMS)
                # "labs reviewed" / "labs adequate" = past/current status, not a future plan
                is_past_status = any(t in lab_lower for t in [
                    "labs reviewed", "labs adequate", "labs are adequate",
                    "reviewed and adequate", "labs were",
                ])
                if has_imaging or is_past_status:
                    # Try to salvage any real lab items by splitting on commas
                    items = [item.strip() for item in re.split(r'[,;]', lab_val) if item.strip()]
                    kept = []
                    for item in items:
                        il = item.lower()
                        item_has_imaging = any(t in il for t in LAB_IMAGING_TERMS)
                        item_is_past = any(t in il for t in ["reviewed", "adequate", "were"])
                        if not item_has_imaging and not item_is_past:
                            kept.append(item)
                    new_val = ", ".join(kept) if kept else "No labs planned."
                    if new_val != lab_val:
                        lab["lab_plan"] = new_val
                        print(f"    [POST-LAB] cleaned: '{lab_val[:80]}' → '{new_val}'")

        # POST-LAB-WHITELIST: Validate Lab_Plan items against lab test whitelist [v14]
        # Remove items that don't match any known lab test (e.g., "lumbar puncture" → procedure, not lab)
        lab = keypoints.get("Lab_Plan", {})
        if isinstance(lab, dict):
            lab_val = lab.get("lab_plan", "") or ""
            if isinstance(lab_val, dict):
                lab_val = "; ".join(f"{k}: {v}" for k, v in lab_val.items() if v)
            if isinstance(lab_val, list):
                lab_val = "; ".join(str(v) for v in lab_val)
            lab_val = str(lab_val) if lab_val else ""
            if lab_val and lab_val.lower() not in ("no labs planned.", "no labs planned", "none", "none planned.", ""):
                items = [item.strip() for item in re.split(r'[,;]', lab_val) if item.strip()]
                kept = []
                removed = []
                # Non-lab items to explicitly exclude
                NON_LAB_TERMS = ["lumbar puncture", " lp ", "biopsy", "imaging",
                                 "mammogram", "mri", "ct scan", "pet", "ultrasound",
                                 "oncotype", "mammaprint", "brca", "genomic", "molecular"]
                for item in items:
                    il = item.lower()
                    is_non_lab = any(t in il for t in NON_LAB_TERMS)
                    matches_whitelist = any(t in il for t in lab_whitelist)
                    if is_non_lab:
                        removed.append(item)
                    elif matches_whitelist:
                        kept.append(item)
                    else:
                        # Item doesn't match whitelist but also isn't explicitly non-lab
                        # Keep it to avoid over-filtering (conservative approach)
                        kept.append(item)
                if removed:
                    new_val = ", ".join(kept) if kept else "No labs planned."
                    lab["lab_plan"] = new_val
                    print(f"    [POST-LAB-WHITELIST] removed non-lab items: {[r[:50] for r in removed]}")

        # POST-LAB-SEARCH: If lab_plan is empty, search full note for ordered labs [v22e]
        lab_search = keypoints.get("Lab_Plan", {})
        if isinstance(lab_search, dict):
            lab_val_s = str(lab_search.get("lab_plan", "") or "").strip().lower()
            if lab_val_s in ("no labs planned.", "no labs planned", "none", "none planned.", ""):
                # Search HPI and Orders section for "ordered labs" or specific lab names
                LAB_PATTERNS = [
                    r'(?:ordered?|will\s+order|plan.*for)\s+[^.]*?(?:labs|blood work|CBC|CMP|tumor marker|CA\s*(?:15-3|27\.29)|CEA)',
                    r'(?:labs\s+to\s+complete|complete.*work\s*up.*labs)',
                ]
                found_labs = []
                for pat in LAB_PATTERNS:
                    m = re.search(pat, note_text, re.IGNORECASE)
                    if m:
                        found_labs.append(m.group(0).strip()[:80])
                if found_labs:
                    lab_search["lab_plan"] = ". ".join(found_labs)
                    print(f"    [POST-LAB-SEARCH] Found labs in full note: {found_labs[0][:60]}")

        # POST-THERAPY: Whitelist filter for Therapy_plan
        # Only keep sentences mentioning actual cancer therapy (drugs, regimens, modalities).
        # Removes contaminants like antiviral drugs (valtrex), antibiotics, etc.
        therapy = keypoints.get("Therapy_plan", {})
        if isinstance(therapy, dict):
            therapy_val = therapy.get("therapy_plan", "") or ""
            if therapy_val and therapy_val.lower() not in ("none", ""):
                # Build therapy whitelist: oncology drugs + therapy category keywords
                THERAPY_CATEGORY_TERMS = {
                    'chemotherapy', 'chemo', 'radiation', 'radiotherapy', 'xrt', ' rt ',
                    'hormonal therapy', 'hormone blockade', 'endocrine therapy', 'anti-hormonal',
                    'aromatase inhibitor', 'ovarian suppression', 'immunotherapy',
                    'targeted therapy', 'bone therapy', 'adjuvant', 'neoadjuvant',
                    'clinical trial', 'dose dense', 'dose-dense', 'dd ac',
                    'ac-t', 'ac-taxol', 'tchp', 'folfox', 'folfiri', 'fec',
                    ' tc ', ' ac ', ' ec ', ' cmf ',
                    'cycle', 'regimen', 'infusion',
                    'exercise', 'rec exercise',  # doctor feedback: exercise Rx is therapy
                }
                therapy_terms = whitelist | THERAPY_CATEGORY_TERMS
                # Split by sentences and keep only therapy-related ones
                sentences = [s.strip() for s in re.split(r'(?<=[.;])\s+', therapy_val) if s.strip()]
                kept = []
                removed = []
                for sent in sentences:
                    sl = " " + sent.lower() + " "  # pad for word-boundary matching (' rt ')
                    if any(term in sl for term in therapy_terms):
                        kept.append(sent)
                    else:
                        removed.append(sent)
                if removed and kept:
                    new_val = " ".join(kept)
                    therapy["therapy_plan"] = new_val
                    print(f"    [POST-THERAPY] removed non-therapy: {[s[:50] for s in removed]}")
                elif removed and not kept:
                    # All sentences removed — check if original has ANY oncology drug names
                    # If no drugs mentioned at all, this is probably a mis-extraction → clear it
                    tv_lower = therapy_val.lower()
                    has_drugs = any(term in tv_lower for term in whitelist)
                    has_unspecified = 'unspecified agent' in tv_lower or 'unspecified' in tv_lower
                    has_continue = 'continue' in tv_lower and ('agent' in tv_lower or 'therapy' in tv_lower)
                    if not has_drugs and not has_unspecified and not has_continue:
                        therapy["therapy_plan"] = "None"
                        print(f"    [POST-THERAPY] cleared (no oncology drugs found): {therapy_val[:80]}")
                    # else: has drug names but mixed with non-therapy context → keep original

        # POST-THERAPY-SUPPLEMENT: If therapy_plan is empty but A/P mentions therapy drugs, supplement
        # Also: if therapy_plan has content but misses drugs found in A/P, append them
        therapy = keypoints.get("Therapy_plan", {})
        if isinstance(therapy, dict):
            tp_val = str(therapy.get("therapy_plan", "") or "")
            tp_lower = tp_val.lower().strip()
            tp_empty = tp_lower in ("none", "", "null", "none.")
            # Search A/P for therapy keywords
            ap_lower = (assessment_and_plan or "").lower()
            # Drug synonym groups — if any synonym is in therapy_plan, skip the drug
            THERAPY_SYNONYMS = {
                'letrozole': ['letrozole', 'femara'],
                'tamoxifen': ['tamoxifen', 'nolvadex'],
                'anastrozole': ['anastrozole', 'arimidex'],
                'exemestane': ['exemestane', 'aromasin'],
                'leuprolide': ['leuprolide', 'lupron'],
                'goserelin': ['goserelin', 'zoladex'],
                'fulvestrant': ['fulvestrant', 'faslodex'],
                'palbociclib': ['palbociclib', 'ibrance'],
                'ribociclib': ['ribociclib', 'kisqali'],
                'abemaciclib': ['abemaciclib', 'verzenio'],
                'trastuzumab': ['trastuzumab', 'herceptin'],
                'pertuzumab': ['pertuzumab', 'perjeta'],
                'denosumab': ['denosumab', 'prolia', 'xgeva'],
                'capecitabine': ['capecitabine', 'xeloda'],
                'everolimus': ['everolimus', 'afinitor'],
                'eribulin': ['eribulin', 'halaven'],
                'gemcitabine': ['gemcitabine', 'gemzar'],
            }
            THERAPY_DRUG_LIST = [
                # breast
                'letrozole','tamoxifen','anastrozole','exemestane',
                'taxol','carboplatin','palbociclib',
                'ribociclib','fulvestrant',
                'doxorubicin','cyclophosphamide','olaparib','abemaciclib',
                'goserelin','leuprolide','denosumab','epirubicin',
                'trastuzumab','lapatinib','neratinib','tucatinib',
                'eribulin',
                # PDAC / GI / pan-cancer
                'gemcitabine','abraxane','nab-paclitaxel','capecitabine','irinotecan',
                'oxaliplatin','fluorouracil','5-fu','leucovorin',
                'cisplatin','temozolomide','streptozocin',
                'everolimus','sunitinib','erlotinib','sorafenib','regorafenib',
                'octreotide','lanreotide',
                'pembrolizumab','nivolumab','atezolizumab','durvalumab','ipilimumab',
                'bevacizumab','ramucirumab',
                'rucaparib','niraparib','larotrectinib','entrectinib',
                'dabrafenib','trametinib',
            ]
            found_therapies = []
            for drug in THERAPY_DRUG_LIST:
                if drug not in ap_lower:
                    continue
                # Check if any synonym already in therapy_plan
                synonyms = THERAPY_SYNONYMS.get(drug, [drug])
                if any(syn in tp_lower for syn in synonyms):
                    continue  # already covered
                # Check future context — strict for non-empty plans, broader for empty plans
                future_words = ['start', 'begin', 'resume', 'recommend', 'rx for', 'rx given',
                                'prescription', 'instructed to']
                if tp_empty:
                    future_words.extend(['continue', 'currently on', 'on ', 'will', 'plan'])
                for m in re.finditer(re.escape(drug), ap_lower):
                    ctx = ap_lower[max(0,m.start()-60):m.end()+60]
                    if any(fc in ctx for fc in future_words):
                        # Exclude past context
                        if not any(pc in ctx for pc in ['s/p', 'status post', 'completed',
                                                         'discontinued', 'stopped', 'was on',
                                                         'previously on', 'had ', 'tolerated']):
                            found_therapies.append(drug)
                            break

            # Check radiation — only add if therapy_plan doesn't mention it and A/P has clear future radiation
            if 'radiation' not in tp_lower and 'xrt' not in tp_lower and 'rt ' not in tp_lower and 'radiotherapy' not in tp_lower:
                # Strict future radiation patterns — require explicit referral or plan
                rad_future = re.search(r'(?:referral?\s+(?:to|for).*?(?:radiation|rt\b|xrt)|'
                                        r'(?:will|plan|recommend|benefit\s+from)\s+[^.]{0,40}(?:chest\s+wall\s+rt|radiation|xrt|radiotherapy)|'
                                        r'(?:radiation|xrt)\s+(?:planned|recommended|scheduled|consult))',
                                       ap_lower)
                if rad_future:
                    rad_ctx = rad_future.group(0)
                    if not any(pc in rad_ctx for pc in ['s/p', 'completed', 'had ', 'status post', 'declined']):
                        found_therapies.append('radiation therapy referral')

            if found_therapies:
                unique = list(dict.fromkeys(found_therapies))
                if tp_empty:
                    therapy["therapy_plan"] = "Continue/start: " + ", ".join(unique)
                    print(f"    [POST-THERAPY-SUPPLEMENT] Found therapy in A/P: {unique}")
                else:
                    # Prepend missing drugs naturally
                    drug_str = ", ".join(unique)
                    therapy["therapy_plan"] = f"Continue {drug_str}. {tp_val}"
                    print(f"    [POST-THERAPY-SUPPLEMENT] Prepended missing drugs: {unique}")

        # POST-THERAPY-EXERCISE: If A/P has exercise recommendation but therapy_plan doesn't [iter8 doctor feedback]
        therapy = keypoints.get("Therapy_plan", {})
        if isinstance(therapy, dict):
            tp_val = str(therapy.get("therapy_plan", "") or "")
            if 'exercise' not in tp_val.lower():
                ap_lower_ex = (assessment_and_plan or "").lower()
                ex_match = re.search(r'(?:rec(?:ommend)?\s+exercise|exercise\s+\d+\s*min)', ap_lower_ex)
                if ex_match:
                    ex_text = assessment_and_plan[ex_match.start():ex_match.end()+40] if assessment_and_plan else ""
                    # Extract the full exercise recommendation sentence
                    ex_sent = re.search(r'[^.;]*exercise[^.;]*', (assessment_and_plan or ""), re.IGNORECASE)
                    if ex_sent:
                        ex_rec = ex_sent.group(0).strip()[:80]
                        if tp_val.lower() in ("none", "", "null"):
                            therapy["therapy_plan"] = ex_rec
                        else:
                            therapy["therapy_plan"] = tp_val + ". " + ex_rec
                        print(f"    [POST-THERAPY-EXERCISE] Added exercise: {ex_rec[:60]}")

        # POST-THERAPY-SUPPORTIVE: Search A/P for supportive care items missing from therapy_plan [iter10]
        therapy = keypoints.get("Therapy_plan", {})
        if isinstance(therapy, dict):
            tp_val = str(therapy.get("therapy_plan", "") or "")
            tp_lower = tp_val.lower()
            if tp_lower not in ("none", "", "null"):
                ap_lower_sc = (assessment_and_plan or "").lower()
                SUPPORTIVE_ITEMS = {
                    'lasix': 'lasix', 'furosemide': 'lasix',
                    'kcl': 'KCL', 'potassium': 'potassium',
                    'elevation': 'elevation for edema',
                    'compression': 'compression stockings',
                    'brace': 'brace',
                    'home health': 'home health',
                    'physical therapy': 'physical therapy',
                    # NOTE: ' pt ' removed — it matched "pt" (the patient abbreviation,
                    # "pt to continue...", "will have pt undergo...") and hallucinated a
                    # physical-therapy referral that was never in the note (pdac16/pdac19/b19).
                    # Only the full phrase "physical therapy" is a safe trigger. [fix#3]
                }
                added = []
                for keyword, label in SUPPORTIVE_ITEMS.items():
                    if keyword in ap_lower_sc and keyword not in tp_lower:
                        # Check it's a current/future plan, not past
                        for m in re.finditer(re.escape(keyword), ap_lower_sc):
                            ctx = ap_lower_sc[max(0,m.start()-40):m.end()+40]
                            if any(fc in ctx for fc in ['continue', 'start', 'daily', 'mg', 'meq',
                                                         'recommend', 'for ', 'with ']):
                                if not any(pc in ctx for pc in ['s/p', 'completed', 'stopped', 'was on']):
                                    added.append(label)
                                    break
                if added:
                    unique = list(dict.fromkeys(added))
                    therapy["therapy_plan"] = tp_val + "; " + ", ".join(unique)
                    print(f"    [POST-THERAPY-SUPPORTIVE] Added supportive items: {unique}")

        # POST-MEDICATION-SUPPLEMENT: Search A/P for drugs missing from medication_plan [iter10]
        med = keypoints.get("Medication_Plan", {})
        if isinstance(med, dict):
            mp_val = str(med.get("medication_plan", "") or "")
            mp_lower = mp_val.lower()
            ap_lower_mp = (assessment_and_plan or "").lower()
            MEDICATION_DRUG_LIST = [
                # oncology drugs (breast)
                'tamoxifen','letrozole','anastrozole','exemestane','palbociclib','ibrance',
                'ribociclib','fulvestrant','faslodex','denosumab',
                'prolia','xgeva','zoladex','goserelin','leuprolide','lupron',
                'trastuzumab','herceptin','pertuzumab','olaparib','abemaciclib',
                # oncology drugs (PDAC / GI / pan-cancer)
                'gemcitabine','gemzar','abraxane','nab-paclitaxel','capecitabine','xeloda',
                'irinotecan','oxaliplatin','fluorouracil','leucovorin','temozolomide',
                'cisplatin','streptozocin',
                'everolimus','afinitor','sunitinib','sutent','erlotinib','tarceva',
                'octreotide','sandostatin','lanreotide','somatuline',
                'nivolumab','opdivo','pembrolizumab','keytruda','atezolizumab','tecentriq',
                'durvalumab','ipilimumab','bevacizumab','avastin',
                'rucaparib','niraparib','larotrectinib','entrectinib',
                'dabrafenib','trametinib','sorafenib','regorafenib',
                'creon','pancrelipase',
                # supportive/symptom meds
                'gabapentin','pregabalin','duloxetine','cymbalta','venlafaxine','effexor',
                'ondansetron','zofran','prochlorperazine','compazine',
                'dexamethasone','prilosec','omeprazole','famotidine',
                'lasix','furosemide','hydrochlorothiazide','lisinopril',
                'alendronate','fosamax','zoledronic','risedronate',
                # common meds often in A/P (iter10 review additions)
                # NOTE: only list GENERIC names here; brand names handled by MED_SYNONYMS
                'doxycycline','acetaminophen','ibuprofen',
                'naproxen','tramadol','morphine','oxycodone','hydrocodone',
                'lorazepam','loratadine','fexofenadine',
                'potassium','calcium','magnesium',
                'docusate','miralax','senna',
                'metformin','atorvastatin','rosuvastatin','levothyroxine',
            ]
            MED_SYNONYMS = {
                'tamoxifen': ['tamoxifen','nolvadex'],
                'letrozole': ['letrozole','femara'],
                'anastrozole': ['anastrozole','arimidex'],
                'exemestane': ['exemestane','aromasin'],
                'palbociclib': ['palbociclib','ibrance'],
                'fulvestrant': ['fulvestrant','faslodex'],
                'denosumab': ['denosumab','prolia','xgeva'],
                'goserelin': ['goserelin','zoladex'],
                'leuprolide': ['leuprolide','lupron'],
                'lupron': ['leuprolide','lupron'],
                'trastuzumab': ['trastuzumab','herceptin'],
                'herceptin': ['trastuzumab','herceptin'],
                'ibrance': ['palbociclib','ibrance'],
                'faslodex': ['fulvestrant','faslodex'],
                'prolia': ['denosumab','prolia','xgeva'],
                'xgeva': ['denosumab','prolia','xgeva'],
                'zoladex': ['goserelin','zoladex'],
                'prilosec': ['omeprazole','prilosec'],
                'gabapentin': ['gabapentin','neurontin'],
                'omeprazole': ['omeprazole','prilosec'],
                'acetaminophen': ['acetaminophen','tylenol'],
                'ibuprofen': ['ibuprofen','advil','motrin'],
                'naproxen': ['naproxen','naprosyn','aleve'],
                'oxycodone': ['oxycodone','roxicodone','percocet'],
                'lorazepam': ['lorazepam','ativan'],
                'docusate': ['docusate','colace'],
                'morphine': ['morphine','ms contin'],
                'tramadol': ['tramadol','ultram'],
                'ondansetron': ['ondansetron','zofran'],
                'zofran': ['ondansetron','zofran'],
                'prochlorperazine': ['prochlorperazine','compazine'],
                'compazine': ['prochlorperazine','compazine'],
                'duloxetine': ['duloxetine','cymbalta'],
                'cymbalta': ['duloxetine','cymbalta'],
                'venlafaxine': ['venlafaxine','effexor'],
                'effexor': ['venlafaxine','effexor'],
                'pregabalin': ['pregabalin','lyrica'],
            }
            # AUTO_SUPPLEMENT_SKIP: generic non-oncology meds must NOT be force-injected into the
            # oncology Medication_Plan. A long A/P often pastes prior medication lists / PRN home
            # meds; auto-adding antibiotics (doxycycline), generic analgesics (acetaminophen,
            # hydrocodone), antihistamines, statins, etc. dumps note noise into the plan (b19:
            # "; also: ondansetron, zofran, doxycycline, acetaminophen, hydrocodone"). Only
            # cancer-specific drugs + clearly chemo-supportive agents are eligible for the
            # supplement; the model's own extraction still keeps anything genuinely in the plan.
            # Principle: 精确忠实 > 不遗漏. [2026-06-06, fix#4, b19]
            AUTO_SUPPLEMENT_SKIP = {
                'doxycycline', 'acetaminophen', 'ibuprofen', 'naproxen', 'tramadol',
                'morphine', 'oxycodone', 'hydrocodone', 'lorazepam', 'loratadine',
                'fexofenadine', 'metformin', 'atorvastatin', 'rosuvastatin', 'levothyroxine',
                'hydrochlorothiazide', 'lisinopril', 'docusate', 'miralax', 'senna',
                'potassium', 'calcium', 'magnesium',
            }
            found_meds = []
            for drug in MEDICATION_DRUG_LIST:
                if drug in AUTO_SUPPLEMENT_SKIP:
                    continue
                if drug not in ap_lower_mp:
                    continue
                # Check if any synonym already in medication_plan
                synonyms = MED_SYNONYMS.get(drug, [drug])
                if any(syn in mp_lower for syn in synonyms):
                    continue
                # Check future/current context (wider window, more context words)
                for m in re.finditer(re.escape(drug), ap_lower_mp):
                    ctx = ap_lower_mp[max(0,m.start()-60):m.end()+60]
                    if any(fc in ctx for fc in ['continue', 'start', 'begin', 'resume', 'rx ',
                                                 'recommend', 'prescri', 'mg', 'daily', 'bid',
                                                 'tid', 'qd', 'tablet', 'capsule', 'sent',
                                                 'take ', 'given', 'ordered', 'increase',
                                                 'current', 'on ', 'prn', 'every', 'weekly',
                                                 'monthly', 'twice', 'three times']):
                        if not any(pc in ctx for pc in ['s/p', 'status post', 'completed',
                                                         'discontinued', 'stopped', 'was on',
                                                         'allergic to', 'allergy',
                                                         'declined', 'refused', 'not taking',
                                                         'not tolerat', 'intoleran',
                                                         'concerned about', 'avoid', 'not recommend',
                                                         'hold ', 'held ', 'risk of', 'toxicity',
                                                         'response rate', 'et al', 'reported',
                                                         'trial', 'study', 'published',
                                                         'without treatment', 'w/o treatment',
                                                         'monitoring', 'expectant management']):
                            found_meds.append(drug)
                            break
            if found_meds:
                # Filter out drugs that are mentioned as stopped/held/switched/progressed in any field
                stopped_fields = [
                    str(keypoints.get("Treatment_Changes", {}).get("recent_changes", "") or "").lower(),
                    str(keypoints.get("Response_Assessment", {}).get("response_assessment", "") or "").lower(),
                    str(keypoints.get("Reason_for_Visit", {}).get("summary", "") or "").lower(),
                ]
                stopped_patterns = ['stopped', 'held', 'discontinued', 'switched', 'changed to', 'replaced', 'no longer', 'progressed on', 'progression on']
                filtered_meds = []
                for drug in found_meds:
                    drug_lower = drug.lower()
                    drug_is_stopped = False
                    for field_text in stopped_fields:
                        for m in re.finditer(re.escape(drug_lower), field_text):
                            window = field_text[max(0, m.start()-60):m.end()+60]
                            if any(p in window for p in stopped_patterns):
                                drug_is_stopped = True
                                break
                        if drug_is_stopped:
                            break
                    if drug_is_stopped:
                        print(f"    [POST-MEDICATION-SUPPLEMENT] Skipping '{drug}' — found as stopped/progressed in extraction fields")
                    else:
                        filtered_meds.append(drug)
                # synonym-aware dedup: ondansetron/zofran, prochlorperazine/compazine etc. are the
                # same drug — collapse to one canonical entry so the plan doesn't list both. [fix#4]
                unique, _seen_canon = [], set()
                for drug in filtered_meds:
                    canon = MED_SYNONYMS.get(drug, [drug])[0]
                    if canon in _seen_canon:
                        continue
                    _seen_canon.add(canon)
                    unique.append(drug)
                if unique:
                    if mp_lower in ("none", "", "null", "none."):
                        med["medication_plan"] = "Continue/start: " + ", ".join(unique)
                    else:
                        med["medication_plan"] = mp_val + "; also: " + ", ".join(unique)
                    print(f"    [POST-MEDICATION-SUPPLEMENT] Added missing meds: {unique}")

        # POST-LAB-SUPPLEMENT: If lab_plan misses palbociclib/ibrance monitoring [breast-only]
        if cancer_type == "breast":
            lab = keypoints.get("Lab_Plan", {})
            if isinstance(lab, dict):
                lp_val = str(lab.get("lab_plan", "") or "")
                ap_lower = (assessment_and_plan or "").lower()
                if ('palbociclib' in ap_lower or 'ibrance' in ap_lower) and 'monthly' in ap_lower:
                    if 'palbociclib' not in lp_val.lower() and 'ibrance' not in lp_val.lower() and 'monthly' not in lp_val.lower():
                        if lp_val.strip() and lp_val.lower() not in ('no labs planned.', 'no labs planned', ''):
                            lab["lab_plan"] = lp_val + ". Monthly blood work for Palbociclib monitoring."
                        else:
                            lab["lab_plan"] = "Monthly blood work for Palbociclib monitoring."
                        print(f"    [POST-LAB-SUPPLEMENT] Added Palbociclib monthly monitoring")

        # POST-IMAGING: Search full note for imaging plans mentioned outside A/P
        # Similar to POST-PROCEDURE: echocardiogram, DEXA, etc. may be in A/P as standalone items
        img = keypoints.get("Imaging_Plan", {})
        if isinstance(img, dict):
            img_val = img.get("imaging_plan", "") or ""
            img_lower = (img_val or "").lower()
            # Search for future imaging patterns in full note
            IMAGING_TYPES = {
                'echocardiogram': 'Echocardiogram',
                'echo ': 'Echocardiogram',
                'dexa': 'DEXA scan',
                'bone density scan': 'DEXA scan',
                'bone density test': 'DEXA scan',
                'mammogram': 'Mammogram',
                'ct chest': 'CT Chest',
                'ct scan': 'CT scan',
                'ct abdomen': 'CT Abdomen',
                'brain mri': 'Brain MRI',
                'mr brain': 'Brain MRI',
                'mri brain': 'Brain MRI',
                'mri of brain': 'Brain MRI',
                'mri breast': 'MRI Breast',
                'pet/ct': 'PET/CT',
                'pet ct': 'PET/CT',
                'bone scan': 'Bone scan',
                'ultrasound': 'Ultrasound',
            }
            # Synonyms for dedup: if any synonym is in existing text, skip adding this label
            IMAGING_SYNONYMS = {
                'Echocardiogram': ['echocardiogram', 'echo ', 'echo.', 'echo,'],
                'DEXA scan': ['dexa', 'bone density scan', 'bone density test'],
                'Mammogram': ['mammogram'],
                'PET/CT': ['pet/ct', 'pet ct', 'petct', 'pet-ct'],
                'Bone scan': ['bone scan'],
                'CT Chest': ['ct chest'],
                'CT scan': ['ct scan', 'ct '],
                'CT Abdomen': ['ct abdomen'],
                'Brain MRI': ['brain mri', 'mr brain', 'mri brain', 'mri of brain'],
                'MRI Breast': ['mri breast'],
                'Ultrasound': ['ultrasound'],
            }
            # Search A/P first; if imaging_plan is empty, also search full note (HPI, Orders) [v22e]
            search_text = assessment_and_plan if assessment_and_plan else note_text
            search_fullnote = img_lower in ("no imaging planned.", "no imaging planned", "none", "none planned.", "")
            for pattern, label in IMAGING_TYPES.items():
                # Check if any synonym already exists in current imaging_plan
                synonyms = IMAGING_SYNONYMS.get(label, [label.lower()])
                if any(syn in img_lower for syn in synonyms):
                    continue  # already captured (including synonyms)
                # Look for imaging with future context in A/P
                regex = (
                    r'(?:will\s+(?:order|schedule|get|have|obtain|need)|'
                    r'plan\s+(?:for|to)|scheduled?\s+(?:for|a)|'
                    r'consider\s+(?:a\s+)?(?:follow\s*up\s+)?|'
                    r'recommend\s+|due\s+(?:for|in)|pending\s+|'
                    r'ordered?\s+(?:a\s+)?|need\s+(?:a\s+)?|baseline\s+)'
                    r'[^.;]{0,30}' + re.escape(pattern)
                    + r'|' + re.escape(pattern) + r'[^.;]{0,20}(?:ordered|planned|scheduled|due)'
                    + r'|' + re.escape(pattern) + r'\.?\s*(?:Port|$|\d)'
                )
                # Search A/P first, then full note if plan is empty
                found = re.search(regex, search_text, re.IGNORECASE)
                if not found and search_fullnote:
                    # Fallback: bare keyword ONLY when imaging_plan is empty
                    # Exclude past-result context (e.g., "PET CT showed stable disease", "incl bone scan")
                    bare_match = re.search(r'(?:^|[\s\-\.])' + re.escape(pattern), search_text, re.IGNORECASE)
                    if bare_match:
                        # Check context AFTER keyword
                        after = search_text[bare_match.end():bare_match.end()+50].lower()
                        past_after = re.search(r'^\s*(?:showed?|demonstrated?|revealed?|found|was\s|had\s|done|'
                                               r'completed|results?[\s:]|on\s+\d|from\s+\d|dated?\s|'
                                               r'as\s|negative|positive|normal|stable|obtained|reviewed|'
                                               r'interpretation|finding|scan\s|report)', after)
                        # Check context BEFORE keyword
                        before_start = max(0, bare_match.start() - 40)
                        before = search_text[before_start:bare_match.start()].lower()
                        past_before = re.search(r'(?:recent|prior|previous|last|incl|including|s/p|negative\s+for|'
                                                r'w/u|workup|work[\s-]*up|probable|based\s+on)', before)
                        if not past_after and not past_before:
                            found = True
                if not found and search_fullnote:
                    found = re.search(regex, note_text, re.IGNORECASE)
                if found:
                    if img_lower in ("no imaging planned.", "no imaging planned", "none", "none planned.", ""):
                        img["imaging_plan"] = label
                    else:
                        img["imaging_plan"] = img_val + ". " + label
                    img_val = img["imaging_plan"]
                    img_lower = img_val.lower()
                    search_fullnote = False  # found something, stop searching full note
                    print(f"    [POST-IMAGING] found in note: {label}")

        # POST-PROCEDURE: Search full note for procedure plans mentioned outside A/P [B75]
        # Like POST-REFERRAL, procedures (port placement, biopsy) may be in HPI, not A/P
        proc = keypoints.get("Procedure_Plan", {})
        if isinstance(proc, dict):
            proc_val = proc.get("procedure_plan", "")
            if isinstance(proc_val, dict):
                proc_val = "; ".join(f"{k}: {v}" for k, v in proc_val.items() if v)
            if isinstance(proc_val, list):
                proc_val = ", ".join(str(v) for v in proc_val)
            proc_val = str(proc_val) if proc_val else ""
            proc_lower = proc_val.lower()
            # Search for future procedure patterns in full note
            # Note: (?:an?\s+)? handles optional articles ("scheduled for a port placement")
            proc_patterns = re.findall(
                r'(?:plan\s+for|scheduled\s+for|will\s+(?:have|undergo|schedule|get|need)|'
                r'pending|to\s+be\s+scheduled|arrange|is\s+scheduled\s+for)'
                r'(?:\s+an?\s+|\s+)'
                r'((?:port|mediport|chemo\s*port)\s+(?:placement|insertion|removal)'
                r'|(?:core|needle|incisional|excisional|skin|punch)?\s*biopsy'
                r'|lumbar\s+puncture|bone\s+marrow\s+(?:biopsy|aspirat)'
                r'|surgery|mastectomy|lumpectomy|colectomy|resection'
                r'|sentinel\s+(?:lymph\s+)?node\s+biopsy)',
                note_text, re.IGNORECASE
            )
            for match in proc_patterns:
                match_clean = match.strip()
                if match_clean and match_clean.lower() not in proc_lower:
                    if proc_lower in ("no procedures planned.", "no procedures planned", "none", "none planned.", ""):
                        proc["procedure_plan"] = match_clean
                    else:
                        proc["procedure_plan"] = proc_val + ", " + match_clean
                    proc_val = proc["procedure_plan"]
                    proc_lower = proc_val.lower()
                    print(f"    [POST-PROCEDURE] found in full note: '{match_clean}'")

        # POST-PLAN-ROUTING: field-routing cleanups — keep each plan field semantically pure. [fix#7]
        # (a) FNA / core biopsy / aspiration are PROCEDURES, not imaging studies and not genetic
        #     tests. The model sometimes duplicates "plan to FNA the mass" into Imaging_Plan and
        #     Genetic_Testing_Plan (b9); strip it there (it is correctly kept in Procedure_Plan).
        PROC_ONLY_RE = re.compile(r'\b(fna|fine[- ]needle|core\s+biopsy|needle\s+biopsy|biopsy|aspiration|excisional|incisional)\b', re.I)
        IMG_MODALITY_RE = re.compile(r'\b(ct|mri|pet|ultrasound|us|mammogram|mammi|x-?ray|bone\s+scan|dexa|echo|tte|doppler|imaging|scan)\b', re.I)
        for fkey, subkey, default_v in (("Imaging_Plan", "imaging_plan", "No imaging planned."),
                                        ("Genetic_Testing_Plan", "genetic_testing_plan", "None planned.")):
            d_pr = keypoints.get(fkey, {})
            if isinstance(d_pr, dict):
                v_pr = str(d_pr.get(subkey, "") or "")
                if v_pr and PROC_ONLY_RE.search(v_pr):
                    parts_pr = [p.strip() for p in re.split(r'[;,]|(?:\.\s)', v_pr) if p.strip()]
                    kept_pr = []
                    for p in parts_pr:
                        if PROC_ONLY_RE.search(p):
                            if fkey == "Imaging_Plan" and not IMG_MODALITY_RE.search(p):
                                continue  # bare procedure, no imaging modality → not imaging
                            if fkey == "Genetic_Testing_Plan" and not re.search(
                                    r'gene|brca|germline|oncotype|mammaprint|molecular|ngs|panel|genomic', p, re.I):
                                continue  # bare procedure, no genetic content → not genetic
                        kept_pr.append(p)
                    new_pr = ". ".join(kept_pr).strip()
                    new_pr = new_pr if new_pr else default_v
                    if new_pr != v_pr:
                        d_pr[subkey] = new_pr
                        print(f"    [POST-PLAN-ROUTING] stripped procedure from {subkey}: '{v_pr[:50]}' → '{new_pr}'")

        # (b) a port / Port-a-Cath / mediport placement is a PROCEDURE. When it appears in Lab_Plan
        #     (b20 "...preparatory studies ... which may include a port...") but Procedure_Plan is
        #     empty/"No procedures planned", route a clean port statement into Procedure_Plan.
        proc_pp = keypoints.get("Procedure_Plan", {})
        lab_pp = keypoints.get("Lab_Plan", {})
        if isinstance(proc_pp, dict) and isinstance(lab_pp, dict):
            pv_pp = str(proc_pp.get("procedure_plan", "") or "")
            lv_pp = str(lab_pp.get("lab_plan", "") or "")
            proc_empty_pp = (not pv_pp) or pv_pp.lower().strip() in (
                "no procedures planned.", "no procedures planned", "none", "none planned.", "")
            if proc_empty_pp and re.search(r'\bport(?:[- ]?a[- ]?cath|acath)?\b', lv_pp, re.I):
                proc_pp["procedure_plan"] = "Port placement planned."
                print(f"    [POST-PROC-PORT] routed port placement Lab_Plan → Procedure_Plan")

        # (c) germline/genetic TESTING that was sent/ordered is a genetic test, not a referral. Move
        #     "genetic testing sent" from Referral.Genetics → Genetic_Testing_Plan when that field is
        #     empty. (A referral to a genetic COUNSELOR stays in Referral.) [b10]
        ref_g = keypoints.get("Referral", {})
        gtp_g = keypoints.get("Genetic_Testing_Plan", {})
        if isinstance(ref_g, dict) and isinstance(gtp_g, dict):
            gen_ref = str(ref_g.get("Genetics", "") or "")
            if re.search(r'genetic\s+testing\s+(?:sent|ordered|done|completed)|germline\s+(?:sent|ordered|testing)|brca\s+(?:testing\s+)?sent', gen_ref, re.I):
                cur_gtp = str(gtp_g.get("genetic_testing_plan", "") or "").strip().lower()
                if (not cur_gtp) or cur_gtp in ("none planned.", "none", "none planned", "no genetic testing planned."):
                    gtp_g["genetic_testing_plan"] = gen_ref
                    ref_g["Genetics"] = "None"
                    print(f"    [POST-REFERRAL-GERMLINE] moved '{gen_ref[:40]}' Referral.Genetics → Genetic_Testing_Plan")

        # POST-PLAN-TEMPORAL: a plan field describes FUTURE actions. A value that reports a study/lab
        # as already DONE (past-completion verbs), or a bare imaging modality the A/P confirms is
        # completed ("echo looks good", "she has done the MRI"), is not a plan — clear it. Genuine
        # future rechecks ("to check ... again", "every N months", "monitor", "will", "plan to",
        # "repeat", "ordered", "pending") are preserved. [2026-06-06, fix#8, b2 lab/b3 imaging]
        PAST_DONE_RE = re.compile(
            r'\b(?:were|was)\s+(?:also\s+)?(?:obtained|done|performed|checked|drawn|completed|reviewed)\b'
            r'|\blooks?\s+good\b|\bshowed\b|\brevealed\b|\bdemonstrated\b'
            r'|\b(?:has|have)\s+been\s+(?:done|performed|completed|obtained)\b', re.I)
        FUTURE_PLAN_RE = re.compile(
            r'\bwill\b|plan\s+to|to\s+(?:check|obtain|repeat|monitor|recheck|evaluate|assess)|\bagain\b'
            r'|every\s+\d|\bq\d|monthly|\bmonitor\b|\brepeat\b|upcoming|scheduled|order(?:ed)?|pending', re.I)
        _img_alias = {'echocardiogram': 'echo', 'tte': 'echo', 'echo': 'echo', 'muga': 'muga',
                      'pet': 'pet', 'ct': 'ct', 'mri': 'mri', 'ultrasound': 'ultrasound',
                      'us': 'ultrasound', 'mammogram': 'mammogram', 'bone scan': 'bone scan', 'dexa': 'dexa'}
        for fkey_t, subkey_t, default_t in (("Lab_Plan", "lab_plan", "No labs planned."),
                                            ("Imaging_Plan", "imaging_plan", "No imaging planned.")):
            d_t = keypoints.get(fkey_t, {})
            if isinstance(d_t, dict):
                v_t = str(d_t.get(subkey_t, "") or "")
                low_t = v_t.lower().strip()
                if not low_t or low_t in (default_t.lower(), "none", "none planned.", ""):
                    continue
                # (1) value reports completion and carries no future marker
                if PAST_DONE_RE.search(v_t) and not FUTURE_PLAN_RE.search(v_t):
                    d_t[subkey_t] = default_t
                    print(f"    [POST-PLAN-TEMPORAL] cleared completed {subkey_t}: '{v_t[:50]}'")
                    continue
                # (2) bare imaging modality the A/P marks as already done
                if fkey_t == "Imaging_Plan":
                    mod_m = re.fullmatch(r'(echocardiogram|echo|tte|muga|pet(?:/ct)?|ct|mri|ultrasound|us|mammogram|bone scan|dexa)\.?', low_t)
                    if mod_m:
                        mod_t = mod_m.group(1).split('/')[0]
                        search_tok = _img_alias.get(mod_t, mod_t)
                        ap_low_t = (assessment_and_plan or "").lower()
                        done_t = False
                        for mm in re.finditer(re.escape(search_tok), ap_low_t):
                            ctx_t = ap_low_t[max(0, mm.start() - 30):mm.end() + 30]
                            if re.search(r'looks?\s+good|has\s+done|have\s+done|done\s+the|performed|completed|reviewed|obtained', ctx_t) \
                                    and not re.search(r'\bwill\b|plan\s+to|order|schedul|repeat|every', ctx_t):
                                done_t = True
                                break
                        if done_t:
                            d_t[subkey_t] = default_t
                            print(f"    [POST-PLAN-TEMPORAL] cleared completed imaging (A/P done): '{v_t[:40]}'")

        # POST-PROCEDURE-FILTER: Remove non-procedure items from Procedure_Plan [v14]
        # Items like IHC, FISH, Oncotype, BRCA belong in genetic_testing_plan, not procedure
        proc = keypoints.get("Procedure_Plan", {})
        if isinstance(proc, dict):
            proc_val = proc.get("procedure_plan", "") or ""
            if isinstance(proc_val, list):
                proc_val = ", ".join(str(x) for x in proc_val)
            proc_val = str(proc_val)
            if proc_val and proc_val.lower() not in ("no procedures planned.", "no procedures planned", "none", "none planned.", ""):
                PROC_BLACKLIST = [
                    "ihc", "fish", "receptor testing", "staining",
                    "oncotype", "mammaprint", "brca", "genomic", "molecular",
                    "genetic testing", "gene panel", "ngs", "next generation",
                    "foundation one", "foundationone", "guardant",
                    # Referral items that don't belong in procedure_plan
                    "fertility preservation", "fertility referral", "egg harvesting",
                    "genetic counseling", "genetics counseling",
                    "rad onc referral", "radiation oncology referral",
                    "social work", "nutrition referral",
                    # Medication items that don't belong in procedure_plan [v31 expanded]
                    "zoladex", "goserelin", "prior auth",
                    "xeloda", "letrozole", "arimidex", "faslodex", "tamoxifen",
                    "exemestane", "anastrozole", "palbociclib", "ribociclib", "abemaciclib",
                    "herceptin", "pertuzumab", "trastuzumab", "t-dm1", "neratinib",
                    "epirubicin", "neupogen", "carboplatin", "docetaxel", "paclitaxel",
                    "abraxane", "capecitabine", "gemcitabine", "doxorubicin", "eribulin",
                    "taxol", "taxotere", "cyclophosphamide", "adriamycin",
                    "olaparib", "pembrolizumab", "durvalumab", "atezolizumab",
                    "continue on", "start on", "could use", "could consider",
                    "chemotherapy", "systemic therapy", "hormonal therapy", "immunotherapy",
                    "adjuvant", "neoadjuvant", "chemo", "endocrine therapy",
                    # Imaging items that don't belong in procedure_plan [v31 expanded]
                    "ct cap", "ct chest", "ct abdomen", "ct pelvis", "mri", "pet",
                    "dexa", "bone scan", "mammogram", "ultrasound", "echocardiogram",
                    "tte", "echo", "doppler", "x-ray", "xray", "petct", "pet/ct",
                    "restaging", "repeat imaging",
                    # Radiation items [v31 new]
                    "radiation", "xrt", " rt ", "gamma knife", "gk ", "srs ",
                    "radiosurgery", "whole brain", "breast radiotherapy",
                    # Non-procedure misc [v31 new]
                    "acupuncture", "pathology review", "path review",
                    "check labs", "lab draw", "blood work", "blood test",
                    "hormone studies", "estradiol", "fsh ", "tumor marker",
                    "referral", "consult", "dental clearance", "dental eval",
                ]
                items = [item.strip() for item in re.split(r'[,;]', proc_val) if item.strip()]
                kept = []
                removed = []
                for item in items:
                    il = item.lower()
                    if any(t in il for t in PROC_BLACKLIST):
                        removed.append(item)
                    else:
                        kept.append(item)
                if removed:
                    new_val = ", ".join(kept) if kept else "No procedures planned."
                    proc["procedure_plan"] = new_val
                    print(f"    [POST-PROCEDURE-FILTER] removed non-procedure items: {[r[:50] for r in removed]}")

        # POST-IMAGING-FILTER: Remove non-imaging items from Imaging_Plan [v14]
        # Items like biopsy, thoracentesis, lumbar puncture belong in procedure, not imaging
        img = keypoints.get("Imaging_Plan", {})
        if isinstance(img, dict):
            img_val = img.get("imaging_plan", "") or ""
            if img_val and img_val.lower() not in ("no imaging planned.", "no imaging planned", "none", "none planned.", ""):
                IMG_BLACKLIST = ["biopsy", "thoracentesis", "lumbar puncture", "paracentesis",
                                 "port placement", "surgery", "mastectomy", "lumpectomy"]
                items = [item.strip() for item in re.split(r'[,;.]', img_val) if item.strip()]
                kept = []
                removed = []
                for item in items:
                    il = item.lower()
                    if any(t in il for t in IMG_BLACKLIST):
                        removed.append(item)
                    else:
                        kept.append(item)
                if removed:
                    new_val = ". ".join(kept) if kept else "No imaging planned."
                    img["imaging_plan"] = new_val
                    print(f"    [POST-IMAGING-FILTER] removed non-imaging items: {[r[:50] for r in removed]}")

        # POST-GENETICS-SEARCH: Search full note for genetic testing plans [v14]
        # Model often misses Oncotype, MammaPrint, BRCA etc. when they're outside A/P
        gen = keypoints.get("Genetic_Testing_Plan", {})
        if isinstance(gen, dict):
            gen_val = gen.get("genetic_testing_plan", "") or ""
            gen_lower = gen_val.lower()
            if gen_lower in ("none planned.", "none planned", "none", "none.", ""):
                # Search full note for genetic test keywords with future context
                FUTURE_CONTEXT = [
                    "will order", "will send", "send for", "plan to",
                    "interested in", "we will await", "pending",
                    "recommend", "discussed", "consider", "plan for",
                    "will check", "will obtain", "refer for", "schedule",
                ]
                PAST_CONTEXT = [
                    "result:", "results:", "result negative", "result positive",
                    "resulted", "completed", "showed", "revealed",
                    "demonstrated", "returned", "negative for", "positive for",
                    "was performed", "were performed", "has been done",
                    "was done", "were done", "already done", "already completed",
                    "prior", "previous",
                    # Declined / refused
                    "will not pursue", "declined", "not interested",
                    "deferred", "refuses", "does not wish", "does not want",
                    "patient does not want", "opted not", "elected not",
                ]
                found_tests = []
                note_lower = note_text.lower()
                for term in genetic_tests:
                    # Use word boundary for short terms to avoid false positives
                    # (e.g. "ngs" matching "stainings", "tmb" matching "thumbs")
                    if len(term) <= 4:
                        term_pattern = r'\b' + re.escape(term) + r'\b'
                        if not re.search(term_pattern, note_lower):
                            continue
                    elif term not in note_lower:
                        continue
                    # Check if there's future context near this term
                    # Look for future keywords within 100 chars before the term
                    pattern = r'\b' + re.escape(term) + r'\b' if len(term) <= 4 else re.escape(term)
                    for match in re.finditer(pattern, note_lower):
                        start = max(0, match.start() - 100)
                        end = min(len(note_lower), match.end() + 100)
                        context_window = note_lower[start:end]
                        # Skip if past-tense/completed context found
                        if any(pc in context_window for pc in PAST_CONTEXT):
                            continue
                        if any(fc in context_window for fc in FUTURE_CONTEXT):
                            found_tests.append(term)
                            break
                if found_tests:
                    # Deduplicate and format
                    unique_tests = list(dict.fromkeys(found_tests))  # preserve order
                    gen["genetic_testing_plan"] = ", ".join(unique_tests[:3])  # cap at 3
                    print(f"    [POST-GENETICS-SEARCH] found in full note: {unique_tests}")

        # POST-GENETICS-RESULT-CHECK: Validate LLM-extracted genetic testing plan [v16]
        # Remove entries that are actually known results or explicitly declined
        gen = keypoints.get("Genetic_Testing_Plan", {})
        if isinstance(gen, dict):
            gen_val = gen.get("genetic_testing_plan", "") or ""
            gen_lower = gen_val.lower()
            if gen_lower not in ("none planned.", "none planned", "none", "none.", ""):
                note_lower_gc = note_text.lower()
                # Check 1: Is this a known mutation result (not a plan)?
                MUTATION_NAMES = ["brca1", "brca2", "palb2", "chek2", "atm", "tp53", "pms2",
                                  "mlh1", "msh2", "msh6", "pten", "cdh1", "stk11", "nbn"]
                RESULT_PHRASES = ["was done", "results reviewed", "results show", "completed",
                                  "result:", "results:", "known mutation", "known carrier",
                                  "history of", "problem list"]
                items = [g.strip() for g in gen_val.split(",")]
                valid_items = []
                for item in items:
                    item_lower = item.lower().strip()
                    if not item_lower:
                        continue
                    # Skip long sentences — only check short items (likely test names)
                    # Long LLM-generated sentences are not test names and would cause false matches
                    if len(item_lower.split()) > 6:
                        valid_items.append(item)
                        continue
                    # Pure mutation name without action words → likely a result
                    is_pure_mutation = item_lower in MUTATION_NAMES
                    has_action = any(w in item_lower for w in ["test", "order", "send", "check", "plan", "profiling", "sequencing"])
                    if is_pure_mutation and not has_action:
                        print(f"    [POST-GENETICS-RESULT-CHECK] Removed known mutation result: '{item}'")
                        continue
                    # Check if original note says this test was declined near the term
                    DECLINE_PHRASES = ["will not pursue", "declined", "not interested",
                                       "deferred", "refuses", "does not wish", "opted not", "elected not"]
                    # Use the full short item as search term (not just first word)
                    term_for_search = item_lower.strip()
                    # For multi-word items, use first meaningful word (skip articles)
                    words = [w for w in term_for_search.split() if len(w) > 3]
                    term_for_search = words[0] if words else term_for_search
                    declined = False
                    for m in re.finditer(re.escape(term_for_search), note_lower_gc):
                        ctx_start = max(0, m.start() - 120)
                        ctx_end = min(len(note_lower_gc), m.end() + 120)
                        ctx = note_lower_gc[ctx_start:ctx_end]
                        if any(dp in ctx for dp in DECLINE_PHRASES):
                            print(f"    [POST-GENETICS-RESULT-CHECK] Removed declined test: '{item}'")
                            declined = True
                            break
                        if any(rp in ctx for rp in RESULT_PHRASES):
                            print(f"    [POST-GENETICS-RESULT-CHECK] Removed completed/result: '{item}'")
                            declined = True
                            break
                    if not declined:
                        valid_items.append(item)
                if len(valid_items) < len(items):
                    gen["genetic_testing_plan"] = ", ".join(valid_items) if valid_items else "None planned."

        # POST: Patch Advance_care with code status from full note (A/P may not contain it)
        adv = keypoints.get("Advance_care_planning", {})
        adv_val = adv.get("Advance care", "") if isinstance(adv, dict) else ""
        if adv_val.lower().startswith("not discussed"):
            # Pattern 1: "Code status: Full code" or "Advance care planning: DNR/DNI"
            code_match = re.search(
                r'(?:code\s+status[:\s]*|advance\s+care\s+planning[:\s]*)(full\s+code|dnr/?dni|dnr|dni|comfort\s+measures)',
                note_text, re.IGNORECASE
            )
            # Pattern 2: Standalone "DNR/DNI" or "DNR" on its own line or in a list
            standalone_dnr = None
            if not code_match:
                standalone_dnr = re.search(
                    r'(?:^|\n|\u0007|\-\s*)\s*(DNR\s*/?\s*DNI|DNR|DNI|full\s+code)\s*(?:\.|$|\n)',
                    note_text, re.IGNORECASE | re.MULTILINE
                )
            # Pattern 3: POLST mentioned
            polst_match = re.search(r'(?:completed\s+)?POLST', note_text, re.IGNORECASE)
            # Pattern 4: Explicit wishes (would not want life support, etc.)
            wishes_match = re.search(
                r'would\s+not\s+want\s+life\s+support|would\s+not\s+want\s+(?:resuscitation|intubation|mechanical\s+ventilation)',
                note_text, re.IGNORECASE
            )
            living_will = re.search(r'living\s+will', note_text, re.IGNORECASE)
            patches = []
            if code_match:
                patches.append(code_match.group(1).strip())
            elif standalone_dnr:
                patches.append(standalone_dnr.group(1).strip())
            if polst_match:
                patches.append("POLST on file")
            if wishes_match:
                patches.append("Patient has documented wishes against life support")
            if living_will:
                patches.append("Living will on file")
            if patches:
                keypoints["Advance_care_planning"] = {"Advance care": ". ".join(patches) + "."}
                print(f"    [POST-ADV] patched from full note: {patches}")

        # POST-STAGE: Cross-validate Stage vs Metastasis [B49]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            stage = cancer.get("Stage_of_Cancer", "")
            met = cancer.get("Metastasis", "")
            dist_met = cancer.get("Distant Metastasis", "")
            met_says_no = met.lower().startswith("no") if met else False
            dist_met_says_no = dist_met.lower().startswith("no") if dist_met else True
            stage_says_iv = bool(re.search(r'stage\s*iv|metastatic', stage, re.IGNORECASE))

            if met_says_no and dist_met_says_no and stage_says_iv:
                cleaned = re.sub(r',?\s*now\s+metastatic\s*\(Stage\s*IV\)', '', stage, flags=re.IGNORECASE)
                cleaned = re.sub(r'metastatic\s*\(Stage\s*IV\)\s*,?\s*', '', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.strip().rstrip(',').strip()
                if cleaned and cleaned != stage:
                    cancer["Stage_of_Cancer"] = cleaned
                    print(f"    [POST-STAGE] contradiction fixed: '{stage}' → '{cleaned}'")

            # POST-STAGE-DISTMET: Stage IV but Distant Metastasis explicitly says "No" [v28]
            # If extraction says Stage IV but also says Distant Met = No, this is a contradiction.
            # Downgrade Stage IV to remove "metastatic" designation.
            if stage_says_iv and dist_met_says_no:
                # Check if Metastasis field only has regional sites
                met_lower = met.lower() if met else ""
                REGIONAL_SITES = ["axillary", "axilla", "sentinel", "supraclavicular",
                                  "infraclavicular", "internal mammary", "chest wall",
                                  "ipsilateral"]
                DISTANT_SITES = ["liver", "lung", "bone", "brain", "pleural", "peritoneal",
                                 "ovary", "skin", "contralateral", "cervical", "distant",
                                 "hepatic", "pulmonary", "osseous", "cerebral"]
                has_distant = any(ds in met_lower for ds in DISTANT_SITES) if met_lower else False
                if not has_distant:
                    # Downgrade: replace Stage IV with Stage III or remove metastatic
                    cleaned = re.sub(r'(?i)\bStage\s*IV\s*\(?\s*metastatic\s*\)?', 'Stage III', stage)
                    cleaned = re.sub(r'(?i)\bmetastatic\s*\(?\s*Stage\s*IV\s*\)?', 'Stage III', cleaned)
                    cleaned = re.sub(r'(?i)\bStage\s*IV\b', 'Stage III', cleaned)
                    cleaned = cleaned.strip().rstrip(',').strip()
                    if cleaned and cleaned != stage:
                        cancer["Stage_of_Cancer"] = cleaned
                        print(f"    [POST-STAGE-DISTMET] Stage IV but Distant Met=No: '{stage}' → '{cleaned}'")

            # POST-STAGE-REGIONAL: Stage IV with only regional LN metastasis [v16] [breast-only]
            # Axillary, sentinel, supraclavicular, infraclavicular, internal mammary LN
            # are regional (Stage III) not distant (Stage IV) — breast cancer specific
            stage = cancer.get("Stage_of_Cancer", "")  # re-read in case POST-STAGE-DISTMET changed it
            stage_says_iv = bool(re.search(r'stage\s*iv|metastatic', stage, re.IGNORECASE))
            if cancer_type == "breast" and stage_says_iv and met and not met_says_no:
                met_lower = met.lower()
                REGIONAL_SITES = ["axillary", "axilla", "sentinel", "supraclavicular",
                                  "infraclavicular", "internal mammary", "chest wall",
                                  "ipsilateral"]
                DISTANT_SITES = ["liver", "lung", "bone", "brain", "pleural", "peritoneal",
                                 "ovary", "skin", "contralateral", "cervical", "distant",
                                 "hepatic", "pulmonary", "osseous", "cerebral"]
                has_regional = any(rs in met_lower for rs in REGIONAL_SITES)
                has_distant = any(ds in met_lower for ds in DISTANT_SITES)
                # Also check if note explicitly says "no distant metastasis"
                note_lower_stage = note_text.lower()
                note_no_distant = bool(re.search(
                    r'negative for distant|no distant metast|no evidence of distant|w/u negative',
                    note_lower_stage))
                if has_regional and not has_distant and note_no_distant:
                    cleaned = re.sub(r'(?i)\s*,?\s*(?:now\s+)?metastatic\s*\(?\s*Stage\s*IV\s*\)?', '', stage)
                    cleaned = re.sub(r'(?i)\bStage\s*IV\b', 'Stage III (regional)', cleaned)
                    cleaned = cleaned.strip().rstrip(',').strip()
                    if cleaned and cleaned != stage:
                        cancer["Stage_of_Cancer"] = cleaned
                        print(f"    [POST-STAGE-REGIONAL] Regional LN only + no distant mets: '{stage}' → '{cleaned}'")

        # POST-STAGE-VERIFY-ORIG: Check for hallucinated "Originally Stage X" [v14]
        # DISABLED in iter6: This hook was too aggressive — it removed legitimate
        # stage inferences when the note has redacted stage data (*****) or uses
        # pTN format instead of "Stage X". The model's stage inference is usually
        # reasonable and removing it causes more harm than keeping it.
        # cancer = keypoints.get("Cancer_Diagnosis", {})
        # ... (disabled)

        # POST-STAGE-VERIFY-NOTE: For non-breast cancers, verify extracted Stage actually appears in note [v32]
        if cancer_type != "breast":
            cancer_sv = keypoints.get("Cancer_Diagnosis", {})
            if isinstance(cancer_sv, dict):
                stage_sv = str(cancer_sv.get("Stage_of_Cancer", "") or "")
                # Extract specific stage designations from the extracted value
                stage_numbers = re.findall(r'Stage\s+(I{1,3}V?[ABC]?|IV|I[AB]|II[AB])', stage_sv, re.IGNORECASE)
                if stage_numbers:
                    note_lower_sv = (note_text or "").lower()
                    ap_lower_sv = (assessment_and_plan or "").lower()
                    all_text_sv = note_lower_sv + " " + ap_lower_sv
                    # Stage IV is definitionally equivalent to metastatic disease; if the
                    # value asserts metastatic/metastases, "Stage IV" is a valid clinical
                    # inference (a PL strength over the baseline), NOT a note-quote fabrication.
                    # Don't strip it, and don't substitute the original local pTN into the
                    # "now metastatic" slot (that produced nonsense like "metastatic (pT2N2)").
                    value_is_metastatic = bool(re.search(r'metasta', stage_sv, re.IGNORECASE))
                    for sn in stage_numbers:
                        if sn.upper().startswith("IV") and value_is_metastatic:
                            continue
                        sn_pattern = r'stage\s*' + re.escape(sn.lower())
                        if not re.search(sn_pattern, all_text_sv):
                            # Stage number not found in note — likely fabricated
                            old_stage_sv = stage_sv
                            # Replace fabricated stage with pTN if available
                            ptn_in_note = re.search(r'pT\d[a-d]?\s*N\d', all_text_sv, re.IGNORECASE)
                            if ptn_in_note:
                                replacement = ptn_in_note.group(0)
                                # keep an "Originally " prefix if present (only swap the stage token)
                                stage_sv = re.sub(r'(?i)Stage\s+' + re.escape(sn), replacement, stage_sv)
                            else:
                                stage_sv = re.sub(r'(?i)(?:Originally\s+)?Stage\s+' + re.escape(sn) + r',?\s*', '', stage_sv).strip()
                            if stage_sv != old_stage_sv:
                                cancer_sv["Stage_of_Cancer"] = stage_sv.strip(', ')
                                print(f"    [POST-STAGE-VERIFY-NOTE] Removed fabricated 'Stage {sn}': '{old_stage_sv}' → '{stage_sv}'")

        # POST-STAGE-VERIFY-ORIG: an "Originally Stage X" historical claim must be supported by the
        # note; an unsupported one is boilerplate fabrication. Targets ONLY the historical prefix,
        # so current-stage inference (POST-STAGE-INFER, breast staging table) is untouched. Runs
        # for all cancers — covers the breast path that POST-STAGE-VERIFY-NOTE skips. [2026-06-05, A4]
        cancer_vo = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_vo, dict):
            stage_vo = str(cancer_vo.get("Stage_of_Cancer", "") or "")
            orig_m = re.search(r'originally\s+stage\s+(IV|I{1,3}V?[ABC]?|I[AB]|II[AB])', stage_vo, re.IGNORECASE)
            if orig_m:
                orig_sn = orig_m.group(1)
                all_text_vo = ((note_text or "") + " " + (assessment_and_plan or "")).lower()
                if not re.search(r'stage\s*' + re.escape(orig_sn.lower()), all_text_vo):
                    new_vo = re.sub(r'(?i)originally\s+stage\s+' + re.escape(orig_sn) + r'\s*,?\s*', '', stage_vo)
                    new_vo = new_vo.strip().strip(',').strip()
                    cancer_vo["Stage_of_Cancer"] = new_vo if new_vo else "Not specified"
                    print(f"    [POST-STAGE-VERIFY-ORIG] unsupported 'Originally Stage {orig_sn}' removed: '{stage_vo}' → '{cancer_vo['Stage_of_Cancer']}'")

        # POST-STAGE-BILATERAL: bilateral breast cancer can carry two different stages. When the
        # physician's assessment spells out a stage per side ("Stage III (T3N1) ... left breast and
        # a Stage I (T1cN0) ... right breast"), report BOTH verbatim rather than collapsing to one
        # (and pre-empt the single-stage staging-table hooks from rewriting one of them). General
        # rule: explicit per-side staging in the note is the ground truth. [2026-06-06, bug1, breast5]
        cancer_bl = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer_bl, dict):
            ap_bl = assessment_and_plan or ""
            # capture "Stage <roman> (<TN>) ... <side> breast" — side within ~90 chars of the stage token
            bl_pairs = re.findall(
                r'stage\s+(IV|III[ABC]?|II[ABC]?|I[ABC]?)\s*\(([^)]*)\)[^.]{0,90}?\b(left|right)\s+breast',
                ap_bl, re.IGNORECASE)
            sides_bl = {}
            for st_bl, tn_bl, side_bl in bl_pairs:
                side_key = side_bl.capitalize()
                if side_key not in sides_bl:  # keep first occurrence per side
                    sides_bl[side_key] = f"Stage {st_bl.upper()} ({tn_bl.strip()})"
            if len(sides_bl) >= 2:
                new_bl = "; ".join(f"{side}: {sides_bl[side]}" for side in ("Left", "Right") if side in sides_bl)
                old_bl = str(cancer_bl.get("Stage_of_Cancer", "") or "")
                if new_bl and new_bl != old_bl:
                    cancer_bl["Stage_of_Cancer"] = new_bl
                    print(f"    [POST-STAGE-BILATERAL] '{old_bl}' → '{new_bl}'")

        # POST-STAGE-PLACEHOLDER: clean up [X] placeholders from redacted data [v18]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            stage = cancer.get("Stage_of_Cancer", "")
            if isinstance(stage, str) and re.search(r'\[X\]|\[REDACTED\]|\[\*+\]', stage):
                old_stage = stage
                cancer["Stage_of_Cancer"] = "Not available (redacted)"
                print(f"    [POST-STAGE-PLACEHOLDER] Replaced placeholder: '{old_stage}' → 'Not available (redacted)'")

        # POST-STAGE-ABBREV: Detect Stage abbreviations in A/P when Stage is empty/not mentioned [v23]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            stage = cancer.get("Stage_of_Cancer", "") or ""
            stage_lower = stage.lower().strip()
            stage_empty = not stage_lower or stage_lower in ("not mentioned", "not mentioned in note", "not available", "not available (redacted)", "")
            if stage_empty:
                ap_text_stage = (assessment_and_plan or '').lower()
                # Match: "St IV", "st II", "stage IV", "Clinical st II/III", "stage 2", etc.
                stage_abbrev = re.search(
                    r'(?:clinical\s+)?(?<![a-z])(?:stage|st\.?)\s+(i{1,3}v?|iv|[1-4])(?:\s*/\s*(i{1,3}v?|iv|[1-4]))?(?=\s|[,;:.\)]|$)(?!\d)',
                    ap_text_stage
                )
                if stage_abbrev:
                    raw_stage = stage_abbrev.group(0).strip().rstrip('.,;:')
                    # Normalize using captured groups: group(1) = main stage, group(2) = optional second stage
                    roman_map = {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}
                    part1 = stage_abbrev.group(1)
                    part1_norm = part1.upper() if part1.isalpha() else roman_map.get(part1, part1)
                    part2 = stage_abbrev.group(2)
                    if part2:
                        part2_norm = part2.upper() if part2.isalpha() else roman_map.get(part2, part2)
                        normalized = f"{part1_norm}/{part2_norm}"
                    else:
                        normalized = part1_norm
                    new_stage = f"Stage {normalized}"
                    cancer["Stage_of_Cancer"] = new_stage
                    print(f"    [POST-STAGE-ABBREV] Found stage abbreviation in A/P: '{raw_stage}' → '{new_stage}'")

        # POST-STAGE-INFER: If Stage is empty/unknown, infer from metastasis + tumor size + node status [breast-only staging table]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            stage = str(cancer.get("Stage_of_Cancer", "") or "")
            stage_lower = stage.lower().strip()
            stage_empty = not stage_lower or stage_lower in ("not mentioned", "not mentioned in note",
                                                              "not available", "not available (redacted)", "")
            if stage_empty:
                note_lower_s = note_text.lower() if note_text else ""

                # STEP 1: Check if metastatic disease → Stage IV (highest priority)
                dist_met = str(cancer.get("Distant Metastasis", "") or "").lower()
                met_field = str(cancer.get("Metastasis", "") or "").lower()
                # Check Distant Met field first (most reliable)
                has_distant_met = "yes" in dist_met or "yes" in met_field
                # If Distant Met explicitly says No/None, do NOT use note text to override
                dist_met_is_no = any(neg in dist_met for neg in ["no", "none", "negative"])
                is_metastatic = has_distant_met
                if not has_distant_met and not dist_met_is_no:
                    # Only check note if Distant Met is empty/unspecified
                    note_met_match = re.search(r'(?<!micro)metastatic|widely metastatic|metastases|stage\s*iv|stage\s*4',
                                               note_lower_s)
                    if note_met_match:
                        met_ctx = note_lower_s[max(0, note_met_match.start()-20):note_met_match.end()+20]
                        if any(excl in met_ctx for excl in ['micrometa', 'biopsy', 'originally', 'no evidence',
                                                             'negative for', 'without', 'no ', 'no definite',
                                                             'rule out', 'r/o', 'unlikely', 'not consistent']):
                            note_met_match = None
                    is_metastatic = note_met_match is not None

                if is_metastatic:
                    cancer["Stage_of_Cancer"] = "Stage IV (metastatic)"
                    print(f"    [POST-STAGE-INFER] Metastatic disease detected → Stage IV")
                # STEP 2 removed (v32): tumor size → AJCC stage inference was too error-prone
                # (50% accuracy on test set, including Stage I vs expert IIIC).
                # If the note doesn't state a stage and isn't metastatic, leave as "Not staged in note".

        # POST-STAGE-CTNM: when no stage is stated, a clinical/pathologic TNM in the assessment
        # (e.g. "clinical T2NX", "cT2NX", "pT3N1") IS the staging information the note provides —
        # capture it instead of punting to "Not mentioned". Runs after metastatic inference (so
        # Stage IV still wins) and before the pTN→Stage translator. General oncology rule. [2026-06-06, bug2, breast18]
        cancer_ct = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_ct, dict):
            stage_ct = str(cancer_ct.get("Stage_of_Cancer", "") or "").strip()
            sl_ct = stage_ct.lower()
            empty_ct = (not sl_ct) or sl_ct in ("not mentioned", "not mentioned in note",
                                                "not available", "not available (redacted)",
                                                "not specified", "not staged in note",
                                                "not specified in the note", "")
            if empty_ct:
                for src_ct in (assessment_and_plan or "", note_text or ""):
                    m_ct = re.search(r'\b(?:(clinical|pathologic|path)\s+)?(c|p|yp)?T(\d)([a-d]?)\s*,?\s*N([0-3X])([a-c]?)',
                                     src_ct, re.IGNORECASE)
                    if m_ct:
                        clin_word, pfx_ct, t_ct, tsfx_ct, n_ct, nsfx_ct = m_ct.groups()
                        pfx_ct = (pfx_ct or "").lower()
                        if not pfx_ct and clin_word and clin_word.lower().startswith("clinic"):
                            pfx_ct = "c"
                        tnm_ct = f"{pfx_ct}T{t_ct}{tsfx_ct or ''}N{n_ct.upper()}{nsfx_ct or ''}"
                        qual_ct = " (clinical staging)" if pfx_ct == "c" else ""
                        cancer_ct["Stage_of_Cancer"] = tnm_ct + qual_ct
                        print(f"    [POST-STAGE-CTNM] filled empty stage from TNM in note: '{tnm_ct}{qual_ct}'")
                        break

        # POST-STAGE-PTNM-VERIFY: a pTNM/ypTNM in the stage field must match the pathology TNM the
        # note actually states. The model occasionally transposes digits ("ypT 3N2" for a note-stated
        # "pT2N3"). When the assessment/note states an explicit p/yp-TNM that differs, transcribe the
        # note's verbatim. Non-breast only (breast has its own pTN→Stage translator). [2026-06-06, bug9, pdac15]
        cancer_pv = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type != "breast" and isinstance(cancer_pv, dict):
            stage_pv = str(cancer_pv.get("Stage_of_Cancer", "") or "")
            val_tnm = re.search(r'(?:yp|p|c)?T\s*\d[a-d]?\s*N\s*\d[a-c]?', stage_pv, re.IGNORECASE)
            if val_tnm:
                src_pv = (assessment_and_plan or "") + " " + (note_text or "")
                note_tnm = re.search(r'\b(yp|p)T(\d)([a-d]?)\s*,?\s*N(\d)([a-c]?)', src_pv, re.IGNORECASE)
                if note_tnm:
                    note_str = (f"{(note_tnm.group(1) or '').lower()}T{note_tnm.group(2)}"
                                f"{note_tnm.group(3) or ''}N{note_tnm.group(4)}{note_tnm.group(5) or ''}")
                    norm = lambda s: re.sub(r'\s', '', s).lower()
                    if norm(val_tnm.group(0)) != norm(note_str):
                        old_pv = stage_pv
                        new_pv = re.sub(r'(?:yp|p|c)?T\s*\d[a-d]?\s*N\s*\d[a-c]?', note_str,
                                        stage_pv, count=1, flags=re.IGNORECASE)
                        cancer_pv["Stage_of_Cancer"] = new_pv
                        print(f"    [POST-STAGE-PTNM-VERIFY] '{old_pv}' → '{new_pv}' (note states {note_str})")

        # POST-STAGE-PTN-TRANSLATE: If Stage field contains only pTN notation, translate to Stage name [breast-only staging table]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            stage = str(cancer.get("Stage_of_Cancer", "") or "")
            # Match pure pTN like "pT3N0", "pT2N1(sn)", "pT1c(m)N1(sn)" without a "Stage" word
            if stage and 'stage' not in stage.lower() and re.match(r'^p?T\d', stage, re.IGNORECASE):
                ptn = re.match(r'p?T(\d)([a-d]?)(?:\([^)]*\))?\s*,?\s*p?N(\d)([a-c]?(?:mi)?)?', stage, re.IGNORECASE)
                if ptn:
                    t_num = int(ptn.group(1))
                    n_num = int(ptn.group(3))
                    n_suffix = (ptn.group(4) or "").lower()
                    # Translate to AJCC stage
                    if n_suffix == 'mi':
                        n_positive = 0  # micromet only
                        is_micromet = True
                    else:
                        n_positive = n_num
                        is_micromet = False

                    if t_num <= 1 and n_positive == 0:
                        stage_name = "Stage IA" if not is_micromet else "Stage IB"
                    elif t_num <= 1 and n_positive >= 1 and n_positive <= 3:
                        stage_name = "Stage IIA"
                    elif t_num == 2 and n_positive == 0:
                        stage_name = "Stage IIA"
                    elif t_num == 2 and n_positive >= 1 and n_positive <= 3:
                        stage_name = "Stage IIB"
                    elif t_num >= 3 and n_positive == 0:
                        stage_name = "Stage IIIA"
                    elif t_num >= 3 and n_positive >= 1:
                        stage_name = "Stage IIIA"
                    elif n_positive >= 4:
                        stage_name = "Stage IIIA"
                    else:
                        stage_name = "Stage II"

                    new_stage = f"{stage_name} ({stage})"
                    cancer["Stage_of_Cancer"] = new_stage
                    print(f"    [POST-STAGE-PTN-TRANSLATE] {stage} → {new_stage}")

        # POST-STAGE-CORRECT: Correct wrong Stage based on tumor size + node count in note [breast-only staging table]
        # LLM sometimes writes "Stage III" when pT2N1 (1-3 nodes) → should be Stage IIB
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            stage = str(cancer.get("Stage_of_Cancer", "") or "")
            stage_lower = stage.lower().strip()
            # Verify "stage iii", "stage 3", or "stage iiia" (possibly with extra text) when node data suggests lower
            # GUARD [bug1, breast5]: never "correct" a stage the physician explicitly wrote. If the
            # note/A/P states "Stage III" (or "Stage 3") verbatim, the LLM is faithfully transcribing
            # it (e.g. a T3N1 = IIIA the staging table can't reproduce from a misparsed tumor size) —
            # downgrading it to IIB is over-inference. Only auto-correct when the stage was inferred.
            note_ap_explicit_v = ((note_text or "") + " " + (assessment_and_plan or "")).lower()
            note_states_iii = bool(re.search(r'stage\s*(iii[a-c]?|3)\b', note_ap_explicit_v))
            if re.match(r'^stage\s*(iii[a-c]?|3)\b', stage_lower) and not note_states_iii:
                note_lower_v = note_text.lower() if note_text else ""
                # Look for node count in note
                node_match = re.search(r'(\d+)/(\d+)\s*(?:nodes?|LN|sentinel|lymph)', note_lower_v)
                # Look for tumor size
                tumor_match = re.search(r'(\d+\.?\d*)\s*(?:cm|mm)\s*(?:tumor|mass|IDC|ILC|carcinoma|invasive)',
                                       note_lower_v, re.IGNORECASE)
                if not tumor_match:
                    tumor_match = re.search(r'(?:tumor|mass|IDC|ILC|carcinoma|invasive)[^.]{0,30}(\d+\.?\d*)\s*(?:cm|mm)',
                                           note_lower_v, re.IGNORECASE)

                # Also try to extract pTN from the stage text itself (e.g., "Stage IIIA (inferred from pT2 N1mi)")
                ptn_in_stage = re.search(r'p?T(\d)\s*N(\d)([a-c]?(?:mi)?)', stage, re.IGNORECASE)

                n_pos = None
                n_is_micro = False
                if node_match:
                    n_pos = int(node_match.group(1))
                elif ptn_in_stage:
                    n_class = int(ptn_in_stage.group(2))  # N classification (0,1,2,3)
                    n_suffix = (ptn_in_stage.group(3) or "").lower()
                    n_is_micro = n_suffix == 'mi'
                    # N classification → approximate node count
                    # N0=0, N1/N1mi=1-3, N2=4-9, N3=10+
                    if n_class >= 2:
                        n_pos = 4  # N2+ means 4+ nodes → IIIA is correct, don't correct
                    elif n_is_micro:
                        n_pos = 0  # N1mi = micromet only
                    else:
                        n_pos = n_class  # N0=0, N1=1

                if n_pos is not None and n_pos <= 3:  # N1 (1-3 nodes) or N1mi → not Stage III
                    size = None
                    if tumor_match:
                        size = float(tumor_match.group(1))
                        if 'mm' in tumor_match.group(0).lower():
                            size = size / 10
                    elif ptn_in_stage:
                        t_val = int(ptn_in_stage.group(1))
                        if t_val <= 1:
                            size = 1.5  # approximate T1
                        elif t_val == 2:
                            size = 3.0  # approximate T2

                    if n_is_micro:
                        # N1mi (micromet only): T1→IB, T2→IIA
                        if size and size <= 2.0:
                            corrected = f"Stage IB (corrected: pT1 N1mi — micrometastasis only)"
                        else:
                            corrected = f"Stage IIA (corrected: pT2 N1mi — micrometastasis only)"
                    elif size and size <= 2.0:
                        corrected = f"Stage IIA (corrected: pT1 N1, {n_pos} positive node{'s' if n_pos > 1 else ''})"
                    elif size and size <= 5.0:
                        corrected = f"Stage IIB (corrected: pT2 N1, {n_pos} positive node{'s' if n_pos > 1 else ''})"
                    else:
                        corrected = f"Stage IIB (corrected: N1 with {n_pos} positive node{'s' if n_pos > 1 else ''})"

                    cancer["Stage_of_Cancer"] = corrected
                    print(f"    [POST-STAGE-CORRECT] {stage} → {corrected}")

        # POST-STAGE-NOBASIS: a numeric AJCC stage (I-IV) must be anchored to evidence the
        # note actually gives for THIS cancer — otherwise it is fabrication (b4: a cT2N0,
        # non-metastatic breast note whose only "stage" text is an unrelated ovarian-cancer
        # history "stage IIIC", yet the model emitted "Originally unspecified, now Stage III").
        # Anchors that count: the value is metastatic (Stage IV), the A/P literally states a
        # stage, or a TNM token (cT/pT/T#N#) appears in the A/P, in the value itself (e.g.
        # bilateral "Stage III (T3N1)"), or in the note body (TNM is staging-specific and
        # almost always names the primary under discussion). We deliberately do NOT scan the
        # full note for "stage X" text — PMH of other cancers (ovarian/lung/etc.) contaminates
        # that. Runs LAST among the stage hooks so legitimate inferences (metastatic IV, CTNM,
        # PTN-translate, bilateral, A/P abbrev) — which all carry one of these anchors —
        # survive. General oncology rule, not test-set-specific. [2026-06-06, fix#1, b4]
        cancer_nb = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_nb, dict):
            stage_nb = str(cancer_nb.get("Stage_of_Cancer", "") or "")
            num_m_nb = re.search(r'stage\s+(IV|III[ABC]?|II[ABC]?|I[ABC]?)\b', stage_nb, re.IGNORECASE)
            if num_m_nb:
                val_low_nb = stage_nb.lower()
                ap_low_nb = (assessment_and_plan or "").lower()
                note_low_nb = (note_text or "").lower()
                tnm_re_nb = r'(?:c|p|yp|yc)?t[0-4]x?\s*,?\s*n[0-3x]'
                supported_nb = (
                    'metasta' in val_low_nb                                  # Stage IV = metastatic
                    or bool(re.search(r'stage\s*(?:iv|iii|ii|i|[1-4])\b', ap_low_nb))  # A/P states a stage
                    or bool(re.search(tnm_re_nb, ap_low_nb))                 # TNM in A/P
                    or bool(re.search(tnm_re_nb, val_low_nb))               # TNM carried in value
                    or bool(re.search(tnm_re_nb, note_low_nb))             # TNM in note body
                )
                if not supported_nb:
                    old_nb = stage_nb
                    cancer_nb["Stage_of_Cancer"] = "Not staged in note"
                    print(f"    [POST-STAGE-NOBASIS] no stage/TNM/metastatic anchor — '{old_nb}' → 'Not staged in note'")

        # POST-STAGE-RECURRENCE: If A/P mentions local recurrence but Stage doesn't [iter10]
        # Only for non-metastatic (don't append to Stage IV)
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            stage = str(cancer.get("Stage_of_Cancer", "") or "")
            stage_lower = stage.lower()
            is_metastatic = 'iv' in stage_lower or 'metastatic' in stage_lower
            if stage and not is_metastatic and 'recurrence' not in stage_lower and 'relapse' not in stage_lower:
                ap_lower_sr = (assessment_and_plan or "").lower()
                recurrence = re.search(r'(?:local\s+(?:recurrence|relapse)|second\s+local\s+relapse)(?!\s+risk)', ap_lower_sr)
                # Exclude "risk of recurrence", "recurrence risk", "reduce recurrence"
                if recurrence:
                    ctx = ap_lower_sr[max(0,recurrence.start()-30):recurrence.end()+30]
                    if any(x in ctx for x in ['risk of', 'risk for', 'reduce', 'decrease', 'minimize']):
                        recurrence = None
                if recurrence:
                    old = stage
                    cancer["Stage_of_Cancer"] = stage + ", now with local recurrence"
                    print(f"    [POST-STAGE-RECURRENCE] {old} → {cancer['Stage_of_Cancer']}")

        # POST-GOALS: adjuvant → curative for non-metastatic [B45]
        goals = keypoints.get("Treatment_Goals", {})
        if isinstance(goals, dict):
            goal_val = str(goals.get("goals_of_treatment", "") or "").lower().strip()
            if goal_val == "adjuvant":
                cancer = keypoints.get("Cancer_Diagnosis", {})
                met = str(cancer.get("Metastasis", "")).lower() if isinstance(cancer, dict) else ""
                stage = str(cancer.get("Stage_of_Cancer", "")).lower() if isinstance(cancer, dict) else ""
                is_metastatic = "yes" in met or "stage iv" in stage or "metastatic" in stage
                if not is_metastatic:
                    goals["goals_of_treatment"] = "curative"
                    print(f"    [POST-GOALS] adjuvant → curative (non-metastatic)")

        # POST-GOALS-SURVEILLANCE: If A/P says "surveillance" and no active treatment, override goal [v32]
        goals_surv = keypoints.get("Treatment_Goals", {})
        if isinstance(goals_surv, dict):
            goal_val_s = str(goals_surv.get("goals_of_treatment", "") or "").lower().strip()
            if goal_val_s in ("curative", "adjuvant"):
                ap_lower_gs = (assessment_and_plan or "").lower()
                meds_val_gs = str(keypoints.get("Current_Medications", {}).get("current_meds", "") or "").strip()
                # Check if A/P explicitly says surveillance AND no active cancer meds
                has_surveillance = bool(re.search(
                    r'continue\s+(?:on\s+)?surveillance|will\s+(?:continue|remain)\s+(?:on\s+)?surveillance|'
                    r'plan\s+for\s+(?:\w+\s+)?surveillance|f/?u\s+surveillance|follow.up\s+surveillance|'
                    r'surveillance\s+(?:imaging|scan|ct|plan|strategy|include|with)|'
                    r'continue\s+to\s+monitor|monitoring/expectant|expectant\s+management|'
                    r'course\s+of\s+monitoring|will\s+(?:continue\s+to\s+)?monitor\s+(?:\w+\s+)?with\s+(?:once|annual|regular)',
                    ap_lower_gs))
                has_active_meds = bool(meds_val_gs)
                if has_surveillance and not has_active_meds:
                    old_goal = goals_surv["goals_of_treatment"]
                    goals_surv["goals_of_treatment"] = "surveillance"
                    print(f"    [POST-GOALS-SURVEILLANCE] {old_goal} → surveillance (A/P says surveillance, no active meds)")

        # POST-DISTMET: Ensure Distant Metastasis field exists [B48]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict) and "Distant Metastasis" not in cancer:
            met = cancer.get("Metastasis", "")
            cancer["Distant Metastasis"] = met
            print(f"    [POST-DISTMET] added Distant Metastasis: '{met}'")

        # POST-DISTMET-NOMET: If imaging says "no metastatic disease" but LLM extracted "Yes", correct [iter8]
        # ONLY trigger when goals=curative/adjuvant (palliative = likely truly metastatic)
        cancer = keypoints.get("Cancer_Diagnosis", {})
        goals_for_dm = str(keypoints.get("Treatment_Goals", {}).get("goals_of_treatment", "") if isinstance(keypoints.get("Treatment_Goals"), dict) else "").lower()
        if isinstance(cancer, dict):
            dist_met = str(cancer.get("Distant Metastasis", "") or "").lower()
            if "yes" in dist_met and goals_for_dm in ("curative", "adjuvant", "risk reduction"):
                no_met_pattern = (r'no\s+(?:definite\s+)?(?:sites?\s+of\s+)?(?:hypermetabolic\s+)?metastatic\s+disease|'
                                  r'no\s+evidence\s+of\s+(?:distant\s+)?metastases|'
                                  r'no\s+distant\s+metastases|'
                                  r'negative\s+for\s+(?:distant\s+)?metastatic\s+disease')
                # Search A/P first, but exclude historical references (dates before 2020)
                ap_lower_dm = (assessment_and_plan or "").lower()
                no_met_evidence = re.search(no_met_pattern, ap_lower_dm)
                if no_met_evidence:
                    # Check if this is a historical reference (near a past date)
                    ctx_before = ap_lower_dm[max(0, no_met_evidence.start()-80):no_met_evidence.start()]
                    if re.search(r'20[01]\d|s/p|history|previously|prior|in \d{4}', ctx_before):
                        no_met_evidence = None  # historical, don't use
                # If not found in A/P, search imaging results in note (not HPI history)
                if not no_met_evidence and note_text:
                    note_lower_dm = note_text.lower()
                    # Search for "no [definite] metastatic disease" anywhere in note,
                    # but verify it's from imaging/staging, not from HPI history
                    for nm in re.finditer(no_met_pattern, note_lower_dm):
                        ctx_before = note_lower_dm[max(0, nm.start()-200):nm.start()]
                        # Accept if near imaging keywords or PET/CT, reject if in HPI history
                        is_imaging = any(k in ctx_before for k in ['pet', 'ct ', 'scan', 'impression',
                                                                     'findings', 'staging', 'no evidence'])
                        is_history = any(k in ctx_before for k in ['history of present',
                                                                     'hpi', 'in 201', 'in 200',
                                                                     'previously', 'prior to'])
                        if is_imaging and not is_history:
                            no_met_evidence = nm
                            break
                if no_met_evidence:
                    old_dm = cancer.get("Distant Metastasis", "")
                    cancer["Distant Metastasis"] = "No"
                    if cancer.get("Metastasis"):
                        cancer["Metastasis"] = "No"
                    print(f"    [POST-DISTMET-NOMET] A/P says '{no_met_evidence.group(0)}' — corrected: '{old_dm}' → 'No'")

        # POST-DISTMET-REGIONAL: correct Distant Metastasis if only regional sites [v17] [breast-only]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            dist_met = cancer.get("Distant Metastasis", "") or ""
            if dist_met and dist_met.lower() not in ("no", "no.", "none", ""):
                dist_lower = dist_met.lower()
                REGIONAL_SITES_DM = ["axillary", "axilla", "sentinel", "supraclavicular",
                                     "infraclavicular", "internal mammary", "chest wall", "ipsilateral"]
                DISTANT_SITES_DM = ["liver", "lung", "bone", "brain", "pleural", "peritoneal",
                                    "ovary", "skin", "adrenal", "contralateral",
                                    "cervical", "distant", "hepatic", "pulmonary",
                                    "osseous", "cerebral", "sternum", "sternal",
                                    "spine", "spinal", "rib", "hip", "femur", "pelvi",
                                    "mediastin", "retroperitoneal", "paraaortic", "para-aortic",
                                    "mesenteric", "inguinal", "scalene"]
                has_regional = any(rs in dist_lower for rs in REGIONAL_SITES_DM)
                has_distant = any(ds in dist_lower for ds in DISTANT_SITES_DM)
                if has_regional and not has_distant:
                    cancer["Distant Metastasis"] = "No"
                    print(f"    [POST-DISTMET-REGIONAL] Corrected Distant Metastasis: regional only → No (was: '{dist_met}')")

        # POST-DISTMET-SUPPLEMENT: If Distant Met says "Yes" but misses lymph nodes from A/P [iter10]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            dist_met = str(cancer.get("Distant Metastasis", "") or "")
            if "yes" in dist_met.lower() and "lymph" not in dist_met.lower() and "node" not in dist_met.lower():
                ap_lower_dm = (assessment_and_plan or "").lower()
                note_lower_dm = note_text.lower() if note_text else ""
                # Check if A/P or note mentions lymph node metastasis
                ln_met = re.search(r'(?:metastatic?\s+(?:to\s+)?(?:.*?)?lymph\s*nodes?|'
                                   r'lymph\s*node\s+metastas|'
                                   r'disease\s+in\s+(?:the\s+)?(?:bone\s+and\s+)?lymph\s*nodes?|'
                                   r'metastatic\s+(?:recurrence|disease)\s+(?:.*?)?(?:lymph|nodes)|'
                                   r'(?:bone|liver)\s+and\s+lymph\s*nodes?|'
                                   r'lymph\s*nodes?\s+(?:and|,)\s+(?:bone|liver|lung))',
                                   ap_lower_dm)
                if ln_met:
                    old_dm = dist_met
                    cancer["Distant Metastasis"] = dist_met.rstrip('.') + " and lymph nodes"
                    print(f"    [POST-DISTMET-SUPPLEMENT] Added lymph nodes: '{old_dm}' → '{cancer['Distant Metastasis']}'")

        # POST-DISTMET-DEFAULT: fill empty Distant Metastasis with "No" when goals=curative [v22]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            dist_met_val = (cancer.get("Distant Metastasis", "") or "").strip()
            goals_val = keypoints.get("Treatment_Goals", {}).get("goals_of_treatment", "") if isinstance(keypoints.get("Treatment_Goals"), dict) else ""
            stage_val = (cancer.get("Stage_of_Cancer", "") or "").lower()
            if not dist_met_val and goals_val == "curative" and "iv" not in stage_val and "metastatic" not in stage_val:
                cancer["Distant Metastasis"] = "No"
                print(f"    [POST-DISTMET-DEFAULT] Filled empty Distant Metastasis → 'No' (curative, non-metastatic)")

        # POST-DISTMET-PENDING: when the assessment says staging imaging is still being obtained to
        # look for metastasis (not yet resulted), distant-met status is genuinely unknown — "No"
        # overstates a workup that hasn't happened. Mark it pending. General rule: pending workup ≠
        # negative workup. Runs after DISTMET-DEFAULT so it can correct a premature "No". [2026-06-06, bug3, breast1]
        cancer_pd = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_pd, dict):
            dm_pd = str(cancer_pd.get("Distant Metastasis", "") or "").strip().lower()
            stage_pd = str(cancer_pd.get("Stage_of_Cancer", "") or "").lower()
            not_metastatic_pd = ("iv" not in stage_pd and "metastatic" not in stage_pd)
            dm_is_no_or_empty = (dm_pd in ("", "no", "no.", "none") or dm_pd.startswith("no "))
            if not_metastatic_pd and dm_is_no_or_empty:
                ap_pd = (assessment_and_plan or "").lower()
                pending_staging = re.search(
                    r'staging\s+(imaging|scans?|work[\s-]?up|ct|pet)[^.]{0,50}(metasta|spread|assess|stage|distant)|'
                    r'(obtain|order|recommend|will\s+(get|obtain|order)|role of|plan(?:ning)?\s+(?:for|to (?:get|obtain)))'
                    r'[^.]{0,40}(pet[\s/]*ct|ct\s+(?:chest|c/?a/?p|of the chest)|bone scan|staging)[^.]{0,40}(metasta|assess|stage|spread)|'
                    r'(imaging|pet[\s/]*ct|scans?)\s+to\s+(assess|evaluate|look)[^.]{0,25}(metasta|spread|for distant)',
                    ap_pd)
                completed_neg = re.search(
                    r'no evidence of (distant\s+)?metasta|staging[^.]{0,20}negative|negative for (distant\s+)?metasta|'
                    r'(w/?u|workup)\s+negative|no distant (disease|metasta)', ap_pd)
                if pending_staging and not completed_neg:
                    cancer_pd["Distant Metastasis"] = "Not sure (staging imaging pending)"
                    print(f"    [POST-DISTMET-PENDING] staging imaging pending → 'Not sure (staging imaging pending)'")

        # POST-LOCOREGIONAL: a recurrence at the primary site or regional nodes is NOT distant
        # metastasis. When the physician's own assessment classifies the disease as a
        # local-regional recurrence (and no biopsy-confirmed distant met), don't let the model
        # inflate it to Stage IV / distant met. General oncology rule, applies to all cancers. [2026-06-05, A2]
        cancer_lr = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_lr, dict):
            ap_only_lr = (assessment_and_plan or "").lower()          # CURRENT assessment only
            note_ap_lr = ap_only_lr + " " + (note_text or "").lower()  # full text (for distant-met guard)
            stage_lr = str(cancer_lr.get("Stage_of_Cancer", "") or "")
            dm_lr = str(cancer_lr.get("Distant Metastasis", "") or "")
            m_lr = str(cancer_lr.get("Metastasis", "") or "")
            # (b) locoregional-recurrence override. CRITICAL: detect the recurrence phrase in the
            # CURRENT assessment (A/P) ONLY — the full note frequently carries a *historical* "local
            # recurrence" heading for a patient who has since progressed to metastatic disease
            # (e.g. breast ROW7). Searching the whole note would wrongly downgrade those P0.
            locoregional = re.search(r'local[\s-]*regional recurrence|locoregional recurrence|'
                                     r'local recurrence(?!\w)', ap_only_lr)
            # Block the override if ANYWHERE in the note distant/metastatic disease is confirmed.
            confirmed_distant = re.search(
                r'biopsy[^.]{0,50}metasta|metasta[^.]{0,30}biopsy|newly diagnosed metastatic|'
                r'metastatic\s+(\w+\s+){0,3}(to|in)\s+(the\s+)?'
                r'(bone|brain|lung|liver|node|hepat|pulmon|osseous|vertebr|spine|adrenal|peritone)|'
                r'metastatic\s+\w+\s+(cancer|carcinoma|adenocarcinoma)\b|'
                r'(biopsy[\s-]*proven|biopsy[\s-]*confirmed|confirmed)\s+(distant\s+)?metasta',
                note_ap_lr)
            stage_says_iv = bool(re.search(r'stage\s*iv|metastatic', stage_lr, re.IGNORECASE))
            dm_says_yes = "yes" in dm_lr.lower()
            if locoregional and (stage_says_iv or dm_says_yes) and not confirmed_distant:
                new_stage_lr = "Locally recurrent (unresectable)" if "unresectable" in ap_only_lr else "Locally recurrent"
                cancer_lr["Stage_of_Cancer"] = new_stage_lr
                cancer_lr["Distant Metastasis"] = "No"
                cancer_lr["Metastasis"] = "No"
                print(f"    [POST-LOCOREGIONAL] A/P says local-regional recurrence (no confirmed distant) → Stage '{stage_lr}'→'{new_stage_lr}', met→No")
            else:
                # (c) primary-organ site named as a "metastasis" = local recurrence, not a met.
                #     Only when DistMet is already an evidence-based No and no true distant organ is listed.
                primary_organ = "breast" if cancer_type == "breast" else "pancrea"
                m_low = m_lr.lower()
                if (dm_lr.strip().lower() in ("no", "no.", "none") and "yes" in m_low
                        and primary_organ in m_low
                        and not any(s in m_low for s in ["liver", "hepatic", "lung", "pulmonary", "bone",
                                                          "osseous", "brain", "cerebral", "peritone", "adrenal",
                                                          "contralateral", "spine", "spinal"])):
                    cancer_lr["Metastasis"] = "No"
                    print(f"    [POST-LOCOREGIONAL] primary-site recurrence, not a met: Metastasis '{m_lr}'→'No' (DistMet already No)")

        # POST-STAGE-SUSPECTED: hedged imaging ("suspicious for"/"suggestive of"/"concerning for"/
        # "early evidence of"/"cannot exclude") metastatic disease is NOT a confirmed Stage IV.
        # If the only met evidence is hedged and nothing is biopsy-proven/confirmed, mark it
        # Suspected rather than confirmed. General radiology/oncology rule. [2026-06-05, A3]
        cancer_su = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_su, dict):
            stage_su = str(cancer_su.get("Stage_of_Cancer", "") or "")
            if re.search(r'stage\s*iv|metastatic', stage_su, re.IGNORECASE) and "suspect" not in stage_su.lower():
                # Scope to the CURRENT assessment (A/P), NOT the full note: HPI history often
                # carries an old "metastatic carcinoma" (e.g. a regional LN biopsy at original dx)
                # that would wrongly satisfy the 'confirmed' guard and block a valid downgrade of a
                # currently-only-suspected met (e.g. breast ROW9). [2026-06-05, A3 scope fix]
                all_text_su = (assessment_and_plan or "").lower()
                hedged = re.search(r'(suspicious for|suggestive of|concerning for|worrisome for|'
                                   r'early evidence of|cannot exclude|cannot be excluded|equivocal|'
                                   r'may represent|likely represents?)[^.]{0,45}(metasta|recurren|disease|lesion|nodul)',
                                   all_text_su)
                # A confirmed metastatic DIAGNOSIS overrides hedged imaging language. Cover
                # "metastatic <organ> adenocarcinoma/cancer", carcinomatosis, omental caking,
                # and biopsy/FNA confirmation.
                confirmed = re.search(r'biopsy[\s-]*(proven|confirmed)|fna[^.]{0,40}(metasta|adenocarc|malignan)|'
                                      r'(confirmed|known|definite|biopsy|diagnosed)[^.]{0,25}metasta|'
                                      r'metastatic\s+(\w+\s+){0,2}(adenocarcinoma|carcinoma|cancer)|'
                                      r'metastatic\s+(disease\s+)?(confirmed|present)|'
                                      r'consistent with metastatic|carcinomatosis|omental caking', all_text_su)
                if hedged and not confirmed:
                    # Replace the confirmed met/Stage-IV assertion with a clean suspected label,
                    # preserving any "Originally ..." historical prefix (no nested parentheses).
                    m_orig_su = re.match(r'(?i)(originally[^,]*,\s*)', stage_su)
                    prefix_su = m_orig_su.group(1) if m_orig_su else ""
                    cancer_su["Stage_of_Cancer"] = prefix_su + "Suspected Stage IV (pending confirmation)"
                    if "yes" in str(cancer_su.get("Distant Metastasis", "")).lower():
                        cancer_su["Distant Metastasis"] = "Not sure"
                    if "yes" in str(cancer_su.get("Metastasis", "")).lower():
                        cancer_su["Metastasis"] = "Not sure"
                    print(f"    [POST-STAGE-SUSPECTED] hedged met evidence (no confirmation) → '{stage_su}' → '{cancer_su['Stage_of_Cancer']}'")

        # POST-METASTATIC-UPGRADE: the mirror of POST-STAGE-SUSPECTED. The model sometimes UNDER-calls
        # confirmed metastatic disease (leaves Stage < IV and met "Not sure"/"No") even when the current
        # assessment states an unambiguous distant-met finding — peritoneal carcinomatosis, omental
        # caking, or a biopsy-confirmed distant met. Those are definitionally Stage IV; upgrade.
        # Negation-guarded so "no carcinomatosis" never triggers. General oncology rule. [2026-06-06, bug6, pdac12]
        cancer_mu = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_mu, dict):
            stage_mu = str(cancer_mu.get("Stage_of_Cancer", "") or "")
            dm_mu = str(cancer_mu.get("Distant Metastasis", "") or "")
            m_mu = str(cancer_mu.get("Metastasis", "") or "")
            already_iv_mu = bool(re.search(r'stage\s*iv|metastatic|suspected', stage_mu, re.IGNORECASE))
            txt_mu = ((assessment_and_plan or "") + " " + (note_text or "")).lower()
            # A hedge/negation ANYWHERE in the ~32 chars before the marker disqualifies that mention
            # ("possibility of peritoneal carcinomatosis", "concerning for ... carcinomatosis",
            # "no carcinomatosis"). We scan ALL mentions and upgrade only if at least one is clean —
            # so pdac12 (note states "unchanged peritoneal carcinomatosis" definitively) upgrades,
            # while pdac6 (only "possibility of ...", laparoscopy unremarkable) does not. [bug6 guard]
            HEDGE_MU = re.compile(
                r'\b(?:no|not|without|negative for|r/o|rule out|resolv\w*|resolution|possib\w*|'
                r'concern\w*|suspicious|suspect\w*|question\w*|may|might|likely|could|evaluat\w*|'
                r'assess\w*|differential|cannot exclude|worrisome|if )\b')
            marker_mu = None
            for pat_mu, site_mu in [
                (r'peritoneal carcinomatosis', 'peritoneal carcinomatosis'),
                (r'\bcarcinomatosis\b', 'carcinomatosis'),
                (r'omental cak(?:e|ing)', 'omental caking'),
                (r'biopsy[\s-]*(?:proven|confirmed)[^.]{0,30}metasta', 'biopsy-confirmed metastasis'),
            ]:
                for mm_mu in re.finditer(pat_mu, txt_mu):
                    ctx_mu = txt_mu[max(0, mm_mu.start() - 32):mm_mu.start()]
                    if not HEDGE_MU.search(ctx_mu):
                        marker_mu = site_mu
                        break
                if marker_mu:
                    break
            if marker_mu:
                # confirmed metastatic. Upgrade the stage unless it is ALREADY a confirmed (not merely
                # "suspected") Stage IV, and ALWAYS reconcile the met fields up to "Yes" — the model
                # sometimes states Stage IV but still hedges the met fields to "Not sure"/"Suspected"
                # (which a later reconcile rule then mangles). [bug6 + pdac12 met-field fix]
                old_stage_mu = stage_mu
                stage_low_mu = stage_mu.lower()
                is_confirmed_iv = ("suspect" not in stage_low_mu) and bool(re.search(r'stage\s*iv|metastatic', stage_low_mu))
                if not is_confirmed_iv:
                    cancer_mu["Stage_of_Cancer"] = "Stage IV (metastatic)"
                site_label_mu = "Yes (peritoneal carcinomatosis)" if "periton" in marker_mu else \
                                ("Yes (omental/peritoneal)" if "omental" in marker_mu else "Yes")
                if "yes" not in dm_mu.lower():
                    cancer_mu["Distant Metastasis"] = site_label_mu
                if "yes" not in m_mu.lower():
                    cancer_mu["Metastasis"] = site_label_mu
                print(f"    [POST-METASTATIC-UPGRADE] confirmed {marker_mu} → Stage IV + distant met Yes (stage was '{old_stage_mu}')")

        # POST-STAGE-METASTATIC: If metastasis=Yes but Stage says "Not available"/"Not mentioned", set Stage IV [v24]
        cancer_diag = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_diag, dict):
            stage_val = cancer_diag.get("Stage_of_Cancer", "")
            met_val = cancer_diag.get("Metastasis", "")
            dist_met_val = cancer_diag.get("Distant Metastasis", "")
            if stage_val and any(x in stage_val.lower() for x in ["not available", "not mentioned"]):
                if (isinstance(met_val, str) and "yes" in met_val.lower()) or \
                   (isinstance(dist_met_val, str) and "yes" in dist_met_val.lower()):
                    cancer_diag["Stage_of_Cancer"] = "Stage IV (metastatic)"
                    print(f"    [POST-STAGE-METASTATIC] '{stage_val}' → 'Stage IV (metastatic)' (Metastasis=Yes)")

        # POST-MET-RECONCILE: keep the two redundant met fields (Distant Metastasis / Metastasis)
        # internally consistent. A single-field baseline can't contradict itself; the pipeline's
        # two fields can, when an earlier gate (e.g. G4-FAITH) trims one but not the other. These
        # rules are conservative & general — they never fabricate a brand-new met claim, only sync
        # or propagate so the two fields agree. [2026-06-05 floor-lock, A1]
        cancer_rc = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_rc, dict) and "Metastasis" in cancer_rc and "Distant Metastasis" in cancer_rc:
            dm_rc = str(cancer_rc.get("Distant Metastasis", "") or "").strip()
            m_rc = str(cancer_rc.get("Metastasis", "") or "").strip()
            def _met_status(v):
                v = v.lower()
                if not v: return "EMPTY"
                if any(k in v for k in ("not sure", "unsure", "suspect", "suspicious", "possible", "concern")): return "UNSURE"
                if "yes" in v: return "YES"
                if v in ("no", "no.", "none") or v.startswith("no ") or v.startswith("no,") or v.startswith("no."): return "NO"
                return "OTHER"
            dm_s, m_s = _met_status(dm_rc), _met_status(m_rc)
            DISTANT_RC = ["liver", "hepatic", "lung", "pulmonary", "bone", "osseous", "brain", "cerebral",
                          "peritone", "pleural", "adrenal", "distant", "contralateral", "spine", "spinal",
                          "mediastin", "retroperitone", "mesenteric", "omentum", "omental", "metastatic",
                          "cervical"]  # cervical (neck) nodes are distant (M1) for breast [bug5, breast15]
            REGIONAL_RC = ["axill", "sentinel", "supraclavicular", "infraclavicular", "internal mammary",
                           "chest wall", "ipsilateral", "regional", "lymph node", "node"]
            m_has_distant = any(s in m_rc.lower() for s in DISTANT_RC)
            m_has_regional = any(s in m_rc.lower() for s in REGIONAL_RC)
            changed_rc = None
            # R1: distant met implies general met — if DistMet=Yes but Metastasis is weaker, sync up
            if dm_s == "YES" and m_s in ("EMPTY", "NO"):
                cancer_rc["Metastasis"] = dm_rc
                changed_rc = f"R1 distant→general: Metastasis '{m_rc}' → '{dm_rc}'"
            # R2: Metastasis claims a distant site but DistMet was trimmed to EMPTY by an earlier
            #     gate (unconfirmed). Don't keep an asserted claim the gate already removed, and
            #     don't silently drop it either → mark both 'Not sure' (honest, consistent).
            elif m_s == "YES" and m_has_distant and dm_s == "EMPTY":
                # keep the named site(s) but mark suspected — the distant claim was never confirmed
                # (DistMet empty) so don't assert "Yes", but don't lose the site detail either.
                sites_rc = re.sub(r'(?i)^\s*(?:yes|suspected|not sure)?[\s,:.()-]*(?:to\s+)?', '', m_rc).strip().strip("()").strip()
                susp_rc = f"Suspected, to {sites_rc}" if sites_rc and sites_rc.lower() not in ("", "yes") else "Not sure"
                cancer_rc["Metastasis"] = susp_rc
                cancer_rc["Distant Metastasis"] = susp_rc
                changed_rc = f"R2 unconfirmed distant claim → '{susp_rc}' (was DistMet empty, Met '{m_rc}')"
            # R3: DistMet explicitly No but Metastasis claims ONLY distant organs (no nodal/regional)
            #     → the evidence-based No wins.
            elif dm_s == "NO" and m_s == "YES" and m_has_distant and not m_has_regional:
                cancer_rc["Metastasis"] = "No"
                changed_rc = f"R3 DistMet=No vs distant-only Metastasis '{m_rc}' → 'No'"
            # R4: mirror an UNSURE across to an EMPTY partner so they don't half-contradict
            elif dm_s == "UNSURE" and m_s == "EMPTY":
                cancer_rc["Metastasis"] = dm_rc
                changed_rc = f"R4 mirror unsure → Metastasis '{dm_rc}'"
            elif m_s == "UNSURE" and dm_s == "EMPTY":
                cancer_rc["Distant Metastasis"] = m_rc
                changed_rc = f"R4 mirror unsure → Distant Metastasis '{m_rc}'"
            if changed_rc:
                print(f"    [POST-MET-RECONCILE] {changed_rc}")

        # NOTE [bug10, pdac3]: a POST-DISTMET-SITES hook (auto-append documented met organs) was
        # prototyped and REMOVED — it fired on negated radiology lines ("No suspicious osseous
        # lesions" → added bone) and ambiguous non-met lesions, manufacturing hallucinated sites.
        # Principle #1 (精确忠实) outranks #2 (不遗漏): an incomplete-but-correct site list (e.g.
        # pdac3 "liver, peritoneum" missing spleen) is preferable to a fabricated one. Accepted as P2.

        # POST-DISTMET-BENIGN: a "Not sure" distant-met hedge driven by a lesion the note itself reads
        # as a BENIGN entity (meningioma, hemangioma, simple cyst, lipoma) — or where metastasis is
        # called "unlikely" — overstates uncertainty. If the current assessment raises NO metastatic
        # concern and the intent is curative, the benign read governs → "No". Conservative: requires an
        # explicit benign qualifier AND no pending/suspected distant-met language, so it never touches
        # real follow-up cases (e.g. liver/lung nodules "pending confirmation"). [2026-06-06, bug4, breast13]
        cancer_bn = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_bn, dict):
            dm_bn = str(cancer_bn.get("Distant Metastasis", "") or "").strip().lower()
            m_bn = str(cancer_bn.get("Metastasis", "") or "").strip().lower()
            stage_bn = str(cancer_bn.get("Stage_of_Cancer", "") or "").lower()
            g_bn = keypoints.get("Treatment_Goals", {})
            goals_bn = str(g_bn.get("goals_of_treatment", "") or "").lower() if isinstance(g_bn, dict) else ""
            dm_unsure = any(k in dm_bn for k in ("not sure", "unsure", "suspect", "possibl"))
            m_unsure = any(k in m_bn for k in ("not sure", "unsure", "suspect", "possibl"))
            # the model also sometimes lists the benign lesion itself as a (possible) met site
            names_benign_site = bool(re.search(r'falx|falcine|parafalcine|dural|meningioma', dm_bn + " " + m_bn))
            is_curative_bn = ("curative" in goals_bn or "adjuvant" in goals_bn or "risk reduction" in goals_bn)
            not_metastatic_bn = ("iv" not in stage_bn and "metastatic" not in stage_bn)
            if (dm_unsure or m_unsure or names_benign_site) and is_curative_bn and not_metastatic_bn:
                note_bn = (note_text or "").lower()
                ap_bn = (assessment_and_plan or "").lower()
                benign_read = re.search(
                    r'most likely (?:a |an )?(?:meningioma|hemangioma|benign|cyst|lipoma|adenoma)|'
                    r'(?:consistent with|favor|likely)\s+(?:a |an )?(?:meningioma|hemangioma|benign|cyst|lipoma)|'
                    r'metasta\w*\s+(?:is|are|remains?)\s+(?:an?\s+)?(?:unlikely|very unlikely)|'
                    r'unlikely\s+(?:to\s+(?:represent|be)\s+)?(?:a\s+)?metasta', note_bn)
                real_pending = re.search(
                    r'pending confirmation|follow[\s-]?up on (?:the )?(?:lung|liver|bone|lesion)|'
                    r'suspicious for (?:distant\s+)?metasta|concerning for (?:distant\s+)?metasta|'
                    r'biopsy[^.]{0,30}(?:lesion|nodule|met)|nodules? pending', ap_bn + " " + dm_bn + " " + m_bn)
                if benign_read and not real_pending:
                    for fld_bn in ("Distant Metastasis", "Metastasis"):
                        v_bn = str(cancer_bn.get(fld_bn, "") or "")
                        # strip the benign-read lesion site (e.g. "and possibly to falx cerebri") so it is
                        # not reported as a met; keep any genuine regional/nodal finding alongside it.
                        v2_bn = re.sub(
                            r'(?i)[,;]?\s*(?:and\s+)?(?:possibl[ye]\s+)?(?:to\s+)?(?:the\s+)?'
                            r'(?:parafalcine|falcine|falx(?:\s+cerebri)?|dural[\s-]?based?|meningioma)[^,.;]*',
                            '', v_bn).strip().strip(',;').strip()
                        # if nothing concrete remains (empty / bare "Yes," / pure hedge), it is "No"
                        if (not v2_bn) or v2_bn.lower() in ("yes", "yes,", "suspected", "suspected,") \
                                or any(k in v2_bn.lower() for k in ("not sure", "unsure", "suspect", "possibl")):
                            v2_bn = "No"
                        if v2_bn != v_bn:
                            cancer_bn[fld_bn] = v2_bn
                    print(f"    [POST-DISTMET-BENIGN] benign lesion read + curative + no met concern → cleaned met fields")

        # POST-STAGE-PARENS-CLEANUP: remove empty "()" and dangling connectives left behind
        # when a prior hook (e.g. POST-STAGE-VERIFY-NOTE) strips a fabricated "Stage X" out of
        # a phrase like "metastatic (Stage IV)" -> "metastatic ()". Deterministic, content-free
        # cosmetic cleanup; runs last among the stage hooks. [2026-06-05 floor-lock]
        cancer_pc = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_pc, dict):
            stage_pc = cancer_pc.get("Stage_of_Cancer", "")
            if isinstance(stage_pc, str) and stage_pc:
                cleaned_pc = re.sub(r'\(\s*\)', '', stage_pc)            # empty parens
                cleaned_pc = re.sub(r'^\s*\(([^()]+)\)', r'\1', cleaned_pc)  # unwrap leading parenthetical left after stripping "Originally Stage X (...)"
                cleaned_pc = re.sub(r'(?i)^\s*(now|currently)\s+', '', cleaned_pc)  # drop leading connective after "Originally ..." prefix was removed
                cleaned_pc = re.sub(r'\s*,\s*,', ', ', cleaned_pc)       # doubled commas
                cleaned_pc = re.sub(r'(?i)\b(now|originally|currently)\s*[,;:]?\s*$', '', cleaned_pc)  # dangling connective at end
                cleaned_pc = re.sub(r'\s{2,}', ' ', cleaned_pc)         # collapse spaces
                cleaned_pc = cleaned_pc.strip().strip(',').strip()
                if cleaned_pc != stage_pc:
                    cancer_pc["Stage_of_Cancer"] = cleaned_pc
                    print(f"    [POST-STAGE-PARENS-CLEANUP] '{stage_pc}' → '{cleaned_pc}'")

        # POST-RESPONSE: cross-reference response_assessment with findings [B53, B60]
        response = keypoints.get("Response_Assessment", {})
        resp_val = response.get("response_assessment", "") if isinstance(response, dict) else ""
        findings = keypoints.get("Clinical_Findings", {})
        find_val = findings.get("findings", "") if isinstance(findings, dict) else ""

        if resp_val and "not mentioned" in resp_val.lower() and find_val:
            RESPONSE_KEYWORDS = ["progression", "progressed", "stable disease", "partial response",
                                 "complete response", "no evidence of recurrence", "no evidence of disease",
                                 "increased", "decreased metabolic", "new metast"]
            for kw in RESPONSE_KEYWORDS:
                if kw in find_val.lower():
                    sentences = re.split(r'[.;]', find_val)
                    relevant = [s.strip() for s in sentences if kw in s.lower()]
                    if relevant:
                        response["response_assessment"] = ". ".join(relevant[:2]) + "."
                        print(f"    [POST-RESPONSE] patched from findings: '{relevant[0][:60]}...'")
                    break

        # POST-RESPONSE-TREATMENT: validate "Not yet on treatment" [v17]
        resp = keypoints.get("Response_Assessment", {})
        if isinstance(resp, dict):
            resp_val = resp.get("response_assessment", "") or ""
            if "not yet on treatment" in resp_val.lower() or "not on treatment" in resp_val.lower():
                ap_lower_rt = (assessment_and_plan or '').lower()  # v18: use local var, not row.get
                # Check if patient IS on treatment
                # [bug7] require evidence of an ACTIVE anticancer agent — bare "continue" matched
                # supportive meds ("continue creon") and mislabeled surveillance patients "On treatment".
                on_treatment = bool(re.search(
                    r'(?:currently on|on cycle|cycle\s*\d|c\d+\s*d\d+|'
                    r'(?:continue|continuing|on)\s+(?:\w+\s+){0,2}\w*'
                    r'(?:oxifen|zole|mab|lib|nib|platin|tabine|rubicin|taxel|fluorouracil|'
                    r'gemcitabine|capecitabine|folfirinox|folfox|folfiri|pembrolizumab|chemo))',
                    ap_lower_rt))
                has_meds = bool((keypoints.get("Current_Medications", {}).get("current_meds", "") or "").strip())
                if on_treatment or has_meds:
                    resp["response_assessment"] = "On treatment; response assessment not available from current visit."
                    print(f"    [POST-RESPONSE-TREATMENT] Corrected 'Not yet on treatment' → on treatment")

        # POST-RESPONSE-PRETREATMENT: Correct "On treatment" for pre-treatment consultations [v31]
        # If response_assessment says "On treatment" but current_meds is empty and it's a new patient consultation,
        # the patient hasn't started treatment yet
        resp = keypoints.get("Response_Assessment", {})
        if isinstance(resp, dict):
            resp_val = resp.get("response_assessment", "") or ""
            if "on treatment" in resp_val.lower() and "not" not in resp_val.lower()[:30]:
                cur_meds = (keypoints.get("Current_Medications", {}).get("current_meds", "") or "").strip()
                pt_type = (keypoints.get("Reason_for_Visit", {}).get("Patient type", "") or "").strip().lower()
                summary = (keypoints.get("Reason_for_Visit", {}).get("summary", "") or "").lower()
                recent = (keypoints.get("Treatment_Changes", {}).get("recent_changes", "") or "").lower()
                # Clues that treatment hasn't started
                is_consultation = ("consultation" in summary or "new patient" in pt_type or
                                   "initial" in summary or "establish care" in summary)
                just_prescribed = bool(re.search(
                    r'(?:rx given|prescription (?:sent|ordered)|will start|instructed to begin|ok to start)',
                    recent + " " + summary))
                # An ACTUALLY-started-treatment signal in the assessment overrides the consult heuristic.
                # (cur_meds alone is unreliable here: the model often lists PLANNED chemo as a "current"
                # med that a later drug-dictionary hook then strips, so by output time cur_meds is empty
                # but at this point it may be non-empty — don't gate solely on it. [bug7, breast10])
                started_signal = bool(re.search(
                    r'c\d+\s*d\d+|cycle\s*\d|s/p\s+\d+\s+(?:cycle|cycles)|received \d+\s+(?:cycle|cycles)|'
                    r'currently receiving|currently on (?:chemo|cycle|treatment)|on cycle',
                    (assessment_and_plan or "").lower() + " " + recent))
                if (is_consultation or just_prescribed) and not started_signal:
                    resp["response_assessment"] = "Not yet on treatment — no response to assess."
                    print(f"    [POST-RESPONSE-PRETREATMENT] Corrected 'On treatment' → Not yet on treatment (consultation/just-prescribed, no started-treatment signal)")

        # POST-RESPONSE-SURVEILLANCE: a patient s/p resection on post-surgical surveillance (no active
        # anticancer drug) is neither "on treatment" nor "not yet on treatment" — treatment is complete
        # and they are being monitored. State the surveillance, and surface a rising-marker / recurrence
        # concern when the assessment raises one. General oncology rule. [2026-06-06, bug7, pdac15]
        resp_sv = keypoints.get("Response_Assessment", {})
        if isinstance(resp_sv, dict):
            rv_sv = (resp_sv.get("response_assessment", "") or "").lower()
            cur_meds_sv = (keypoints.get("Current_Medications", {}).get("current_meds", "") or "").strip()
            ctx_sv = (assessment_and_plan or "").lower() + " " + (note_text or "").lower()
            resected_sv = re.search(r'\b(?:resected|s/p\s+(?:resection|whipple|pancrea\w*ectomy|mastectomy|lumpectomy|surgery)|'
                                    r'status post (?:resection|surgery)|post[\s-]?(?:operative|surgical resection))', ctx_sv)
            surveillance_sv = re.search(r'surveillance|rising (?:marker|ca\s*19|cea)|high risk for recurrence|'
                                        r'monitor(?:ing)? for recurrence|recheck (?:ca\s*19|cea|markers)|'
                                        r'concern\w* for recurrence', ctx_sv)
            misstated_sv = (("on treatment" in rv_sv and "not" not in rv_sv[:30]) or
                            "not yet on treatment" in rv_sv or "not on treatment" in rv_sv or not rv_sv.strip())
            # GUARD [b17 regression]: a post-surgical patient who is here to PLAN/START adjuvant therapy
            # is pre-treatment, NOT in surveillance. Exclude new-patient / initial consults and any
            # assessment that recommends starting (neo)adjuvant chemo — those belong to PRETREATMENT.
            pt_type_sv = (keypoints.get("Reason_for_Visit", {}).get("Patient type", "") or "").strip().lower()
            summary_sv = (keypoints.get("Reason_for_Visit", {}).get("summary", "") or "").lower()
            is_new_consult_sv = ("new patient" in pt_type_sv or "consultation" in summary_sv or
                                 "initial" in summary_sv or "establish care" in summary_sv)
            planning_adjuvant_sv = re.search(
                r'recommend\w*\s+(?:\w+\s+){0,2}(?:adjuvant|neoadjuvant|chemo)|'
                r'will\s+(?:start|begin|recommend|proceed)|proceed with (?:adjuvant|neoadjuvant|chemo)|'
                r'plan(?:ned|ning)?\s+(?:for\s+)?(?:adjuvant|neoadjuvant)',
                (assessment_and_plan or "").lower())
            if (resected_sv and surveillance_sv and not cur_meds_sv and misstated_sv
                    and not is_new_consult_sv and not planning_adjuvant_sv):
                rising_sv = re.search(r'rising (?:marker|ca\s*19|cea)|increas\w* (?:ca\s*19|cea|marker)|'
                                      r'high risk for recurrence|concern\w* for recurrence', ctx_sv)
                msg_sv = "On post-surgical surveillance; no active cancer treatment at this visit."
                if rising_sv:
                    msg_sv += " Rising tumor markers are concerning for possible recurrence."
                resp_sv["response_assessment"] = msg_sv
                print(f"    [POST-RESPONSE-SURVEILLANCE] resected + surveillance, no active tx → surveillance statement")

        # POST-RESPONSE-GENOMIC: Remove genomic test results from response_assessment [v29] [breast-only]
        # Oncotype/MammaPrint are prognostic tools, not treatment response assessments
        resp = keypoints.get("Response_Assessment", {})
        if cancer_type == "breast" and isinstance(resp, dict):
            resp_val = resp.get("response_assessment", "") or ""
            if resp_val and len(resp_val) < 50:
                # Short response that looks like a genomic test result
                is_genomic = bool(re.search(
                    r'(?i)(?:low|high|intermediate)\s+risk\s+\[?REDACTED\]?|'
                    r'(?:oncotype|mammaprint|genomic)\s+(?:score|result|test)',
                    resp_val))
                if is_genomic:
                    # Check if note has post-neoadjuvant surgical pathology
                    note_lower_rg = note_text.lower()
                    has_neoadj_path = bool(re.search(
                        r'(?:s/p|status post|post)\s*(?:neoadjuvant|NAC)|'
                        r'(?:residual\s+(?:tumor|disease|carcinoma))|'
                        r'(?:mastectomy|lumpectomy).*(?:neoadjuvant|preoperative)',
                        note_lower_rg))
                    if has_neoadj_path:
                        # Try to extract pathology response from findings
                        find_dict = keypoints.get("Clinical_Findings", {})
                        find_val_rg = find_dict.get("findings", "") if isinstance(find_dict, dict) else ""
                        if find_val_rg:
                            # Look for residual tumor / pathologic response info in findings
                            path_sentences = [s.strip() for s in re.split(r'[.;]', find_val_rg)
                                              if re.search(r'(?i)residual|patholog|mastectomy|lumpectomy|'
                                                           r'lymph node|pCR|tumor bed|treatment effect', s)]
                            if path_sentences:
                                resp["response_assessment"] = ". ".join(path_sentences[:3]) + "."
                                print(f"    [POST-RESPONSE-GENOMIC] Replaced genomic test with surgical pathology: '{resp_val}' → '{resp['response_assessment'][:80]}...'")
                            else:
                                resp["response_assessment"] = "Post-neoadjuvant pathologic assessment — see findings for surgical pathology details."
                                print(f"    [POST-RESPONSE-GENOMIC] Replaced genomic test with pathology reference: '{resp_val}'")

        # POST-RESPONSE-COMPRESS: response_assessment should be a concise CONCLUSION about how the
        # disease is responding — not a copy of the CT/findings report (E, verbose), and not
        # treatment-plan directives (G, "Continue X / Recommend Y"). Drop plan-directive sentences,
        # then compress to the conclusion (+ at most one evidence sentence). General rule: the field
        # answers "how is it responding", the plan fields answer "what's next". [2026-06-05, E/G]
        resp_cz = keypoints.get("Response_Assessment", {})
        if isinstance(resp_cz, dict):
            rv = (resp_cz.get("response_assessment", "") or "").strip()
            if rv and rv.lower().rstrip(".") not in ("not mentioned", "not available", "n/a", "not assessed at this visit"):
                sents_cz = [s.strip() for s in re.split(r'(?<=[.;])\s+', rv) if s.strip()]
                def _is_plan_sentence(s):
                    sl = s.lower().strip()
                    if re.match(r'(continue|start|begin|repeat|obtain|recommend|consider|refer|schedule|'
                                r'plan\b|check|follow[\s-]?up|f/u)\b', sl):
                        return True
                    if re.search(r'\brecommend|will\s+(start|begin|repeat|obtain|check|continue|consider|plan|refer|schedule|hold)|'
                                 r'plan(s|ning)?\s+to|\bf/?u\b|refer(ral)?\s+to|rx\s+given|prescription|'
                                 r'next\s+(appointment|visit|cycle)|should\s+(start|begin|consider)', sl):
                        return True
                    return False
                kept_cz = [s for s in sents_cz if not _is_plan_sentence(s)]
                changed_cz = len(kept_cz) < len(sents_cz)
                RESP_KW = re.compile(r'(?i)(stable|progress|partial response|complete response|respond|'
                                     r'no evidence of (disease|recurrence)|\bned\b|decreas|increas|improv|'
                                     r'new (lesion|metast)|resolution|regress|no response|mixed response|'
                                     r'remission|tolerat)')
                if len(kept_cz) > 2 or len(" ".join(kept_cz)) > 400:
                    concl_cz = [s for s in kept_cz if RESP_KW.search(s)]
                    if concl_cz:
                        kept_cz = concl_cz[:2]
                        changed_cz = True
                if changed_cz:
                    new_rv = " ".join(kept_cz).strip()
                    if not new_rv:
                        new_rv = "Not assessed at this visit."
                    elif not new_rv.endswith((".", ";")):
                        new_rv += "."
                    resp_cz["response_assessment"] = new_rv
                    print(f"    [POST-RESPONSE-COMPRESS] {len(rv)}→{len(new_rv)} chars (dropped plan/verbose)")

        # POST-DRUG-VERIFY: Remove hallucinated drugs not found in original note text
        for drug_field_key in ["Current_Medications", "Treatment_Changes"]:
            drug_dict = keypoints.get(drug_field_key, {})
            if not isinstance(drug_dict, dict):
                continue
            for field_name in ["current_meds", "recent_changes"]:
                val = drug_dict.get(field_name, "")
                if not val or not isinstance(val, str):
                    continue
                meds = [m.strip() for m in val.split(",")]
                verified = []
                for med in meds:
                    if not med:
                        continue
                    # Extract core drug name (first word or known pattern)
                    # e.g. "Doxorubicin 50mg" -> check "doxorubicin"
                    words = re.findall(r'[a-zA-Z]{4,}', med)
                    found = any(w.lower() in note_lower for w in words) if words else True
                    if found:
                        verified.append(med)
                    else:
                        print(f"    [POST-DRUG-VERIFY] REMOVED hallucinated drug: '{med}' (not found in note)")
                if len(verified) < len(meds):
                    drug_dict[field_name] = ", ".join(verified) if verified else ""

        # POST-MEDS-FILTER: Remove non-cancer medications from current_meds [v16]
        NON_CANCER_MEDS = [
            # Ophthalmology
            "latanoprost", "xalatan", "timolol", "brimonidine", "dorzolamide",
            "travoprost", "bimatoprost", "ophthalmic",
            # Blood pressure
            "lisinopril", "amlodipine", "losartan", "valsartan", "atenolol",
            "metoprolol", "hydrochlorothiazide", "hctz", "nifedipine",
            # Diabetes
            "metformin", "glipizide", "glyburide", "sitagliptin", "pioglitazone",
            "empagliflozin", "liraglutide", "semaglutide",
            # Vitamins / supplements (non-cancer)
            "fish oil", "omega-3", "multivitamin", "coq10",
            # Allergy
            "allegra", "zyrtec", "cetirizine", "fexofenadine", "loratadine",
            "claritin", "montelukast", "singulair",
            # Psychiatric (standalone, not cancer-related)
            "buspirone", "citalopram", "escitalopram", "sertraline",
            "fluoxetine", "paroxetine", "venlafaxine", "duloxetine",
            # Thyroid
            "levothyroxine", "synthroid",
            # Cholesterol
            "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin",
            # GI (non-supportive)
            "sucralfate",
            # Other
            "albuterol", "fluticasone", "montelukast",
        ]
        # Oncology whitelist — never remove these even if they appear in blacklist
        ONCO_WHITELIST = [
            # breast
            "tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant",
            "trastuzumab", "pertuzumab", "herceptin", "perjeta",
            "palbociclib", "ribociclib", "abemaciclib", "tucatinib", "sacituzumab",
            # pan-cancer chemo
            "doxorubicin", "cyclophosphamide", "paclitaxel", "docetaxel", "carboplatin",
            "capecitabine", "xeloda", "gemcitabine", "eribulin", "vinorelbine",
            "cisplatin", "oxaliplatin", "irinotecan", "fluorouracil", "leucovorin",
            "temozolomide", "streptozocin", "abraxane", "nab-paclitaxel",
            # targeted / PARP / IO
            "everolimus", "sunitinib", "erlotinib", "sorafenib", "regorafenib",
            "olaparib", "talazoparib", "rucaparib", "niraparib",
            "larotrectinib", "entrectinib", "bevacizumab", "ramucirumab",
            "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab",
            "dabrafenib", "trametinib",
            # NET / SSA
            "octreotide", "sandostatin", "lanreotide", "somatuline",
            # supportive
            "zoledronic", "zometa", "denosumab", "xgeva", "reclast",
            "ondansetron", "zofran", "granisetron", "prochlorperazine",
            "dexamethasone", "filgrastim", "pegfilgrastim", "neulasta",
            "epoetin", "darbepoetin", "creon", "pancrelipase",
        ]
        drug_dict_meds = keypoints.get("Current_Medications", {})
        if isinstance(drug_dict_meds, dict):
            meds_val = drug_dict_meds.get("current_meds", "")
            if meds_val and isinstance(meds_val, str):
                meds_list = [m.strip() for m in meds_val.split(",")]
                filtered = []
                for med in meds_list:
                    if not med:
                        continue
                    med_lower = med.lower()
                    # Clean dosing frequency prefixes (e.g. "Every 6 Hours; latanoprost")
                    med_cleaned = re.sub(r'^(?:every\s+\d+\s+hours?;?\s*|daily;?\s*|twice\s+daily;?\s*|bid;?\s*|tid;?\s*)', '', med_lower, flags=re.IGNORECASE).strip()
                    # Check whitelist first
                    is_onco = any(ow in med_cleaned for ow in ONCO_WHITELIST)
                    if is_onco:
                        filtered.append(med)
                        continue
                    # Check blacklist
                    is_noncancer = any(nc in med_cleaned for nc in NON_CANCER_MEDS)
                    if is_noncancer:
                        print(f"    [POST-MEDS-FILTER] Removed non-cancer med: '{med}'")
                        continue
                    filtered.append(med)
                if len(filtered) < len(meds_list):
                    drug_dict_meds["current_meds"] = ", ".join(filtered) if filtered else ""

        # POST-SELF-MANAGED: Clear current_meds if physician disapproves of patient's self-managed drugs [v21]
        self_managed_cleared = False
        drug_dict_sm = keypoints.get("Current_Medications", {})
        if isinstance(drug_dict_sm, dict):
            meds_val_sm = (drug_dict_sm.get("current_meds", "") or "").strip()
            if meds_val_sm:
                ap_text_sm = (assessment_and_plan or '').lower()
                # Skeptical language signals (physician disapproval)
                SKEPTICAL_SIGNALS = [
                    'apparently', 'so-called', 'so called', 'claims to be',
                    'for some reason', 'believes is different', 'believes is better',
                    'discontinue our medications', 'on her own at home',
                    'on his own at home', 'self-administered', 'self-administer',
                    'administer on her own', 'administer on his own',
                    'in mexico', 'another country', 'outside the',
                ]
                signal_count = sum(1 for sig in SKEPTICAL_SIGNALS if sig in ap_text_sm)
                if signal_count >= 2:
                    # Cross-verify: check if Current Outpatient Medications has cancer drugs
                    med_section_match = re.search(
                        r'(?i)(?:Current Outpatient Medications|MEDICATIONS).*?(?:No current|Allergi|ALLERGI|PAST|Family|Social|Review of|Physical|Objective)',
                        note_text, re.DOTALL)
                    has_outpatient_cancer_med = False
                    if med_section_match:
                        med_section = med_section_match.group(0).lower()
                        for onco_drug in ONCO_WHITELIST:
                            if onco_drug in med_section and 'not taking' not in med_section[max(0, med_section.index(onco_drug)-50):med_section.index(onco_drug)+50]:
                                has_outpatient_cancer_med = True
                                break
                    if not has_outpatient_cancer_med:
                        print(f"    [POST-SELF-MANAGED] Cleared current_meds (physician disapproval: {signal_count} signals, no outpatient cancer meds)")
                        print(f"      before: '{meds_val_sm}'")
                        drug_dict_sm["current_meds"] = ""
                        self_managed_cleared = True

        # POST-SELF-MANAGED-PLAN: Clean plan fields if self-managed drugs were cleared [v22]
        if self_managed_cleared:
            for plan_key in ["Medication_Plan", "Therapy_plan"]:
                plan_dict = keypoints.get(plan_key, {})
                if isinstance(plan_dict, dict):
                    for field_name in plan_dict:
                        plan_val = plan_dict[field_name]
                        if not plan_val or not isinstance(plan_val, str):
                            continue
                        # Split into sentences, remove ones containing self-managed drug names
                        sentences = re.split(r'(?<=[.!])\s+', plan_val)
                        cleaned = []
                        removed = []
                        for sent in sentences:
                            sent_lower = sent.lower()
                            has_sm_drug = any(d in sent_lower for d in ['gemcitabine', 'docetaxel', 'doxorubicin', 'pamidronate', 'metabolic therapy', 'immunological vaccine', 'insulin'])
                            has_continue = 'continue' in sent_lower
                            if has_sm_drug and has_continue:
                                removed.append(sent[:50])
                            else:
                                cleaned.append(sent)
                        if removed:
                            plan_dict[field_name] = " ".join(cleaned).strip() if cleaned else ""
                            print(f"    [POST-SELF-MANAGED-PLAN] Cleaned {field_name}.{plan_key}: removed {len(removed)} self-managed sentences")

        # POST-SELF-MANAGED-SUMMARY: Fix "currently on" stopped drugs in summary [v22]
        if self_managed_cleared:
            rfv = keypoints.get("Reason_for_Visit", {})
            if isinstance(rfv, dict):
                summary_val = rfv.get("summary", "")
                if summary_val and isinstance(summary_val, str) and "currently on" in summary_val.lower():
                    # Check if note says these drugs were stopped
                    stopped_drugs = re.findall(r'(?i)(?:stopped|discontinued|last dose of)\s+(?:taking\s+)?(\w+)', note_text)
                    stopped_lower = [d.lower() for d in stopped_drugs]
                    # Also check A/P for "stopped" + drug names
                    ap_stopped = re.findall(r'(?i)(?:stopped|recently stopped)\s+(\w+)', assessment_and_plan or '')
                    stopped_lower.extend([d.lower() for d in ap_stopped])
                    if stopped_lower:
                        old_summary = summary_val
                        summary_val = re.sub(r'(?i)currently on\b', 'previously on', summary_val)
                        if summary_val != old_summary:
                            rfv["summary"] = summary_val
                            print(f"    [POST-SELF-MANAGED-SUMMARY] Fixed 'currently on' → 'previously on' (drugs stopped)")

        # POST-SUPP-ALLERGY: Remove supportive_meds that are actually from the Allergies list [v20]
        tc_dict = keypoints.get("Treatment_Changes", {})
        if isinstance(tc_dict, dict):
            supp_raw = tc_dict.get("supportive_meds", "") or ""
            supp_val = (", ".join(supp_raw) if isinstance(supp_raw, list) else str(supp_raw)).strip()
            if supp_val:
                # Extract allergy drug names from note text
                # Patterns: "ALL: drug1, drug2, ..." or "Allergies: ..." or "Allergies/Contraindications ..."
                allergy_drugs = set()
                # Pattern 1: "ALL:" abbreviation (common in clinical notes)
                all_match = re.search(r'\bALL:\s*(.+?)(?:\.\s|\n|$)', note_text)
                if all_match:
                    for drug in re.split(r'[,;]', all_match.group(1)):
                        d = drug.strip().lower()
                        if d and d not in ('nkda', 'none', 'no known'):
                            allergy_drugs.add(d)
                # Pattern 2: "Allergies:" or "Allergy:" section
                alg_match = re.search(r'(?i)\bAllerg(?:ies|y)(?:/Contraindications)?[\s:]+(.+?)(?:\n\n|\n\s*\n|Review of)', note_text)
                if alg_match:
                    for drug in re.split(r'[,;]', alg_match.group(1)):
                        d = drug.strip().lower()
                        # Skip common non-drug entries
                        if d and len(d) > 1 and d not in ('nkda', 'none', 'no known', 'no known drug allergies'):
                            allergy_drugs.add(d)
                if allergy_drugs:
                    supp_list = [s.strip() for s in supp_val.split(",")]
                    filtered_supp = []
                    for med in supp_list:
                        if not med:
                            continue
                        med_lower = med.lower().strip()
                        # Check if any allergy name appears in the medication name
                        is_allergy = any(alg in med_lower for alg in allergy_drugs)
                        if is_allergy:
                            print(f"    [POST-SUPP-ALLERGY] Removed allergy item from supportive_meds: '{med}'")
                            continue
                        filtered_supp.append(med)
                    if len(filtered_supp) < len(supp_list):
                        tc_dict["supportive_meds"] = ", ".join(filtered_supp) if filtered_supp else ""

        # POST-SUPP-WHITELIST: keep only cancer-supportive-care drugs in supportive_meds [extraction-audit fix]
        # Belt-and-suspenders: ensures home/non-cancer meds (eye drops, multivitamin, nasal spray,
        # melatonin, topical creams) do not leak into supportive_meds even if the gate-level
        # POST-SUPP filter did not run for this row.
        tc_dict = keypoints.get("Treatment_Changes", {})
        if isinstance(tc_dict, dict):
            supp_raw2 = tc_dict.get("supportive_meds", "") or ""
            supp_val2 = (", ".join(supp_raw2) if isinstance(supp_raw2, list) else str(supp_raw2)).strip()
            supp_low2 = supp_val2.lower()
            if supp_val2 and not supp_low2.startswith(("none", "not ", "no ")) and "not taking" not in supp_low2:
                # Blacklist of clearly NON-cancer-supportive home meds. Blacklist (not whitelist)
                # so legitimate chemo-supportive meds (antiemetics, PPIs, mag oxide, muscle
                # relaxants, pain/neuropathy meds) are PROTECTED from over-filtering.
                NON_SUPP = [
                    "ophthalmic", "eye drop", "into both eyes", "brimonidine", "latanoprost", "alphagan",
                    "xalatan", "patanol", "olopatadine", "prolensa", "vigamox", "timolol", "dorzolamide",
                    "nasal", "nasonex", "flonase", "fluticasone",
                    "cream", "ointment", "topical", "clotrimazole", "lotrimin", "lotrisone", "bengay", "menthol",
                    "multivitamin", "fish oil", "omega-3", "melatonin", "ascorbic", "vitamin c", "vitamin b",
                    "b-12", "cyanocobalamin", "ergocalciferol",
                    "loratadine", "claritin", "cetirizine", "zyrtec", "fexofenadine", "allegra",
                    "blood glucose", "test strip", "lancet", "glucose monitor", "monitor kit", "syringe-needle",
                ]
                items2 = [m.strip() for m in re.split(r',(?![^(]*\))|[;\n]', supp_val2) if m.strip()]
                kept_supp2 = [m for m in items2 if not any(b in m.lower() for b in NON_SUPP)]
                if len(kept_supp2) < len(items2):
                    removed2 = [m for m in items2 if m not in kept_supp2]
                    tc_dict["supportive_meds"] = ", ".join(kept_supp2) if kept_supp2 else ""
                    print(f"    [POST-SUPP-BLACKLIST] Removed non-supportive home meds: {removed2}")

        # POST-SUPP-SUPPLEMENT: recover high-value cancer-supportive meds the physician is clearly
        # continuing but the model dropped from supportive_meds. Two classes are core oncology
        # supportive care and were systematically missed (9 pdac samples in the full-field audit):
        # pancreatic enzyme replacement (Creon) — standard of care in pancreatic cancer — and
        # anticoagulation for cancer-associated thrombosis (Xarelto/Lovenox). Add when the drug is
        # (a) "continue/continues/resume/start X" anywhere in the note or A/P, or (b) a pancreatic
        # enzyme present in the outpatient medication list and not flagged "not taking". Both satisfy
        # the field definition ("supportive meds the patient is CURRENTLY TAKING related to cancer
        # treatment"). General oncology rule, not test-set-specific. [2026-06-06, fix#5, pdac2/5/8/9/13/15/18/19]
        ENZYME_TOKENS = ['creon', 'pancrelipase', 'zenpep', 'pertzye', 'viokace',
                         'lipase-protease-amylase', 'amylase-lipase-protease', 'lipase-amylase-protease']
        ANTICOAG_TOKENS = ['xarelto', 'rivaroxaban', 'eliquis', 'apixaban', 'lovenox', 'enoxaparin',
                           'warfarin', 'coumadin', 'dalteparin', 'fragmin', 'fondaparinux', 'edoxaba']
        tc_dict_su = keypoints.get("Treatment_Changes", {})
        if isinstance(tc_dict_su, dict):
            supp_raw_su = tc_dict_su.get("supportive_meds", "") or ""
            supp_val_su = (", ".join(supp_raw_su) if isinstance(supp_raw_su, list) else str(supp_raw_su)).strip()
            supp_low_su = supp_val_su.lower()
            note_low_su = (note_text or "").lower()
            ap_low_su = (assessment_and_plan or "").lower()
            full_low_su = note_low_su + " \n " + ap_low_su
            added_labels_su = []
            enzyme_already_su = ('creon' in supp_low_su or 'pancreli' in supp_low_su
                                 or 'enzyme' in supp_low_su or 'lipase' in supp_low_su)

            def _already_su(tok):
                # consider present if the brand/generic OR a pancreatic-enzyme equivalent already listed
                if tok in supp_low_su:
                    return True
                if tok in ENZYME_TOKENS and enzyme_already_su:
                    return True
                return False

            for tok in ENZYME_TOKENS + ANTICOAG_TOKENS:
                if _already_su(tok):
                    continue
                # only one pancreatic-enzyme entry total; label faithfully to what the note says
                if tok in ENZYME_TOKENS:
                    if any('enzyme' in l.lower() for l in added_labels_su):
                        continue  # already added one enzyme this pass
                    if tok == 'creon':
                        label = "Creon (pancreatic enzyme)"
                    elif tok in ('pancrelipase', 'zenpep', 'pertzye', 'viokace'):
                        label = f"{tok.capitalize()} (pancreatic enzyme)"
                    else:
                        label = "Pancreatic enzyme replacement"
                else:
                    label = tok.capitalize()
                if label in added_labels_su:
                    continue
                # (a) explicit continue/start phrasing anywhere in note or A/P
                cont_re = (r'(?:continue[sd]?|continuing|resume[sd]?|start(?:ed|ing)?|increase[sd]?)'
                           r'\s+(?:to\s+take\s+|with\s+)?(?:\w+[\s,]+){0,3}?' + re.escape(tok))
                hit = bool(re.search(cont_re, full_low_su))
                # (b) pancreatic enzyme present in the outpatient med list (currently taking)
                if not hit and tok in ENZYME_TOKENS and cancer_type != "breast":
                    for m in re.finditer(re.escape(tok), note_low_su):
                        ctx = note_low_su[max(0, m.start() - 12):m.end() + 60]
                        if any(neg in ctx for neg in ('not taking', 'discontinued', 'stopped', 'd/c')):
                            continue
                        hit = True
                        break
                if hit:
                    added_labels_su.append(label)
            if added_labels_su:
                if supp_val_su and not supp_low_su.startswith(("none", "not ", "no ")) and "not taking" not in supp_low_su:
                    tc_dict_su["supportive_meds"] = supp_val_su + ", " + ", ".join(added_labels_su)
                else:
                    tc_dict_su["supportive_meds"] = ", ".join(added_labels_su)
                print(f"    [POST-SUPP-SUPPLEMENT] recovered supportive meds: {added_labels_su}")

        # POST-MEDS-ENZYME-STRIP: pancreatic enzymes are SUPPORTIVE care, not anticancer therapy —
        # they belong in supportive_meds (recovered there by POST-SUPP-SUPPLEMENT), not current_meds.
        # Stripping them keeps current_meds = active anticancer therapy AND lets POST-MEDS-IV-CHECK
        # below run (it only fires when current_meds is empty). [2026-06-06, fix#6, pdac9]
        drug_dict_es = keypoints.get("Current_Medications", {})
        if isinstance(drug_dict_es, dict):
            cm_es = (drug_dict_es.get("current_meds", "") or "").strip()
            if cm_es:
                toks_es = [t.strip() for t in cm_es.split(",") if t.strip()]
                kept_es = [t for t in toks_es if not any(e in t.lower() for e in
                           ['creon', 'pancreli', 'lipase', 'amylase', 'protease', 'zenpep', 'pertzye', 'viokace'])]
                if len(kept_es) < len(toks_es):
                    drug_dict_es["current_meds"] = ", ".join(kept_es)
                    print(f"    [POST-MEDS-ENZYME-STRIP] removed pancreatic enzyme from current_meds (→ supportive)")

        # POST-MEDS-IV-CHECK: detect active IV chemo from A/P if current_meds is empty [v19]
        # v19: positive-match only (no fallback drug name scan — too many false positives in v18)
        # Skip if POST-SELF-MANAGED already cleared (physician disapproves — don't re-inject)
        drug_dict_meds = keypoints.get("Current_Medications", {})
        if isinstance(drug_dict_meds, dict):
            meds_val = (drug_dict_meds.get("current_meds", "") or "").strip()
            if not meds_val and not self_managed_cleared:
                ap_lower_iv = (assessment_and_plan or '').lower()
                if ap_lower_iv:
                    # v19: patterns that indicate ACTIVE/CURRENT chemo only
                    IV_CHEMO_PATTERNS = [
                        # "continue/continuing [with] [cycle N [of]] DRUG"
                        r'(?:continue|continuing)\s+(?:with\s+)?(?:cycle\s+\d+\s+(?:of\s+)?)?(\w+(?:\s*/\s*\w+)?)',
                        # "currently on DRUG" / "still on DRUG"
                        r'(?:currently\s+on|still\s+on)\s+(\w+(?:\s*/\s*\w+)?)',
                        # "cycle N [day N] of DRUG"
                        r'cycle\s+\d+\s+(?:day\s+\d+\s+)?(?:of\s+)?(\w+)',
                        # "started DRUG on DATE" (recent start)
                        r'started\s+(\w+)\s+on\s+\d',
                        # "on DRUG cycle" / "on DRUG day" (e.g. "on Gemzar Cycle #2")
                        r'\bon\s+(\w+)\s+(?:cycle|day)',
                        # "receiving DRUG" / "was/been/being given DRUG" (exclude prepositional "given" and future "be given")
                        r'(?:receiving|(?:was|been|being)\s+given)\s+(\w+)',
                        # "DRUG day N" (e.g. "AC day 1")
                        r'(\w+)\s+(?:day|d)\s*\d+',
                        # "DRUG Cycle N" / "DRUG cycle #N" (e.g. "Gemzar Cycle #2") [v23]
                        r'(\w+)\s+cycle\s*#?\d+',
                        # "switch/switched/changed to DRUG" (e.g. "switch to Gemzar") [v23]
                        r'(?:switch(?:ed)?|changed?)\s+to\s+(\w+)',
                        # "if DRUG is working" (implies currently on it) [v23]
                        r'if\s+(\w+)\s+is\s+working',
                    ]
                    KNOWN_CHEMO_IV = [
                        # regimens
                        "ac", "tc", "fec", "caf", "tac", "tchp", "thp",
                        "folfox", "folfiri", "folfoxiri", "folfirinox",
                        # chemo
                        "doxorubicin", "cyclophosphamide", "paclitaxel", "docetaxel", "taxol",
                        "taxotere", "carboplatin", "cisplatin", "gemcitabine", "gemzar",
                        "capecitabine", "xeloda", "irinotecan", "oxaliplatin",
                        "fluorouracil", "5-fu", "leucovorin",
                        "eribulin", "vinorelbine", "temozolomide", "streptozocin",
                        "abraxane", "nab-paclitaxel",
                        # breast targeted
                        "pertuzumab", "perjeta", "trastuzumab", "herceptin",
                        "olaparib", "lynparza", "palbociclib", "ibrance",
                        "ribociclib", "kisqali", "abemaciclib", "verzenio",
                        "fulvestrant", "faslodex", "lupron", "leuprolide", "goserelin", "zoladex",
                        # PDAC / GI / NET targeted
                        "everolimus", "afinitor", "sunitinib", "sutent",
                        "erlotinib", "tarceva", "sorafenib", "regorafenib",
                        "octreotide", "sandostatin", "lanreotide", "somatuline",
                        "bevacizumab", "avastin", "ramucirumab",
                        "rucaparib", "niraparib", "larotrectinib", "entrectinib",
                        # immunotherapy
                        "pembrolizumab", "keytruda", "atezolizumab", "tecentriq",
                        "nivolumab", "opdivo", "durvalumab", "imfinzi",
                        "ipilimumab", "yervoy",
                        # BRAF/MEK
                        "dabrafenib", "trametinib",
                    ]
                    PAST_CHEMO = ["previously on", "prior", "completed", "finished", "was on",
                                  "had received", "history of", "s/p"]
                    # alias map: abbreviations / regimen-prefix forms → canonical drug name [fix#6]
                    CHEMO_ALIAS = {"gem": "gemcitabine", "mfolfirinox": "folfirinox",
                                   "mfolfox": "folfox", "nab-paclitaxel": "abraxane",
                                   "5fu": "5-fu", "nal-iri": "irinotecan"}
                    found_chemo = []
                    for pattern in IV_CHEMO_PATTERNS:
                        for m in re.finditer(pattern, ap_lower_iv):
                            drug = m.group(1).strip().lower()
                            drug = CHEMO_ALIAS.get(drug, drug)
                            if drug in KNOWN_CHEMO_IV:
                                # Exclude past-tense mentions
                                start = max(0, m.start() - 30)
                                before = ap_lower_iv[start:m.start()]
                                if any(pc in before for pc in PAST_CHEMO):
                                    continue
                                found_chemo.append(drug)

                    # fix#6 — regimen-aware patterns the (\w+) templates above miss in PDAC notes.
                    # These notations indicate the patient IS on the regimen; "held"/"postponed"/
                    # "s/p N cycles" are ONGOING treatment, not discontinuation (held ≠ stopped).
                    DISCONT_IV = ('discontinued', 'd/c', 'stopped', 'switched off', 'no longer',
                                  'progressed on', 'progression on', 'completed all', 'off chemo permanently')
                    REGIMEN_RE = r'm?(folfirinox|folfoxiri|folfox|folfiri|gemcitabine|gemzar|abraxane|nab-paclitaxel|capecitabine|xeloda)'
                    regimen_patterns = [
                        # "on mFOLFIRINOX", "PDAC on FOLFOX", "started on gemcitabine"
                        r'\b(?:on|started on|receiving|treated with)\s+' + REGIMEN_RE,
                        # "s/p N cycles of FOLFIRINOX" — mid-regimen, ongoing
                        r'(?:s/p|status\s+post)\s+\d+\s+cycles?\s+of\s+' + REGIMEN_RE,
                        # "tolerated [subsequent] cycles of FOLFOX"
                        r'tolerat\w+\s+(?:subsequent\s+)?cycles?\s+of\s+' + REGIMEN_RE,
                        # "responding to gem abraxane" / "responding to FOLFIRINOX"
                        r'respond\w*\s+to\s+(?:gem\s+)?' + REGIMEN_RE,
                    ]
                    for pattern in regimen_patterns:
                        for m in re.finditer(pattern, ap_lower_iv):
                            drug = m.group(1).strip().lower()
                            drug = CHEMO_ALIAS.get(drug, drug)
                            ctx = ap_lower_iv[max(0, m.start() - 40):m.end() + 40]
                            if any(dc in ctx for dc in DISCONT_IV):
                                continue
                            found_chemo.append(drug)

                    # "gem abraxane" / "gem/abraxane" doublet → both components [fix#6, pdac9]
                    if re.search(r'gem\s*[-/ ]\s*abraxane|gemcitabine\s*[-/ ]\s*abraxane|abraxane\s*[-/ ]\s*gem', ap_lower_iv):
                        found_chemo.extend(['gemcitabine', 'abraxane'])

                    # "we will [then] resume treatment/chemo" + a named chemo drug → held, still current
                    if re.search(r'resume\s+(?:treatment|therapy|chemo)', ap_lower_iv):
                        for mr in re.finditer(REGIMEN_RE, ap_lower_iv):
                            d = CHEMO_ALIAS.get(mr.group(1).lower(), mr.group(1).lower())
                            ctx = ap_lower_iv[max(0, mr.start() - 40):mr.end() + 40]
                            if any(dc in ctx for dc in DISCONT_IV):
                                continue
                            found_chemo.append(d)

                    if found_chemo:
                        found_chemo = list(dict.fromkeys(found_chemo))  # dedup preserving order
                        drug_dict_meds["current_meds"] = ", ".join(found_chemo)
                        print(f"    [POST-MEDS-IV-CHECK] Added from A/P: {', '.join(found_chemo)}")

        # POST-MEDS-STOPPED: Remove stopped/discontinued drugs from current_meds [v23]
        # If recent_changes mentions "stopped/discontinued X", remove X from current_meds
        drug_dict_meds = keypoints.get("Current_Medications", {})
        changes_dict = keypoints.get("Treatment_Changes", {})
        if isinstance(drug_dict_meds, dict) and isinstance(changes_dict, dict):
            meds_val_ms = (drug_dict_meds.get("current_meds", "") or "").strip()
            changes_val = (changes_dict.get("recent_changes", "") or "").lower()
            if meds_val_ms and changes_val:
                # Find drugs mentioned as stopped/discontinued in recent_changes
                STOP_PATTERNS = [
                    r'(?:stopped|discontinued|d/c|d/ced|held|off)\s+(\w+)',
                    r'(\w+)\s+(?:stopped|discontinued|was\s+stopped|was\s+discontinued|held)',
                ]
                stopped_drugs = set()
                for pattern in STOP_PATTERNS:
                    for m in re.finditer(pattern, changes_val):
                        stopped_drugs.add(m.group(1).strip().lower())
                if stopped_drugs:
                    # Remove stopped drugs from current_meds
                    meds_list = [m.strip() for m in meds_val_ms.split(',')]
                    original_count = len(meds_list)
                    meds_list = [m for m in meds_list if m.lower().split('(')[0].split()[0].strip() not in stopped_drugs]
                    if len(meds_list) < original_count:
                        removed = original_count - len(meds_list)
                        drug_dict_meds["current_meds"] = ", ".join(meds_list) if meds_list else ""
                        print(f"    [POST-MEDS-STOPPED] Removed {removed} stopped drug(s) from current_meds: {stopped_drugs}")
                        print(f"      current_meds now: '{drug_dict_meds['current_meds']}'")

        # POST-MEDS-REGIMEN-FAB: a chemo REGIMEN acronym (AC/TC/AC-T/FOLFIRINOX/...) in
        # current_meds is "current" only if the note shows it is actually being administered
        # (cycle N, C#D#, s/p N cycles, currently on/receiving, continue). When the A/P frames
        # it as a not-yet-started option/discussion/recommendation/refusal — "options include
        # AC/weekly T or TC" (b17), "we discussed the regimen ... (TC)" (b12), "not receiving AC
        # as planned" (b17) — it is a PLAN, not a current med, and belongs nowhere in
        # current_meds. CROSSCHECK's coarse keyword scan is fooled here (its "start" matched a
        # neighbouring "start dignicap"), so handle regimen acronyms explicitly. General
        # oncology rule: you cannot be "currently on" a regimen with no administration evidence.
        # [2026-06-06, fix#2, b12/b17]
        REGIMEN_ACRONYMS = {"ac", "tc", "act", "ac-t", "ac/t", "t/ac", "tac", "tch", "tchp",
                            "thp", "ddac", "ec", "fec", "caf", "cmf", "cef",
                            "folfirinox", "folfox", "folfiri", "folfoxiri", "capox", "gemox",
                            "nal-iri", "ac/weekly t"}
        drug_dict_rf = keypoints.get("Current_Medications", {})
        if isinstance(drug_dict_rf, dict):
            meds_val_rf = (drug_dict_rf.get("current_meds", "") or "").strip()
            if meds_val_rf:
                hay_rf = ((assessment_and_plan or "") + " " + (note_text or "")).lower()
                active_sig_rf = re.compile(
                    r'c\d+\s*d\d+'
                    r'|cycle\s*#?\s*\d'
                    r'|on\s+cycle'
                    r'|s/?p\s+\d+\s+cycles?'
                    r'|(?:received|completed|currently\s+on|started)\s+(?:\w+\s+){0,3}?\d+\s+cycles?'
                    r'|currently\s+(?:on|receiving|being\s+treated)')
                noncurrent_frame_rf = ('option', 'discuss', 'recommend', 'we will need',
                                       'plan to', 'consider', 'candidate for', 'prefer',
                                       'proceed with', 'not receiving', 'not yet', 'would',
                                       'versus', ' vs ', 'either', 'or ')
                current_verb_rf = ('continue', 'continuing', 'currently', 'tolerating',
                                   'receiving', 'on treatment with', 'is on ', 'remains on')
                kept_rf = []
                for tok in [m.strip() for m in meds_val_rf.split(",") if m.strip()]:
                    base = tok.lower().split('(')[0].strip()
                    if base not in REGIMEN_ACRONYMS:
                        kept_rf.append(tok)
                        continue
                    # locate occurrences of the acronym as a whole token in the text
                    tok_re = r'\b' + re.escape(base).replace(r'\-', r'[-/ ]?').replace(r'\/', r'[-/ ]?') + r'\b'
                    is_current = False
                    for occ in re.finditer(tok_re, hay_rf):
                        win = hay_rf[max(0, occ.start() - 90):occ.end() + 90]
                        if any(nf in win for nf in noncurrent_frame_rf):
                            continue  # framed as plan/option → this occurrence is not current
                        if active_sig_rf.search(win) or any(cv in win for cv in current_verb_rf):
                            is_current = True
                            break
                    if is_current:
                        kept_rf.append(tok)
                    else:
                        print(f"    [POST-MEDS-REGIMEN-FAB] removed planned/discussed regimen '{tok}' (no administration evidence)")
                if len(kept_rf) < len([m for m in meds_val_rf.split(",") if m.strip()]):
                    drug_dict_rf["current_meds"] = ", ".join(kept_rf)

        # POST-MEDS-CROSSCHECK: Verify current_meds drugs appear in note's medication list or A/P as current [v32]
        # Runs AFTER IV-CHECK and STOPPED to catch drugs added by other hooks
        drug_dict_cc = keypoints.get("Current_Medications", {})
        if isinstance(drug_dict_cc, dict):
            meds_val_cc = (drug_dict_cc.get("current_meds", "") or "").strip()
            if meds_val_cc:
                # Find medication list section in note
                med_section = ""
                med_match = re.search(
                    r'(?i)(?:Current Outpatient Medications|Current.*?Medications|MEDICATIONS).*?(?:Allergi|PAST|Family|Social|Review of|Physical|Objective|ROS)',
                    note_text or "", re.DOTALL)
                if med_match:
                    med_section = med_match.group(0).lower()
                ap_lower_cc = (assessment_and_plan or "").lower()
                meds_list_cc = [m.strip() for m in meds_val_cc.split(",") if m.strip()]
                verified = []
                for med in meds_list_cc:
                    med_clean = med.strip().lower()
                    if not med_clean or med_clean == "[redacted]":
                        verified.append(med)
                        continue
                    in_med_list = med_clean in med_section
                    # If drug is in the note's medication list, ALWAYS keep it — it's a verified current med
                    if in_med_list:
                        verified.append(med)
                        continue
                    # Otherwise check A/P for current-use context
                    in_ap_current = False
                    if med_clean in ap_lower_cc:
                        for m in re.finditer(re.escape(med_clean), ap_lower_cc):
                            ctx = ap_lower_cc[max(0, m.start()-80):m.end()+80]
                            # FIRST check negative: literature citation — skip this match entirely
                            if any(w in ctx for w in ['reported', 'trial', 'study', 'response rate', 'et al',
                                                       'published', 'data confirm', 'phase iii', 'phase ii',
                                                       'patients receiving', 'patients treated']):
                                continue
                            # THEN check positive: drug is current/planned
                            if any(w in ctx for w in ['continue', 'currently', 'on ', 'taking', 'resume', 'start',
                                                       'on treatment', 'treatment with', 'therapy with',
                                                       'cycle', 'cycles', 'receiving', 'given',
                                                       'stable on', 'tolerating', 'respond']):
                                in_ap_current = True
                                break
                    # Also check full note (not just A/P) for active treatment context
                    if not in_ap_current:
                        note_lower_cc = (note_text or "").lower()
                        for m in re.finditer(re.escape(med_clean), note_lower_cc):
                            ctx = note_lower_cc[max(0, m.start()-60):m.end()+60]
                            # Skip if literature/historical context
                            if any(w in ctx for w in ['et al', 'reported', 'study', 'trial', 'published']):
                                continue
                            if any(w in ctx for w in ['currently on', 'now on', 'is on ', 'started on',
                                                       'patient on', 'continue ', 'present:', 'present,',
                                                       'cycles of', 'cycle of', 'on treatment with',
                                                       'been through', 'now s/p', 'receiving']):
                                in_ap_current = True
                                break
                    if in_ap_current:
                        verified.append(med)
                    else:
                        print(f"    [POST-MEDS-CROSSCHECK] Removed '{med}' — not found in medication list or A/P as current")
                if len(verified) < len(meds_list_cc):
                    drug_dict_cc["current_meds"] = ", ".join(verified) if verified else ""

        # POST-ER-CHECK: Infer ER status from medications when Type_of_Cancer lacks it [v16] [breast-only]
        ER_POS_DRUGS = ["tamoxifen", "letrozole", "anastrozole", "exemestane", "arimidex",
                        "femara", "aromasin", "fulvestrant", "faslodex",
                        "goserelin", "zoladex", "leuprolide", "lupron"]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val = cancer.get("Type_of_Cancer", "")
            # v17: expanded ER/PR detection — also match numeric patterns like "ER 95", "ER >90%"
            if isinstance(type_val, str) and not re.search(r'(?i)\b(?:ER|HR|PR)\s*[+-]|\b(?:ER|HR|PR)\s+(?:positive|negative)|\b(?:ER|HR|estrogen|progesterone)\s*\d|estrogen|progesterone|hormone\s+receptor|triple', type_val):
                # Type_of_Cancer has no receptor status at all — try to infer from meds
                meds_val = ""
                med_dict = keypoints.get("Current_Medications", {})
                if isinstance(med_dict, dict):
                    meds_val = (med_dict.get("current_meds", "") or "").lower()
                # Also check note text
                note_lower_er = note_text.lower()
                er_inferred = False
                # Check meds
                for drug in ER_POS_DRUGS:
                    if drug in meds_val or drug in note_lower_er:
                        # v29: Skip goserelin/zoladex/leuprolide/lupron if used for fertility preservation
                        if drug in ("goserelin", "zoladex", "leuprolide", "lupron"):
                            fertility_ctx = bool(re.search(
                                r'fertility\s+preserv|egg\s+harvest|cryopreserv|'
                                r'goserelin.*fertility|fertility.*goserelin|'
                                r'during\s+chemotherapy.*fertility|fertility.*during\s+chemo',
                                note_lower_er))
                            tnbc_ctx = bool(re.search(r'\btnbc\b|triple.negative', note_lower_er))
                            if fertility_ctx or tnbc_ctx:
                                print(f"    [POST-ER-CHECK] Skipped {drug} — fertility preservation / TNBC context")
                                continue
                        er_info = "ER+ (inferred from " + drug + ")"
                        # v17: avoid leading comma when type_val is empty
                        if type_val.strip():
                            type_val = type_val.rstrip() + ", " + er_info
                        else:
                            type_val = er_info
                        cancer["Type_of_Cancer"] = type_val
                        er_inferred = True
                        print(f"    [POST-ER-CHECK] Inferred ER+ from drug: {drug}")
                        break
                # Check note for explicit ER mention if not found from drugs
                if not er_inferred:
                    er_match = re.search(r'(?i)\b(ER|estrogen\s+receptor)\s*[\s:]*\s*(positive|\+|negative|-)', note_lower_er)
                    if er_match:
                        status = "+" if er_match.group(2) in ("positive", "+") else "-"
                        er_info = f"ER{status}"
                        if type_val.strip():
                            type_val = type_val.rstrip() + ", " + er_info
                        else:
                            type_val = er_info
                        cancer["Type_of_Cancer"] = type_val
                        er_inferred = True
                        print(f"    [POST-ER-CHECK] Found ER status in note: ER{status}")

        # POST-HER2-CHECK: If Type_of_Cancer has ER/PR but no HER2, search note and append [v15] [breast-only]
        HER2_POS_DRUGS = ["trastuzumab", "pertuzumab", "herceptin", "t-dm1", "t-dxd",
                          "ado-trastuzumab", "lapatinib", "tykerb", "tucatinib"]
        HER2_POS_REGIMENS = ["tchp", "thp", "ac-thp", "acthp"]
        HER2_SEARCH_KEYWORDS = [
            "her2", "her-2", "her2neu", "her 2", "her 2 neu",
            "ihc", "fish ratio",
        ] + HER2_POS_DRUGS + HER2_POS_REGIMENS
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val = cancer.get("Type_of_Cancer", "")
            if isinstance(type_val, str) and re.search(r'(?i)\b(?:ER|HR|PR)[+-]|\b(?:ER|HR|PR)\s+(?:positive|negative)', type_val) and not re.search(r'(?i)HER2|HER-2|HER 2|triple.neg|TNBC', type_val):
                # Has ER status but missing HER2 — search the note
                her2_found = None
                for kw in HER2_SEARCH_KEYWORDS:
                    pattern = r'\b' + re.escape(kw) + r'\b' if len(kw) <= 4 else re.escape(kw)
                    m = re.search(pattern, note_lower)
                    if m:
                        # Try to determine HER2 status from context
                        ctx_start = max(0, m.start() - 80)
                        ctx_end = min(len(note_lower), m.end() + 80)
                        ctx = note_lower[ctx_start:ctx_end]
                        if kw in [d for d in HER2_POS_DRUGS] or kw in HER2_POS_REGIMENS:
                            her2_found = "HER2+"
                            break
                        # Check negative BEFORE positive (avoid "non-amplified" matching "amplified")
                        elif "triple negative" in ctx or "tnbc" in ctx:
                            her2_found = "HER2-"
                            break
                        elif "non-amplified" in ctx or "non amplified" in ctx or "not amplified" in ctx:
                            her2_found = "HER2-"
                            break
                        elif "negative" in ctx or "1+" in ctx:
                            her2_found = "HER2-"
                            break
                        elif "equivocal" in ctx or "2+" in ctx:
                            her2_found = "HER2 equivocal"
                            break
                        elif "positive" in ctx or "3+" in ctx or "amplified" in ctx:
                            her2_found = "HER2+"
                            break
                        else:
                            her2_found = "HER2: status unclear"
                if her2_found is None:
                    her2_found = "HER2: not tested"
                old_val = type_val
                type_val = type_val.rstrip() + ", " + her2_found
                cancer["Type_of_Cancer"] = type_val
                print(f"    [POST-HER2-CHECK] Appended missing HER2 status: {her2_found}")
                print(f"      before: '{old_val}'")
                print(f"      after:  '{type_val}'")

        # POST-HER2-VERIFY: If note mentions HER2+ drugs but extraction says HER2-, override [breast-only]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val = cancer.get("Type_of_Cancer", "")
            if isinstance(type_val, str) and "her2-" in type_val.lower().replace(" ", ""):
                # Check if note mentions HER2+ treatment
                her2_evidence = []
                for drug in HER2_POS_DRUGS:
                    if drug in note_lower:
                        her2_evidence.append(drug)
                for regimen in HER2_POS_REGIMENS:
                    if re.search(rf'\b{regimen}\b', note_lower):
                        her2_evidence.append(regimen)
                if her2_evidence:
                    old_val = type_val
                    # Replace HER2- with HER2+ (case-insensitive)
                    type_val = re.sub(r'(?i)HER2[\s-]*(?:neg(?:ative)?|-)', 'HER2+', type_val)
                    cancer["Type_of_Cancer"] = type_val
                    print(f"    [POST-HER2-VERIFY] Overrode HER2- → HER2+ (evidence: {her2_evidence})")
                    print(f"      before: '{old_val}'")
                    print(f"      after:  '{type_val}'")

        # POST-RECEPTOR-UPDATE: Update receptor status from Addendum if different from A/P [v22] [breast-only]
        cancer_ru = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer_ru, dict):
            type_val_ru = cancer_ru.get("Type_of_Cancer", "")
            if type_val_ru and isinstance(type_val_ru, str):
                # Search for Addendum section in note
                addendum_match = re.search(r'(?i)Addendum.*?(?:progesterone|estrogen|HER|Ki-?67).*?(?:positive|negative|equivocal)', note_text, re.DOTALL)
                if addendum_match:
                    addendum_text = addendum_match.group(0).lower()
                    # Check PR status in Addendum
                    if 'progesterone receptors is negative' in addendum_text or 'progesterone receptors is negative' in note_text.lower():
                        if 'pr+' in type_val_ru.lower() or 'pr positive' in type_val_ru.lower() or '/PR+/' in type_val_ru:
                            old_type = type_val_ru
                            type_val_ru = re.sub(r'PR\+|PR positive', 'PR-', type_val_ru)
                            cancer_ru["Type_of_Cancer"] = type_val_ru
                            print(f"    [POST-RECEPTOR-UPDATE] PR+ → PR- (Addendum: progesterone negative)")
                            print(f"      before: '{old_type}'")
                            print(f"      after:  '{type_val_ru}'")

        # POST-HER2-FISH: Resolve "HER2: status unclear/not tested" when FISH result exists [v22] [breast-only]
        cancer_hf = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer_hf, dict):
            type_val_hf = cancer_hf.get("Type_of_Cancer", "")
            if type_val_hf and isinstance(type_val_hf, str) and ('unclear' in type_val_hf.lower() or 'not tested' in type_val_hf.lower()):
                # Search note for FISH negative/non-amplified
                fish_neg = re.search(r'(?i)FISH\s+(?:negative|non[- ]?amplified|ratio\s+(?:0\.\d+|1\.\d+))', note_text)
                if fish_neg:
                    old_type = type_val_hf
                    type_val_hf = re.sub(r'HER2:\s*(?:status unclear|not tested)', 'HER2-', type_val_hf)
                    cancer_hf["Type_of_Cancer"] = type_val_hf
                    print(f"    [POST-HER2-FISH] Resolved HER2 status → HER2- (FISH negative found)")
                    print(f"      before: '{old_type}'")
                    print(f"      after:  '{type_val_hf}'")

        # POST-TYPE-VERIFY: Fix HER2+/triple-negative contradiction [v15] [breast-only]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val = cancer.get("Type_of_Cancer", "")
            if isinstance(type_val, str):
                type_lower = type_val.lower()
                if "her2+" in type_lower.replace(" ", "").replace("-", "") and "triple negative" in type_lower:
                    # HER2+ and triple negative are mutually exclusive — HER2+ is more specific
                    old_val = type_val
                    type_val = re.sub(r'(?i),?\s*triple[\s-]*negative', '', type_val).strip().rstrip(',').strip()
                    cancer["Type_of_Cancer"] = type_val
                    print(f"    [POST-TYPE-VERIFY] Removed contradictory 'triple negative' (HER2+ present)")
                    print(f"      before: '{old_val}'")
                    print(f"      after:  '{type_val}'")

        # POST-TYPE-VERIFY-TNBC: If A/P says TNBC, override HER2+ → HER2- [v17] [breast-only]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val = cancer.get("Type_of_Cancer", "")
            if isinstance(type_val, str) and re.search(r'(?i)HER2\s*\+|HER2\s*pos', type_val):
                ap_text_tnbc = (assessment_and_plan or '').lower()  # v18: use local var, not row.get
                tnbc_in_ap = bool(re.search(r'\btnbc\b|triple.negative', ap_text_tnbc))
                if not tnbc_in_ap:
                    # Also check if note has conclusive TNBC language
                    tnbc_in_ap = bool(re.search(
                        r'(?:appears to be|confirmed|is)\s+(?:tnbc|triple.negative)', note_lower))
                if tnbc_in_ap:
                    old_val = type_val
                    type_val = re.sub(r'(?i)HER2\s*\+', 'HER2-', type_val)
                    type_val = re.sub(r'(?i)HER2\s*pos\w*', 'HER2-', type_val)
                    if 'triple negative' not in type_val.lower() and 'tnbc' not in type_val.lower():
                        type_val += ', triple negative'
                    cancer["Type_of_Cancer"] = type_val
                    print(f"    [POST-TYPE-VERIFY-TNBC] A/P says TNBC, overrode HER2+")
                    print(f"      before: '{old_val}'")
                    print(f"      after:  '{type_val}'")

        # POST-TYPE-TNBC-ER: If note says TNBC but Type has "ER+ (inferred from goserelin)", remove the wrong inference [v29] [breast-only]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val_te = cancer.get("Type_of_Cancer", "") or ""
            if isinstance(type_val_te, str) and "ER+" in type_val_te and "inferred from goserelin" in type_val_te.lower():
                ap_text_te = (assessment_and_plan or '').lower()
                note_lower_te = note_text.lower()
                tnbc_found = bool(re.search(r'\btnbc\b|triple.negative', ap_text_te)) or \
                             bool(re.search(r'\btnbc\b|triple.negative', note_lower_te))
                er_neg_found = bool(re.search(r'ER[\s/]*(?:negative|neg\b|-)|ER/PR/.*negative', note_lower_te))
                if tnbc_found or er_neg_found:
                    old_val_te = type_val_te
                    # Remove the "ER+ (inferred from goserelin)" part
                    type_val_te = re.sub(r',?\s*ER\+\s*\(inferred from goserelin\)', '', type_val_te).strip().rstrip(',').strip()
                    cancer["Type_of_Cancer"] = type_val_te
                    print(f"    [POST-TYPE-TNBC-ER] Removed 'ER+ (inferred from goserelin)' — note says TNBC/ER-")
                    print(f"      before: '{old_val_te}'")
                    print(f"      after:  '{type_val_te}'")

        # POST-TYPE-HER2-BREAST-OVERRIDE: If breast biopsy says HER2- but Type says HER2+ (confusion with other cancer) [v29] [breast-only]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val_ho = cancer.get("Type_of_Cancer", "") or ""
            if isinstance(type_val_ho, str) and re.search(r'(?i)HER2\s*\+|HER2\s*pos', type_val_ho):
                # Check if breast cancer biopsy/pathology in A/P explicitly says HER- / FISH negative / ratio < 2.0
                ap_lower_ho = (assessment_and_plan or '').lower()
                breast_her_neg = bool(re.search(
                    r'(?:breast|IDC|ductal).*?(?:HER-|her\s*2\s*neg|FISH\s*(?:ratio\s*)?(?:1\.\d|0\.\d)|'
                    r'HER-?\s*(?:with|,)\s*FISH\s*(?:ratio\s*)?(?:1\.\d))',
                    ap_lower_ho, re.IGNORECASE))
                # Also check for explicit "ER+/PR+/HER-" pattern in A/P
                if not breast_her_neg:
                    breast_her_neg = bool(re.search(
                        r'ER\+/PR\+/HER-|ER\+.*PR\+.*(?:her|HER)\s*(?:-|neg)',
                        assessment_and_plan or '', re.IGNORECASE))
                if breast_her_neg:
                    old_val_ho = type_val_ho
                    type_val_ho = re.sub(r'(?i)HER2\s*\+', 'HER2-', type_val_ho)
                    type_val_ho = re.sub(r'(?i)HER2\s*pos\w*', 'HER2-', type_val_ho)
                    cancer["Type_of_Cancer"] = type_val_ho
                    print(f"    [POST-TYPE-HER2-BREAST-OVERRIDE] Breast biopsy says HER2-, overrode Type HER2+")
                    print(f"      before: '{old_val_ho}'")
                    print(f"      after:  '{type_val_ho}'")

        # POST-TYPE-UNCLEAR: If note says biomarkers/receptors "unclear"/"unknown", correct fabricated receptor status [v23] [breast-only]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val_uc = cancer.get("Type_of_Cancer", "") or ""
            # Check if note explicitly says biomarker results are unclear/unknown
            unclear_match = re.search(r'biomarker\s+results?\s+(?:unclear|unknown|not\s+(?:available|known|reported))', note_lower)
            if unclear_match and type_val_uc:
                # Check if Type contains specific receptor claims (ER+/PR+/HER2+) preceded by "Originally"
                orig_receptor = re.search(r'(?i)originally\s+(ER[+-]|PR[+-]|HER2[+-])', type_val_uc)
                if orig_receptor:
                    old_type = type_val_uc
                    # Replace "Originally ER+/PR+/HER2+" with "Originally unclear receptor status"
                    type_val_uc = re.sub(
                        r'(?i)originally\s+(?:ER[+-]/PR[+-]/HER2[+-]|ER[+-]/PR[+-]|ER[+-])',
                        'Originally unclear receptor status',
                        type_val_uc
                    )
                    cancer["Type_of_Cancer"] = type_val_uc
                    print(f"    [POST-TYPE-UNCLEAR] Note says '{unclear_match.group()}', corrected fabricated receptors")
                    print(f"      before: '{old_type}'")
                    print(f"      after:  '{type_val_uc}'")

        # POST-TYPE-HR-EXPAND: Expand "HR+" to "ER+" when note has specific "estrogen receptor positive" [v23] [breast-only]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if cancer_type == "breast" and isinstance(cancer, dict):
            type_val_hr = cancer.get("Type_of_Cancer", "") or ""
            if re.search(r'\bHR\+', type_val_hr) and 'ER' not in type_val_hr:
                has_er_positive = bool(re.search(r'estrogen\s+receptor\s+(?:positive|is\s+positive)', note_lower))
                has_pr_info = bool(re.search(r'progesterone\s+receptor', note_lower))
                if has_er_positive:
                    old_type = type_val_hr
                    if has_pr_info:
                        pr_positive = bool(re.search(r'progesterone\s+receptor\s+(?:positive|is\s+positive)', note_lower))
                        pr_str = "PR+" if pr_positive else "PR-"
                        type_val_hr = type_val_hr.replace("HR+", f"ER+/{pr_str}")
                    else:
                        type_val_hr = type_val_hr.replace("HR+", "ER+")
                    cancer["Type_of_Cancer"] = type_val_hr
                    print(f"    [POST-TYPE-HR-EXPAND] Expanded HR+ using note receptor details")
                    print(f"      before: '{old_type}'")
                    print(f"      after:  '{type_val_hr}'")

        # POST-GENETICS-RECHECK: Clear genetic_testing_plan if it only contains recheck/LVEF/marker language [v24] [breast-only]
        gen_test = keypoints.get("Genetic_Testing_Plan", {})
        if cancer_type == "breast" and isinstance(gen_test, dict):
            gen_val = gen_test.get("genetic_testing_plan", "")
            if gen_val and isinstance(gen_val, str) and gen_val.lower() not in ("none planned.", "none", ""):
                gen_lower = gen_val.lower()
                # Check if it's a non-genetic recheck
                is_recheck = any(x in gen_lower for x in ["recheck", "would recheck", "repeat echo", "repeat lvef"])
                has_genetic_keyword = any(x in gen_lower for x in [
                    "oncotype", "mammaprint", "foundation", "guardant", "brca", "germline",
                    "genetic testing", "genetic counseling", "molecular", "ngs", "panel",
                    "ctdna", "liquid biopsy", "msi", "pd-l1", "tumor profiling"
                ])
                if is_recheck and not has_genetic_keyword:
                    gen_test["genetic_testing_plan"] = "None planned."
                    print(f"    [POST-GENETICS-RECHECK] Cleared non-genetic recheck: '{gen_val[:60]}'")

        # POST-STAGE-FINAL: Final consistency check — Stage IV must agree with Distant Metastasis [v29]
        # This runs AFTER all other POST hooks to catch cases where earlier hooks
        # created inconsistencies (e.g., POST-STAGE-DISTMET downgraded but POST-STAGE-METASTATIC re-upgraded,
        # or Distant Metastasis was corrected by gates after Stage was already downgraded)
        cancer_final = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_final, dict):
            stage_final = cancer_final.get("Stage_of_Cancer", "") or ""
            met_final = cancer_final.get("Metastasis", "") or ""
            dist_final = cancer_final.get("Distant Metastasis", "") or ""
            stage_iv_final = bool(re.search(r'stage\s*iv|metastatic', stage_final, re.IGNORECASE))
            dist_no_final = dist_final.lower().startswith("no") if dist_final else True

            # Case 1: Stage says IV but Distant Met says No — downgrade
            if stage_iv_final and dist_no_final:
                met_lower_f = met_final.lower() if met_final else ""
                DISTANT_SITES_F = ["liver", "lung", "bone", "brain", "pleural", "peritoneal",
                                   "ovary", "skin", "distant", "hepatic", "pulmonary", "osseous", "cerebral"]
                has_distant_f = any(ds in met_lower_f for ds in DISTANT_SITES_F)
                if not has_distant_f:
                    cleaned_f = re.sub(r'(?i)\bStage\s*IV\s*\(?\s*metastatic\s*\)?', 'Stage III', stage_final)
                    cleaned_f = re.sub(r'(?i)\bmetastatic\s*\(?\s*Stage\s*IV\s*\)?', 'Stage III', cleaned_f)
                    cleaned_f = re.sub(r'(?i)\bStage\s*IV\b', 'Stage III', cleaned_f)
                    cleaned_f = cleaned_f.strip().rstrip(',').strip()
                    if cleaned_f and cleaned_f != stage_final:
                        cancer_final["Stage_of_Cancer"] = cleaned_f
                        print(f"    [POST-STAGE-FINAL] Stage IV but Distant Met=No (final check): '{stage_final}' → '{cleaned_f}'")

            # Case 2: Stage was downgraded to III but Distant Met says Yes — re-upgrade
            if not stage_iv_final and "stage iii" in stage_final.lower():
                dist_yes = dist_final.lower().startswith("yes") if dist_final else False
                if dist_yes:
                    # Distant Met says Yes but Stage says III — this is inconsistent, upgrade back
                    cleaned_f = re.sub(r'(?i)\bStage\s*III\b', 'Stage IV', stage_final)
                    if cleaned_f != stage_final:
                        cancer_final["Stage_of_Cancer"] = cleaned_f
                        print(f"    [POST-STAGE-FINAL] Stage III but Distant Met=Yes: '{stage_final}' → '{cleaned_f}'")

        # Source attribution — find evidence quotes for each extracted field
        attribution = {}
        if config.get("extraction", {}).get("attribution", False):
            attr_gen_config = keypoint_config.copy()
            attr_gen_config["max_new_tokens"] = 128
            attr_start = time.time()
            attribution = attribute_row(
                note_text, keypoints, model, tokenizer,
                chat_tmpl, attr_gen_config, base_cache
            )
            attributable = get_attributable_fields(keypoints)
            print(f"  [ATTRIBUTION] {len(attribution)}/{len(attributable)} fields sourced ({time.time() - attr_start:.1f}s)")

        # Patient letter generation — tagged with [source:field_name]
        letter = ""
        traceability = {}
        if config.get("extraction", {}).get("letter", False) and "letter_generation" in config.get("_prompts", {}):
            letter_prompt_template = config["_prompts"]["letter_generation"]["patient_letter"]
            letter_gen_config = keypoint_config.copy()
            letter_gen_config["max_new_tokens"] = 768
            letter_start = time.time()
            tagged_text = generate_tagged_letter(
                keypoints, model, tokenizer, chat_tmpl,
                letter_gen_config, fullnote_cache, letter_prompt_template,
                note_text=note_text,
            )
            traceability = parse_tagged_letter(tagged_text, keypoints, attribution)
            letter = traceability.get("letter_text", "")
            letter, post_warnings = post_check_letter(letter)
            # POST-LETTER-EMOTIONAL: Remove fabricated emotional sentences when no emotional_context [v24]
            emotional_phrases = [
                "you appear to be emotional",
                "we understand you are feeling emotional",
                "we understand that you are feeling anxious and depressed",
                "we understand that you are feeling anxious",
                "you seem to be feeling emotional",
            ]
            has_emotional_context = bool(keypoints.get("emotional_context"))
            if not has_emotional_context:
                letter_lines = letter.split("\n")
                cleaned_lines = []
                for line in letter_lines:
                    if any(ep in line.lower() for ep in emotional_phrases):
                        print(f"  [POST-LETTER-EMOTIONAL] Removed fabricated emotional line: '{line[:80]}'")
                    else:
                        cleaned_lines.append(line)
                letter = "\n".join(cleaned_lines)
            letter, _ = post_fix_letter(letter)
            # Letter Faithfulness Gate: verify letter against original note
            letter, faith_log = verify_letter_faithfulness(
                letter, note_text, model, tokenizer, chat_tmpl,
                gen_config, fullnote_cache,
            )
            for fl in faith_log:
                print(f"  {fl}")
            traceability["letter_text"] = letter
            for w in post_warnings:
                print(f"  {w}")
            n_sentences = len(traceability.get("sentences", []))
            n_attributed = sum(
                1 for s in traceability.get("sentences", [])
                if s["source_fields"] != ["unattributed"] and s["source_fields"] != ["none"]
            )
            note_grade = flesch_kincaid_grade(note_text)
            if "metrics" in traceability:
                traceability["metrics"]["note_readability_grade"] = note_grade
            metrics = traceability.get("metrics", {})
            print(f"  [LETTER] {n_sentences} sent, {n_attributed} attr, "
                  f"grade={metrics.get('readability_grade', '?')}/{note_grade} "
                  f"cov={metrics.get('field_coverage_pct', '?')}% "
                  f"chain={metrics.get('attribution_complete_pct', '?')}% "
                  f"({time.time() - letter_start:.1f}s)")

        print(f"  Row {index} total: {time.time() - row_start:.1f}s")

        # Build row result
        row_result = {
            "coral_idx": int(row["coral_idx"]),
            "note_text": note_text,
            "assessment_and_plan": assessment_and_plan,
            "keypoints": keypoints,
            "attribution": attribution,
            "letter": letter,
            "traceability": traceability,
        }

        # Append to results.txt
        append_row_result(results_path, index, row_result)

        # Normalize: convert any list values in keypoints to comma-separated strings [v22]
        for section_key, section_val in keypoints.items():
            if isinstance(section_val, dict):
                for field_key, field_val in section_val.items():
                    if isinstance(field_val, list):
                        section_val[field_key] = ", ".join(str(v) for v in field_val)

        # Update progress
        completed_indices.add(index)
        current_progress["completed_indices"] = sorted(completed_indices)
        current_progress["results"][str(index)] = row_result
        save_progress(run_dir, current_progress)
        print(f"  Row {index} saved to progress")

    # 12. Mark completed
    current_progress["completed"] = True
    save_progress(run_dir, current_progress)

    total_time = time.time() - global_start
    print(f"\nTotal processing time: {total_time:.1f}s ({total_time/60:.1f}min)")

    # 13. Copy results.txt to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(results_path, os.path.join(project_root, "results.txt"))
    print(f"Done! Results copied to ./results.txt")
    print(f"Full results in: {run_dir}/")


if __name__ == "__main__":
    main()
