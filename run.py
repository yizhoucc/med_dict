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
    build_base_cache,
    extract_and_verify,
    extract_and_verify_v2,
    run_model,
    gc,
)


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
        r'(?:Assessment|ASSESSMENT)\s*(?:/|and|&|\\)\s*(?:Plan|PLAN|Recommendations|RECOMMENDATIONS)\s*:?',
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


def extract_assessment_plan(note_text, model, tokenizer, config):
    """Extract assessment/plan section: regex first, LLM fallback with chat template."""

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
            current_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a medical text extraction tool. You extract sections from medical notes exactly as written. "
                f"Return ONLY the extracted text, nothing else. No commentary, no 'here is the text', no repetition."
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"Here is a medical note:\n\n{note_text}\n\n"
                f"Extract the 'Assessment and Plan' or 'Assessment/Plan' section. "
                f"Return ONLY the original text from that section to the end of the note. "
                f"Do not modify, rephrase, or summarize. Do not repeat the text."
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
            current_config = gen_config_greedy
        else:
            current_prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a copy-paste tool. Return ONLY the requested section, exactly as written. "
                f"No commentary. No repetition. Stop after the section ends."
                f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"SOURCE TEXT:\n{note_text}\n\n"
                f"Copy-paste the Assessment and Plan section exactly as written. "
                f"Do not summarize. Do not fix grammar. Do not repeat."
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
            current_config = gen_config_retry

        print(f"  LLM Extraction Attempt {attempt + 1}...")
        candidate_text, _ = run_model(
            current_prompt, model, tokenizer, current_config
        )

        # --- Step C: Truncate repeated text ---
        candidate_text = truncate_repeated_text(candidate_text)

        # LLM sanity check
        sanity_check_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a strict text auditor. Your job is to verify data fidelity."
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            f"I have a SOURCE text and an EXTRACTED snippet. \n"
            f"Task: Verify if the EXTRACTED snippet appears in the SOURCE text.\n"
            f"Rules:\n"
            f"1. Ignore differences in whitespace (newlines, tabs, spaces).\n"
            f"2. Ignore minor formatting (bullet points vs dashes).\n"
            f"3. STRICTLY FAIL if the text is summarized, reworded, or contains new words.\n\n"
            f"--- SOURCE TEXT START ---\n{note_text}\n--- SOURCE TEXT END ---\n\n"
            f"--- EXTRACTED SNIPPET START ---\n{candidate_text}\n--- EXTRACTED SNIPPET END ---\n\n"
            f'Does the snippet match? Reply with exactly one JSON object: {{"match": true}} or {{"match": false}}.'
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
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


def main():
    parser = argparse.ArgumentParser(description="Run medical extraction experiment")
    parser.add_argument("config", help="Path to experiment YAML config")
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a previous run directory to resume from",
    )
    args = parser.parse_args()

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

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        device_map=model_cfg.get("device_map", "auto"),
        dtype=torch_dtype,
        quantization_config=quantization_config,
    )
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
    print(f"Using pipeline: {pipeline}")

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

    # 11. Main loop
    global_start = time.time()
    print(f"\nProcessing {len(df)} rows...")
    for index, row in df.iterrows():
        if index in completed_indices:
            print(f"Skipping row {index} (already completed)")
            continue

        row_start = time.time()
        print(f"\nProcessing row {index} ({list(df.index).index(index) + 1}/{len(df)})...")

        note_text = row["note_text"]

        # Extract assessment/plan with retries
        ap_start = time.time()
        assessment_and_plan = extract_assessment_plan(
            note_text, model, tokenizer, config
        )
        print(f"  A/P extraction: {time.time() - ap_start:.1f}s")

        # Extract keypoints from full note
        ext_start = time.time()
        base_cache = build_base_cache(note_text, model, tokenizer)
        keypoints = extract_fn(
            extraction_prompts, model, tokenizer, keypoint_config, base_cache, verify=verify
        )
        print(f"  Extraction prompts: {time.time() - ext_start:.1f}s")

        # Extract plan keypoints from assessment/plan section
        if assessment_and_plan is not None:
            plan_start = time.time()
            base_cache = build_base_cache(assessment_and_plan, model, tokenizer)
            plan_keypoints = extract_fn(
                plan_extraction_prompts,
                model,
                tokenizer,
                keypoint_config,
                base_cache,
                verify=verify,
            )
            keypoints.update(plan_keypoints)
            print(f"  Plan extraction prompts: {time.time() - plan_start:.1f}s")

        print(f"  Row {index} total: {time.time() - row_start:.1f}s")

        # Build row result
        row_result = {
            "coral_idx": int(row["coral_idx"]),
            "note_text": note_text,
            "assessment_and_plan": assessment_and_plan,
            "keypoints": keypoints,
        }

        # Append to results.txt
        append_row_result(results_path, index, row_result)

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
