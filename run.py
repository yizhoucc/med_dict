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
    gc,
)
from source_attribution import attribute_row, get_attributable_fields


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

        keypoints = extract_fn(
            phase1_prompts, model, tokenizer, keypoint_config, base_cache,
            verify=verify, chat_tmpl=chat_tmpl, oncology_whitelist=whitelist,
            gate_config=gate_config, supportive_whitelist=supp_whitelist
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
                if cross_context:
                    phase2_prompts[k] = cross_context + "\n\n" + extraction_prompts[k]
                else:
                    phase2_prompts[k] = extraction_prompts[k]

            phase2_keypoints = extract_fn(
                phase2_prompts, model, tokenizer, keypoint_config, base_cache,
                verify=verify, chat_tmpl=chat_tmpl, oncology_whitelist=whitelist,
                gate_config=gate_config, supportive_whitelist=supp_whitelist
            )
            keypoints.update(phase2_keypoints)
            print(f"  Phase 2 extraction ({len(phase2_prompts)} prompts): {time.time() - phase2_start:.1f}s")

        print(f"  Total extraction: {time.time() - ext_start:.1f}s")

        # Extract plan keypoints from assessment/plan section
        # Pop Referral — it needs full note context, not just A/P [v14]
        referral_prompt = plan_extraction_prompts.pop("Referral", None)

        if assessment_and_plan is not None:
            plan_start = time.time()
            base_cache = build_base_cache(model_ap, model, tokenizer, defs_context, chat_tmpl=chat_tmpl)
            plan_keypoints = extract_fn(
                plan_extraction_prompts,
                model,
                tokenizer,
                keypoint_config,
                base_cache,
                verify=verify,
                chat_tmpl=chat_tmpl,
                oncology_whitelist=whitelist,
                gate_config=gate_config,
                supportive_whitelist=supp_whitelist,
            )
            keypoints.update(plan_keypoints)
            print(f"  Plan extraction prompts: {time.time() - plan_start:.1f}s")

        # Extract Referral from full note (not just A/P) [v14]
        if referral_prompt:
            ref_start = time.time()
            ref_keypoints = extract_fn(
                {"Referral": referral_prompt},
                model, tokenizer, keypoint_config, fullnote_cache,
                verify=verify, chat_tmpl=chat_tmpl, oncology_whitelist=whitelist,
                gate_config=gate_config, supportive_whitelist=supp_whitelist,
            )
            keypoints.update(ref_keypoints)
            print(f"  Referral extraction (full note): {time.time() - ref_start:.1f}s")
            # Restore the prompt for next iteration
            plan_extraction_prompts["Referral"] = referral_prompt

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
            if lab_val and lab_val.lower() not in ("no labs planned.", "no labs planned", "none", ""):
                lab_lower = lab_val.lower()
                LAB_IMAGING_TERMS = [
                    "doppler", "ultrasound", "ct ", "ct,", "mri", "pet", "dexa",
                    "bone scan", "x-ray", "xray", "mammogram", "echocardiogram",
                    "echo ", "scan ",
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
            lab_val_s = (lab_search.get("lab_plan", "") or "").strip().lower()
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
                    if not any(term in therapy_val.lower() for term in whitelist):
                        therapy["therapy_plan"] = "None"
                        print(f"    [POST-THERAPY] cleared (no oncology drugs found): {therapy_val[:80]}")
                    # else: has drug names but mixed with non-therapy context → keep original

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
                'bone density': 'DEXA scan',
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
                'DEXA scan': ['dexa', 'bone density'],
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
                    r'ordered?\s+(?:a\s+)?|need\s+(?:a\s+)?)'
                    r'[^.;]{0,30}' + re.escape(pattern)
                    + r'|' + re.escape(pattern) + r'\.?\s*(?:Port|$|\d)'
                )
                # Search A/P first, then full note if plan is empty
                found = re.search(regex, search_text, re.IGNORECASE)
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
            proc_lower = (proc_val or "").lower()
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

        # POST-PROCEDURE-FILTER: Remove non-procedure items from Procedure_Plan [v14]
        # Items like IHC, FISH, Oncotype, BRCA belong in genetic_testing_plan, not procedure
        proc = keypoints.get("Procedure_Plan", {})
        if isinstance(proc, dict):
            proc_val = proc.get("procedure_plan", "") or ""
            if proc_val and proc_val.lower() not in ("no procedures planned.", "no procedures planned", "none", "none planned.", ""):
                PROC_BLACKLIST = [
                    "ihc", "fish", "receptor testing", "staining",
                    "oncotype", "mammaprint", "brca", "genomic", "molecular",
                    "genetic testing", "gene panel", "ngs", "next generation",
                    "foundation one", "foundationone", "guardant",
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
            code_match = re.search(
                r'(?:code\s+status[:\s]*|advance\s+care\s+planning[:\s]*)(full\s+code|dnr/?dni|dnr|dni|comfort\s+measures)',
                note_text, re.IGNORECASE
            )
            living_will = re.search(r'living\s+will', note_text, re.IGNORECASE)
            patches = []
            if code_match:
                patches.append(code_match.group(1).strip())
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

            # POST-STAGE-REGIONAL: Stage IV with only regional LN metastasis [v16]
            # Axillary, sentinel, supraclavicular, infraclavicular, internal mammary LN
            # are regional (Stage III) not distant (Stage IV)
            if stage_says_iv and met and not met_says_no:
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

        # POST-STAGE-VERIFY: Check for hallucinated "Originally Stage X" [v14]
        # Model sometimes fabricates original stage (e.g., "Originally Stage IIA") when note
        # only mentions metastatic recurrence without specifying the original stage.
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            stage = cancer.get("Stage_of_Cancer", "")
            if stage and isinstance(stage, str):
                orig_match = re.search(r'Originally\s+Stage\s+([A-Za-z0-9]+)', stage, re.IGNORECASE)
                if orig_match:
                    claimed_stage = orig_match.group(1)  # e.g., "IIA"
                    # Check if the original note actually mentions this stage
                    # Search for patterns like "Stage IIA", "stage IIA", "T2N0" etc.
                    stage_pattern = re.compile(
                        rf'(?:stage\s+{re.escape(claimed_stage)}|'
                        rf'diagnosed\s+(?:at|as|with)\s+stage\s+{re.escape(claimed_stage)})',
                        re.IGNORECASE
                    )
                    if not stage_pattern.search(note_text):
                        # Original stage not found in note — remove the "Originally Stage X" part
                        cleaned = re.sub(
                            r'Originally\s+Stage\s+[A-Za-z0-9]+,?\s*',
                            '', stage, flags=re.IGNORECASE
                        ).strip().lstrip(',').strip()
                        if cleaned:
                            cancer["Stage_of_Cancer"] = cleaned
                            print(f"    [POST-STAGE-VERIFY] removed unsupported 'Originally Stage {claimed_stage}': '{stage}' → '{cleaned}'")
                        else:
                            cancer["Stage_of_Cancer"] = "Metastatic (Stage IV)"
                            print(f"    [POST-STAGE-VERIFY] removed unsupported 'Originally Stage {claimed_stage}': '{stage}' → 'Metastatic (Stage IV)'")

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

        # POST-GOALS: adjuvant → curative for non-metastatic [B45]
        goals = keypoints.get("Treatment_Goals", {})
        if isinstance(goals, dict):
            goal_val = goals.get("goals_of_treatment", "").lower().strip()
            if goal_val == "adjuvant":
                cancer = keypoints.get("Cancer_Diagnosis", {})
                met = str(cancer.get("Metastasis", "")).lower() if isinstance(cancer, dict) else ""
                stage = str(cancer.get("Stage_of_Cancer", "")).lower() if isinstance(cancer, dict) else ""
                is_metastatic = "yes" in met or "stage iv" in stage or "metastatic" in stage
                if not is_metastatic:
                    goals["goals_of_treatment"] = "curative"
                    print(f"    [POST-GOALS] adjuvant → curative (non-metastatic)")

        # POST-DISTMET: Ensure Distant Metastasis field exists [B48]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict) and "Distant Metastasis" not in cancer:
            met = cancer.get("Metastasis", "")
            cancer["Distant Metastasis"] = met
            print(f"    [POST-DISTMET] added Distant Metastasis: '{met}'")

        # POST-DISTMET-REGIONAL: correct Distant Metastasis if only regional sites [v17]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
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

        # POST-DISTMET-DEFAULT: fill empty Distant Metastasis with "No" when goals=curative [v22]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            dist_met_val = (cancer.get("Distant Metastasis", "") or "").strip()
            goals_val = keypoints.get("Treatment_Goals", {}).get("goals_of_treatment", "") if isinstance(keypoints.get("Treatment_Goals"), dict) else ""
            stage_val = (cancer.get("Stage_of_Cancer", "") or "").lower()
            if not dist_met_val and goals_val == "curative" and "iv" not in stage_val and "metastatic" not in stage_val:
                cancer["Distant Metastasis"] = "No"
                print(f"    [POST-DISTMET-DEFAULT] Filled empty Distant Metastasis → 'No' (curative, non-metastatic)")

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
                on_treatment = bool(re.search(
                    r'(?:currently on|continue|continuing|cycle \d|on \w+(?:oxifen|zole|mab|lib|nib))',
                    ap_lower_rt))
                has_meds = bool((keypoints.get("Current_Medications", {}).get("current_meds", "") or "").strip())
                if on_treatment or has_meds:
                    resp["response_assessment"] = "On treatment; response assessment not available from current visit."
                    print(f"    [POST-RESPONSE-TREATMENT] Corrected 'Not yet on treatment' → on treatment")

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
            "tamoxifen", "letrozole", "anastrozole", "exemestane", "fulvestrant",
            "trastuzumab", "pertuzumab", "herceptin", "perjeta", "doxorubicin",
            "cyclophosphamide", "paclitaxel", "docetaxel", "carboplatin",
            "capecitabine", "xeloda", "gemcitabine", "eribulin", "vinorelbine",
            "palbociclib", "ribociclib", "abemaciclib", "everolimus",
            "olaparib", "talazoparib", "sacituzumab", "tucatinib",
            "zoledronic", "zometa", "denosumab", "xgeva", "reclast",
            "ondansetron", "zofran", "granisetron", "prochlorperazine",
            "dexamethasone", "filgrastim", "pegfilgrastim", "neulasta",
            "epoetin", "darbepoetin",
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
                        "ac", "tc", "fec", "caf", "tac", "tchp", "thp", "folfox", "folfiri",
                        "doxorubicin", "cyclophosphamide", "paclitaxel", "docetaxel", "taxol",
                        "taxotere", "carboplatin", "cisplatin", "gemcitabine", "gemzar",
                        "capecitabine", "xeloda", "irinotecan", "eribulin", "vinorelbine",
                        "pertuzumab", "perjeta", "trastuzumab", "herceptin",
                        "pembrolizumab", "keytruda", "atezolizumab", "tecentriq",
                        "olaparib", "lynparza", "palbociclib", "ibrance",
                        "ribociclib", "kisqali", "abemaciclib", "verzenio",
                        "fulvestrant", "faslodex", "lupron", "leuprolide", "goserelin", "zoladex",
                    ]
                    PAST_CHEMO = ["previously on", "prior", "completed", "finished", "was on",
                                  "had received", "history of", "s/p"]
                    found_chemo = []
                    for pattern in IV_CHEMO_PATTERNS:
                        for m in re.finditer(pattern, ap_lower_iv):
                            drug = m.group(1).strip().lower()
                            if drug in KNOWN_CHEMO_IV:
                                # Exclude past-tense mentions
                                start = max(0, m.start() - 30)
                                before = ap_lower_iv[start:m.start()]
                                if any(pc in before for pc in PAST_CHEMO):
                                    continue
                                found_chemo.append(drug)
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

        # POST-ER-CHECK: Infer ER status from medications when Type_of_Cancer lacks it [v16]
        ER_POS_DRUGS = ["tamoxifen", "letrozole", "anastrozole", "exemestane", "arimidex",
                        "femara", "aromasin", "fulvestrant", "faslodex",
                        "goserelin", "zoladex", "leuprolide", "lupron"]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
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

        # POST-HER2-CHECK: If Type_of_Cancer has ER/PR but no HER2, search note and append [v15]
        HER2_POS_DRUGS = ["trastuzumab", "pertuzumab", "herceptin", "t-dm1", "t-dxd",
                          "ado-trastuzumab", "lapatinib", "tykerb", "tucatinib"]
        HER2_POS_REGIMENS = ["tchp", "thp", "ac-thp", "acthp"]
        HER2_SEARCH_KEYWORDS = [
            "her2", "her-2", "her2neu", "her 2", "her 2 neu",
            "ihc", "fish ratio",
        ] + HER2_POS_DRUGS + HER2_POS_REGIMENS
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
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

        # POST-HER2-VERIFY: If note mentions HER2+ drugs but extraction says HER2-, override
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
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

        # POST-RECEPTOR-UPDATE: Update receptor status from Addendum if different from A/P [v22]
        cancer_ru = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_ru, dict):
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

        # POST-HER2-FISH: Resolve "HER2: status unclear/not tested" when FISH result exists [v22]
        cancer_hf = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer_hf, dict):
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

        # POST-TYPE-VERIFY: Fix HER2+/triple-negative contradiction [v15]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
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

        # POST-TYPE-VERIFY-TNBC: If A/P says TNBC, override HER2+ → HER2- [v17]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
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

        # POST-TYPE-UNCLEAR: If note says biomarkers/receptors "unclear"/"unknown", correct fabricated receptor status [v23]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
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

        # POST-TYPE-HR-EXPAND: Expand "HR+" to "ER+" when note has specific "estrogen receptor positive" [v23]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
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

        print(f"  Row {index} total: {time.time() - row_start:.1f}s")

        # Build row result
        row_result = {
            "coral_idx": int(row["coral_idx"]),
            "note_text": note_text,
            "assessment_and_plan": assessment_and_plan,
            "keypoints": keypoints,
            "attribution": attribution,
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
