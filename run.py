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

        # POST-GENETICS: Remove mutation findings from Genetics referral field [B70]
        # Prompt says "do NOT list genetic test RESULTS or known mutations" but model
        # sometimes still puts "BRCA1 mutation" etc. in the Genetics referral field.
        if isinstance(referral, dict):
            gen_val = referral.get("Genetics", "None") or "None"
            if gen_val != "None":
                gen_lower = gen_val.lower()
                # If it mentions mutation/carrier/positive/negative but NOT refer/consult → it's a finding, not a referral
                has_finding = any(w in gen_lower for w in [
                    "mutation", "carrier", "positive", "negative", "variant",
                    "pathogenic", "wild type", "wild-type", "detected", "identified",
                ])
                has_referral = any(w in gen_lower for w in [
                    "refer", "consult", "counseling", "counsel", "evaluation",
                    "recommend", "genetic testing", "send for",
                ])
                if has_finding and not has_referral:
                    referral["Genetics"] = "None"
                    print(f"    [POST-GENETICS] cleared finding from referral: '{gen_val}'")

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
                'Brain MRI': ['brain mri'],
                'MRI Breast': ['mri breast'],
                'Ultrasound': ['ultrasound'],
            }
            # Search A/P text for imaging ordered as standalone items or with future tense
            search_text = assessment_and_plan if assessment_and_plan else note_text
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
                if re.search(regex, search_text, re.IGNORECASE):
                    if img_lower in ("no imaging planned.", "no imaging planned", "none", "none planned.", ""):
                        img["imaging_plan"] = label
                    else:
                        img["imaging_plan"] = img_val + ". " + label
                    img_val = img["imaging_plan"]
                    img_lower = img_val.lower()
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

        # POST-DRUG-VERIFY: Remove hallucinated drugs not found in original note text
        note_lower = note_text.lower()
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

        # POST-ER-CHECK: Infer ER status from medications when Type_of_Cancer lacks it [v16]
        ER_POS_DRUGS = ["tamoxifen", "letrozole", "anastrozole", "exemestane", "arimidex",
                        "femara", "aromasin", "fulvestrant", "faslodex",
                        "goserelin", "zoladex", "leuprolide", "lupron"]
        cancer = keypoints.get("Cancer_Diagnosis", {})
        if isinstance(cancer, dict):
            type_val = cancer.get("Type_of_Cancer", "")
            if isinstance(type_val, str) and not re.search(r'(?i)\b(?:ER|HR|PR)[+-]|\b(?:ER|HR|PR)\s+(?:positive|negative)|estrogen|progesterone|hormone\s+receptor|triple', type_val):
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
                        type_val = type_val.rstrip() + ", ER+ (inferred from " + drug + ")"
                        cancer["Type_of_Cancer"] = type_val
                        er_inferred = True
                        print(f"    [POST-ER-CHECK] Inferred ER+ from drug: {drug}")
                        break
                # Check note for explicit ER mention if not found from drugs
                if not er_inferred:
                    er_match = re.search(r'(?i)\b(ER|estrogen\s+receptor)\s*[\s:]*\s*(positive|\+|negative|-)', note_lower_er)
                    if er_match:
                        status = "+" if er_match.group(2) in ("positive", "+") else "-"
                        type_val = type_val.rstrip() + f", ER{status}"
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
