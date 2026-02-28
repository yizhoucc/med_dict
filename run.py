#!/usr/bin/env python3
"""
Experiment runner for medical note extraction pipeline.

Usage:
    python run.py exp/default.yaml
    python run.py exp/default.yaml --resume results/default_20260228_103000/
"""

import argparse
import hashlib
import json
import os
import random
import shutil
from datetime import datetime

import torch
import yaml
import pandas as pd

from ult import (
    AutoModelForCausalLM,
    AutoTokenizer,
    build_base_cache,
    extract_and_verify,
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


def extract_assessment_plan(note_text, model, tokenizer, config):
    """Extract assessment/plan section with retries and LLM verification."""
    max_retries = config.get("extraction", {}).get("max_retries", 3)
    gen_config_greedy = config["generation"]["assessment_plan"].copy()
    gen_config_greedy["eos_token_id"] = tokenizer.eos_token_id
    gen_config_retry = config["generation"]["retry"].copy()
    gen_config_retry["eos_token_id"] = tokenizer.eos_token_id

    for attempt in range(max_retries):
        if attempt == 0:
            current_prompt = (
                "here is a medical note \n\n"
                + note_text
                + "\n\n "
                'now, return me all the ORIGINAL TEXT (do not do any modifications, do no rephrase and summarize) '
                'after the words like "Assessment and Plan" or "Assessment/Plan". '
                "ignore anything before that. ignore the line breaking characters."
            )
            current_config = gen_config_greedy
        else:
            current_prompt = (
                "The previous attempt failed because it was not an exact copy. "
                'Your task is to COPY-PASTE the "Assessment and Plan" section.'
                "\n\nSOURCE TEXT:\n"
                + note_text
                + "\n\n"
                "INSTRUCTION: Extract the Assessment and Plan section exactly as written. "
                "Do not summarize. Do not fix grammar. Do not change punctuation."
            )
            current_config = gen_config_retry

        print(f"  Extraction Attempt {attempt + 1}...")
        candidate_text, _ = run_model(
            current_prompt, model, tokenizer, current_config
        )

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
            print(f"  Success on attempt {attempt + 1} (Verified by LLM)")
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

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        device_map=model_cfg.get("device_map", "auto"),
        dtype=torch_dtype,
    )
    print("Model loaded.")

    # 6. Load data
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["dataset_path"])
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
    print(f"\nProcessing {len(df)} rows...")
    for index, row in df.iterrows():
        if index in completed_indices:
            print(f"Skipping row {index} (already completed)")
            continue

        print(f"\nProcessing row {index} ({index - row_range[0] + 1}/{len(df)})...")

        note_text = row["note_text"]

        # Extract assessment/plan with retries
        assessment_and_plan = extract_assessment_plan(
            note_text, model, tokenizer, config
        )

        # Extract keypoints from full note
        base_cache = build_base_cache(note_text, model, tokenizer)
        keypoints = extract_and_verify(
            extraction_prompts, model, tokenizer, keypoint_config, base_cache, verify=verify
        )
        print("  Keypoints from extraction_prompts done")

        # Extract plan keypoints from assessment/plan section
        if assessment_and_plan is not None:
            base_cache = build_base_cache(assessment_and_plan, model, tokenizer)
            plan_keypoints = extract_and_verify(
                plan_extraction_prompts,
                model,
                tokenizer,
                keypoint_config,
                base_cache,
                verify=verify,
            )
            keypoints.update(plan_keypoints)
        print("  Keypoints from plan_extraction_prompts done")

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

    # 13. Copy results.txt to project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    shutil.copy2(results_path, os.path.join(project_root, "results.txt"))
    print(f"\nDone! Results copied to ./results.txt")
    print(f"Full results in: {run_dir}/")


if __name__ == "__main__":
    main()
