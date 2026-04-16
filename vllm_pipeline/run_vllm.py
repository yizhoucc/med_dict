"""
vLLM Pipeline Runner — standalone extraction pipeline using vLLM API.

This is a simplified version of run.py that uses vLLM HTTP API instead of
direct HuggingFace model loading. It reuses the same prompts, post-processing
hooks, and output format.

Usage:
    1. Start vLLM server: bash vllm_pipeline/start_vllm.sh
    2. Run pipeline: python vllm_pipeline/run_vllm.py exp/v32_vllm.yaml
"""

import sys
import os

# Add parent dir to path so we can import from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import pandas as pd
import json
import time
import re
from datetime import datetime
from typing import Dict

from vllm_pipeline.vllm_client import VLLMClient
from vllm_pipeline.inference import build_base_prompt, vllm_generate
from ult import (
    ChatTemplate,
    try_parse_json,
    extract_schema_keys,
)


def load_config(config_path: str) -> dict:
    """Load YAML config and resolve prompt files."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load prompt files
    config["_prompts"] = {}
    prompts_cfg = config.get("prompts", {})
    for key in ["extraction", "plan_extraction", "letter_generation"]:
        path = prompts_cfg.get(key)
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                config["_prompts"][key] = yaml.safe_load(f)

    return config


def extract_keypoint(
    task_prompt: str,
    client: VLLMClient,
    gen_config: Dict,
    base_prompt: str,
    chat_tmpl: ChatTemplate,
) -> dict:
    """
    Extract a single keypoint section using vLLM.

    Args:
        task_prompt: The extraction task prompt text
        client: VLLMClient
        gen_config: Generation config
        base_prompt: Base prompt string (system + note)
        chat_tmpl: ChatTemplate instance

    Returns:
        Parsed JSON dict of extracted keypoints
    """
    # Format as a new user turn
    formatted_prompt = chat_tmpl.user_assistant(task_prompt)

    # Generate
    result, _ = vllm_generate(formatted_prompt, client, gen_config, base_prompt)

    # Parse JSON from result
    parsed = try_parse_json(result)
    if parsed is None:
        # Try to fix JSON format
        fix_prompt = chat_tmpl.user_assistant(
            f"The following text should be valid JSON but has errors. "
            f"Fix it and return ONLY the corrected JSON:\n{result}"
        )
        fix_result, _ = vllm_generate(fix_prompt, client, gen_config, base_prompt)
        parsed = try_parse_json(fix_result)

    return parsed if parsed else {}


def extract_assessment_plan_regex(note_text: str) -> str:
    """Extract Assessment/Plan section using regex (same as run.py)."""
    patterns = [
        r'(?:Assessment\s*(?:and|&|/|\\)?\s*Plan|A\s*/\s*P|IMP\s*(?:RESSION)?|ASSESSMENT\s*(?:AND|&)?\s*PLAN|REC(?:OMMENDATIONS)?)\s*[:.]?\s*\n',
    ]
    for pattern in patterns:
        match = re.search(pattern, note_text, re.IGNORECASE)
        if match:
            return note_text[match.start():]
    return ""


def main():
    parser = argparse.ArgumentParser(description="vLLM Pipeline Runner")
    parser.add_argument("config", help="Path to YAML config file")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    vllm_cfg = model_cfg.get("vllm", {})

    # Create vLLM client
    client = VLLMClient(
        base_url=vllm_cfg.get("base_url", "http://localhost:8000/v1"),
        model_name=model_cfg["name"],
    )

    # Health check
    print("Checking vLLM server...")
    if not client.health_check():
        print("ERROR: vLLM server not reachable. Start it first with: bash vllm_pipeline/start_vllm.sh")
        sys.exit(1)
    print("vLLM server OK")

    # Chat template
    chat_tmpl = ChatTemplate(model_cfg.get("chat_template", "qwen2"))

    # Load data
    data_cfg = config["data"]
    df = pd.read_csv(data_cfg["dataset_path"])
    row_indices = data_cfg.get("row_indices")
    if row_indices:
        df = df.iloc[row_indices]
    else:
        row_range = data_cfg.get("row_range", [0, len(df)])
        df = df.iloc[row_range[0]:row_range[1]]
    print(f"Data loaded: {len(df)} rows")

    # Load prompts
    extraction_prompts = config["_prompts"].get("extraction", {})
    plan_extraction_prompts = config["_prompts"].get("plan_extraction", {})
    letter_prompt_template = config["_prompts"].get("letter_generation", {}).get("patient_letter", "")

    # Generation configs
    keypoint_config = config["generation"]["keypoint"].copy()
    retry_config = config["generation"].get("retry", keypoint_config).copy()

    # Setup output directory
    exp_name = config["experiment"]["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"results/{exp_name}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, "results.txt")

    # Save config
    with open(os.path.join(run_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Starting run: {run_dir}")
    print(f"Model: {model_cfg['name']}")
    print(f"Samples: {len(df)}")
    print()

    total_start = time.time()

    for idx, (row_idx, row) in enumerate(df.iterrows()):
        row_num = row_idx + 1
        note_text = str(row.get("note_text", row.get("text", "")))
        coral_idx = row.get("coral_idx", row_idx)

        print(f"{'='*60}")
        print(f"ROW {row_num} (coral_idx {coral_idx}) [{idx+1}/{len(df)}]")
        row_start = time.time()

        # 1. Build base prompt (replaces build_base_cache)
        base_prompt = build_base_prompt(note_text, chat_tmpl=chat_tmpl)

        # 2. Extract A/P section (regex first)
        assessment_and_plan = extract_assessment_plan_regex(note_text)
        if not assessment_and_plan:
            # LLM fallback
            ap_task = chat_tmpl.user_assistant(
                "Extract the Assessment and Plan section from the note. Return ONLY the A/P text, nothing else."
            )
            assessment_and_plan, _ = vllm_generate(ap_task, client, keypoint_config, base_prompt)

        # 3. Phase 1: Extract keypoints from full note
        keypoints = {}
        phase1_keys = [
            "Reason_for_Visit", "Cancer_Diagnosis", "Lab_Results",
            "Clinical_Findings", "Current_Medications", "Treatment_Changes"
        ]
        for key in phase1_keys:
            prompt = extraction_prompts.get(key, "")
            if not prompt:
                continue
            t0 = time.time()
            result = extract_keypoint(prompt, client, keypoint_config, base_prompt, chat_tmpl)
            keypoints[key] = result
            print(f"  {key}: {time.time()-t0:.1f}s")

        # 4. Phase 2: Extract with context injection
        phase2_keys = ["Treatment_Goals", "Response_Assessment"]
        # Build context from Phase 1 results
        context_parts = []
        for k in ["Cancer_Diagnosis", "Current_Medications", "Clinical_Findings"]:
            if k in keypoints and keypoints[k]:
                context_parts.append(f"{k}: {json.dumps(keypoints[k])}")
        context_str = "\n".join(context_parts)

        for key in phase2_keys:
            prompt = extraction_prompts.get(key, "")
            if not prompt:
                continue
            if context_str:
                prompt = f"Context from earlier extraction:\n{context_str}\n\n{prompt}"
            t0 = time.time()
            result = extract_keypoint(prompt, client, keypoint_config, base_prompt, chat_tmpl)
            keypoints[key] = result
            print(f"  {key}: {time.time()-t0:.1f}s")

        # 5. Plan extraction from A/P
        if assessment_and_plan:
            ap_base = build_base_prompt(assessment_and_plan, chat_tmpl=chat_tmpl)
            for key, prompt in plan_extraction_prompts.items():
                if not prompt:
                    continue
                t0 = time.time()
                result = extract_keypoint(prompt, client, keypoint_config, ap_base, chat_tmpl)
                keypoints[key] = result
                print(f"  {key}: {time.time()-t0:.1f}s")

        # 6. Sanitize: convert list values to strings
        for section_key, section_val in keypoints.items():
            if isinstance(section_val, dict):
                for field_key, field_val in section_val.items():
                    if isinstance(field_val, list):
                        section_val[field_key] = "; ".join(str(v) for v in field_val)

        # 7. Letter generation
        letter = ""
        if letter_prompt_template and config.get("extraction", {}).get("letter", False):
            keypoints_json = json.dumps(keypoints, indent=2)
            letter_prompt_filled = letter_prompt_template.replace("{keypoints_json}", keypoints_json)
            letter_task = chat_tmpl.user_assistant(letter_prompt_filled)
            # Use full note as context for letter
            letter_base = build_base_prompt(note_text, chat_tmpl=chat_tmpl)
            letter_config = keypoint_config.copy()
            letter_config["max_new_tokens"] = 2048
            letter, _ = vllm_generate(letter_task, client, letter_config, letter_base)
            print(f"  Letter: generated ({len(letter)} chars)")

        # 8. Write results
        row_time = time.time() - row_start
        with open(results_path, 'a') as f:
            f.write(f"RESULTS FOR ROW {row_num}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"--- Column: coral_idx ---\n{coral_idx}\n\n")
            f.write(f"--- Column: note_text ---\n{json.dumps(note_text)}\n\n")
            f.write(f"--- Column: assessment_and_plan ---\n{json.dumps(assessment_and_plan)}\n\n")
            f.write(f"--- Column: keypoints ---\n{json.dumps(keypoints, indent=2)}\n\n")
            f.write(f"--- Column: letter ---\n{json.dumps(letter)}\n\n")
            f.write("\n\n\n\n\n")

        print(f"  Row {row_num} total: {row_time:.1f}s")
        print()

    total_time = time.time() - total_start
    print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"Results in: {run_dir}/")


if __name__ == "__main__":
    main()
