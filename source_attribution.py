#!/usr/bin/env python3
"""
Source Attribution (LLM second-pass): ask the LLM to find source sentences
in the original note for each extracted field value.

Standalone usage (requires GPU + model):
    python source_attribution.py exp/default_qwen.yaml results/.../progress.json --row 49

Integration: called from run.py after extraction, reusing loaded model + KV cache.
"""

import argparse
import json
import re
import sys
import time


# ── skip detection ─────────────────────────────────────────────────

_SKIP_EXACT = {
    '', 'none', 'no', 'yes', 'n/a', 'not mentioned', 'not specified',
    'not applicable', 'unknown', '[]', '{}',
}

_SKIP_PREFIXES = (
    'no ',
    'not ',
    'none ',
)

_SKIP_CONTAINS = (
    'no response to assess',
    'the note does not',
    'not provided in the note',
    'not specified in',
    'not mentioned in',
    'not discussed during',
    'not documented',
    'no evidence of response',
    'cannot be determined',
    'unable to determine',
    'no data available',
)


def _is_skip_value(value_str):
    """Check if a value is non-informative for source attribution."""
    val = value_str.strip()
    if len(val) < 3:
        return True
    val_lower = val.lower().rstrip('.').rstrip()
    if val_lower in _SKIP_EXACT:
        return True
    if len(val_lower) < 50 and val_lower.startswith(_SKIP_PREFIXES):
        return True
    for phrase in _SKIP_CONTAINS:
        if phrase in val_lower:
            return True
    return False


def _flatten_value(field_value):
    """Flatten a field value (list/dict/str) to a string."""
    if isinstance(field_value, list):
        parts = []
        for item in field_value:
            if isinstance(item, dict):
                parts.extend(str(v) for v in item.values() if v)
            else:
                parts.append(str(item))
        return ' '.join(parts)
    elif isinstance(field_value, dict):
        return ' '.join(str(v) for v in field_value.values() if v)
    return str(field_value)


# ── prompt construction ────────────────────────────────────────────

_ATTRIBUTION_SYSTEM = (
    "You are a source attribution expert. Your job is to find the exact "
    "sentences in a clinical note that support each extracted field value."
)

_ATTRIBUTION_PROMPT_TEMPLATE = """\
Below is a clinical note and a list of extracted fields with their values.
For each field, find 1-3 exact quotes from the note that support the extracted value.

## Rules
- Quote the note text EXACTLY as it appears — do not paraphrase or modify
- Each quote should be a meaningful phrase or sentence (15-150 words)
- If the value was INFERRED (not directly stated), quote the text the inference is based on
- Only include fields where you can find supporting text
- Skip fields you cannot find support for

## Clinical Note
--- BEGIN NOTE ---
{note_text}
--- END NOTE ---

## Extracted Fields
{fields_text}

## Output
Return a JSON object. Keys = field names exactly as listed above, values = list of exact quote strings from the note.
```json
"""


def build_attribution_prompt(note_text, keypoints):
    """Build the attribution prompt for a single row.

    Args:
        note_text: Original clinical note
        keypoints: Dict of prompt_name -> {field: value}

    Returns:
        (prompt_text, attributable_fields) where attributable_fields is
        a dict of field_name -> value_str for fields worth attributing.
    """
    attributable = {}
    for prompt_name, fields in keypoints.items():
        if not isinstance(fields, dict):
            continue
        for field_name, field_value in fields.items():
            value_str = _flatten_value(field_value)
            if not _is_skip_value(value_str):
                attributable[field_name] = value_str

    if not attributable:
        return None, {}

    # Build fields text
    lines = []
    for i, (fname, fval) in enumerate(attributable.items(), 1):
        # Truncate very long values
        display_val = fval[:200] + '...' if len(fval) > 200 else fval
        lines.append(f"{i}. {fname}: \"{display_val}\"")
    fields_text = '\n'.join(lines)

    prompt = _ATTRIBUTION_PROMPT_TEMPLATE.format(
        note_text=note_text,
        fields_text=fields_text,
    )
    return prompt, attributable


# ── LLM call ───────────────────────────────────────────────────────

def run_attribution_llm(prompt, model, tokenizer, chat_tmpl, gen_config,
                        base_cache=None):
    """Run the attribution prompt through the LLM.

    Can work two ways:
    1. With base_cache: append as user turn (efficient, reuses note encoding)
    2. Without base_cache: full prompt (standalone mode)

    Returns: parsed dict or None
    """
    from ult import try_parse_json, clone_cache, run_model

    if base_cache is not None:
        # Append to existing cache (note already encoded)
        full_prompt = chat_tmpl.user_assistant(prompt)
        cache = clone_cache(base_cache)
        output, _ = run_model(full_prompt, model, tokenizer, gen_config, cache)
    else:
        # Full prompt (standalone)
        full_prompt = chat_tmpl.system_user_assistant(_ATTRIBUTION_SYSTEM, prompt)
        output, _ = run_model(full_prompt, model, tokenizer, gen_config)

    # Parse the JSON output
    result = try_parse_json(output)
    if result is None:
        # Try extracting JSON from markdown code block
        match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if match:
            result = try_parse_json(match.group(1))
    return result, output


# ── main attribution function ─────────────────────────────────────

def attribute_row(note_text, keypoints, model, tokenizer, chat_tmpl,
                  gen_config, base_cache=None):
    """Run LLM source attribution for all fields in a row.

    Args:
        note_text: Original clinical note
        keypoints: Dict of prompt_name -> {field: value}
        model, tokenizer, chat_tmpl: Model infrastructure
        gen_config: Generation config dict
        base_cache: Optional KV cache with note already encoded

    Returns:
        Dict of field_name -> list of source quotes
    """
    prompt, attributable = build_attribution_prompt(note_text, keypoints)
    if prompt is None:
        return {}

    result, raw_output = run_attribution_llm(
        prompt, model, tokenizer, chat_tmpl, gen_config, base_cache
    )

    if result is None:
        print(f"  [ATTRIBUTION] Failed to parse LLM output", file=sys.stderr)
        return {}

    # Validate: only keep fields that were in the request
    attribution = {}
    for field_name, quotes in result.items():
        if field_name in attributable:
            if isinstance(quotes, list):
                # Filter out empty/very short quotes
                valid = [q for q in quotes if isinstance(q, str) and len(q) > 10]
                if valid:
                    attribution[field_name] = valid
            elif isinstance(quotes, str) and len(quotes) > 10:
                attribution[field_name] = [quotes]

    return attribution


# ── display ────────────────────────────────────────────────────────

def format_attribution(attribution, keypoints, max_quote=150):
    """Format attribution results for display."""
    lines = []
    for prompt_name, fields in keypoints.items():
        if not isinstance(fields, dict):
            continue
        for field_name, field_value in fields.items():
            if field_name not in attribution:
                continue
            val_str = str(field_value)[:100]
            lines.append(f'\n  {field_name}: "{val_str}"')
            for quote in attribution[field_name]:
                q = quote[:max_quote]
                lines.append(f'    → "...{q}..."')
    return '\n'.join(lines)


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LLM-based source attribution for extracted fields'
    )
    parser.add_argument('config', help='Path to experiment config YAML')
    parser.add_argument('progress_file', help='Path to progress.json')
    parser.add_argument('--row', type=int, help='Specific row index')
    parser.add_argument('--output', '-o', help='Output JSON file')
    args = parser.parse_args()

    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from ult import ChatTemplate

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model (same as run.py)
    model_cfg = config["model"]
    print("Loading model...")

    hf_token_path = "hf.token"
    if __import__('os').path.exists(hf_token_path):
        with open(hf_token_path) as f:
            hftoken = f.read().strip()
        from huggingface_hub import login
        login(token=hftoken)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(model_cfg.get("dtype", "bfloat16"), torch.bfloat16)

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

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    load_kwargs = {
        "device_map": model_cfg.get("device_map", "auto"),
        "torch_dtype": torch_dtype,
    }
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **load_kwargs)
    print("Model loaded.")

    chat_tmpl = ChatTemplate(model_cfg.get("chat_template", "llama3"))

    gen_config = config["generation"]["keypoint"].copy()
    gen_config["eos_token_id"] = tokenizer.eos_token_id
    gen_config["max_new_tokens"] = 1024  # attribution needs more space

    # Load progress
    with open(args.progress_file) as f:
        progress = json.load(f)

    results = progress.get('results', {})
    all_attributions = {}

    row_keys = [str(args.row)] if args.row is not None else sorted(results.keys(), key=int)

    for row_key in row_keys:
        if row_key not in results:
            print(f'Row {row_key} not found', file=sys.stderr)
            continue

        row = results[row_key]
        note_text = row.get('note_text', '')
        keypoints = row.get('keypoints', {})
        coral_idx = row.get('coral_idx', '?')

        print(f'\n{"="*60}')
        print(f'ROW {row_key} (coral_idx={coral_idx})')
        print(f'{"="*60}')

        t0 = time.time()
        attribution = attribute_row(
            note_text, keypoints, model, tokenizer,
            chat_tmpl, gen_config
        )
        elapsed = time.time() - t0
        all_attributions[row_key] = attribution

        output = format_attribution(attribution, keypoints)
        print(output)

        # Stats
        _, attributable = build_attribution_prompt(note_text, keypoints)
        total = len(attributable)
        found = len(attribution)
        print(f'\n  --- {found}/{total} attributable fields sourced ({elapsed:.1f}s) ---')

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_attributions, f, indent=2, ensure_ascii=False)
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
