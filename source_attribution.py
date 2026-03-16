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
    '', 'none', 'n/a', '[]', '{}',
}

_SKIP_CONTAINS = (
    'the note does not',
    'not provided in the note',
    'not specified in',
    'not mentioned in',
    'not discussed during',
    'not documented',
    'cannot be determined',
    'unable to determine',
    'no data available',
)


def _is_skip_value(value_str):
    """Check if a value is non-informative for source attribution.

    Only skips truly empty values and LLM meta-responses.
    Short values like 'yes', 'palliative', 'New patient' are kept
    so the LLM can find inference sources.
    """
    val = value_str.strip()
    if len(val) == 0:
        return True
    val_lower = val.lower().rstrip('.').rstrip()
    if val_lower in _SKIP_EXACT:
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
    "sentences in a clinical note that support each extracted field value. "
    "You MUST respond with valid JSON only. No markdown, no explanations."
)

_ATTRIBUTION_PROMPT_TEMPLATE = """\
Below is a clinical note and {n_fields} extracted fields. For each field, find 1 short exact quote (5-25 words) from the note that supports it.

Rules:
- Quote the note EXACTLY. Do not paraphrase.
- For INFERRED values (e.g. "palliative", "yes"), quote the inference basis.
- You MUST provide a quote for EVERY field listed. Do not skip any.
- Return ONLY a JSON object. No markdown, no explanation.

Note:
{note_text}

Fields:
{fields_text}

JSON output (one key per field, value = exact quote):"""


def get_attributable_fields(keypoints):
    """Extract all fields worth attributing from keypoints.

    Returns dict of field_name -> value_str.
    """
    attributable = {}
    for prompt_name, fields in keypoints.items():
        if not isinstance(fields, dict):
            continue
        for field_name, field_value in fields.items():
            value_str = _flatten_value(field_value)
            if not _is_skip_value(value_str):
                attributable[field_name] = value_str
    return attributable


def build_batch_prompt(note_text, field_batch):
    """Build attribution prompt for a batch of fields.

    Args:
        note_text: Original clinical note
        field_batch: dict of field_name -> value_str (max ~10 fields)

    Returns: prompt string
    """
    lines = []
    for i, (fname, fval) in enumerate(field_batch.items(), 1):
        display_val = fval[:150] + '...' if len(fval) > 150 else fval
        lines.append(f"{i}. {fname}: \"{display_val}\"")
    fields_text = '\n'.join(lines)

    return _ATTRIBUTION_PROMPT_TEMPLATE.format(
        n_fields=len(field_batch),
        note_text=note_text,
        fields_text=fields_text,
    )


# ── JSON repair for truncated output ───────────────────────────────

def _repair_truncated_json(text):
    """Try to repair truncated JSON by closing open structures."""
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```\w*\s*', '', text)
        text = re.sub(r'\s*```\s*$', '', text)
    text = text.strip()

    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # progressively trim from the end until we can close the JSON
    for trim in range(len(text) - 1, max(0, len(text) - 500), -1):
        candidate = text[:trim].rstrip().rstrip(',')
        for suffix in ['"}', '"]', '}', '"}]', '"]}']:
            attempt = candidate + suffix
            try:
                result = json.loads(attempt)
                if isinstance(result, dict):
                    return result
            except (json.JSONDecodeError, ValueError):
                continue
    return None


# ── LLM call ───────────────────────────────────────────────────────

def run_attribution_llm(prompt, model, tokenizer, chat_tmpl, gen_config,
                        base_cache=None):
    """Run the attribution prompt through the LLM.

    Returns: (parsed_dict, raw_output)
    """
    from ult import try_parse_json, clone_cache, run_model

    if base_cache is not None:
        full_prompt = chat_tmpl.user_assistant(prompt)
        cache = clone_cache(base_cache)
        output, _ = run_model(full_prompt, model, tokenizer, gen_config, cache)
    else:
        full_prompt = chat_tmpl.system_user_assistant(_ATTRIBUTION_SYSTEM, prompt)
        output, _ = run_model(full_prompt, model, tokenizer, gen_config)

    result = try_parse_json(output)
    if result is None:
        result = _repair_truncated_json(output)

    return result, output


# ── main attribution function ─────────────────────────────────────

BATCH_SIZE = 8  # fields per LLM call


def attribute_row(note_text, keypoints, model, tokenizer, chat_tmpl,
                  gen_config, base_cache=None):
    """Run LLM source attribution for all fields in a row.

    Splits fields into batches of BATCH_SIZE to avoid output truncation.

    Returns:
        Dict of field_name -> list of source quotes
    """
    attributable = get_attributable_fields(keypoints)
    if not attributable:
        return {}

    # Split into batches
    field_items = list(attributable.items())
    batches = []
    for i in range(0, len(field_items), BATCH_SIZE):
        batches.append(dict(field_items[i:i + BATCH_SIZE]))

    attribution = {}
    for batch_idx, batch in enumerate(batches):
        prompt = build_batch_prompt(note_text, batch)
        result, raw_output = run_attribution_llm(
            prompt, model, tokenizer, chat_tmpl, gen_config, base_cache
        )

        if result is None:
            print(f"  [ATTR batch {batch_idx+1}/{len(batches)}] Parse failed", file=sys.stderr)
            print(f"  Raw ({len(raw_output)} chars): {raw_output[:500]}", file=sys.stderr)
            continue

        # Map results: handle both field name keys and numbered keys
        for key, quote in result.items():
            # try field name directly
            if key in batch:
                field_name = key
            else:
                # try numbered key -> field name mapping
                try:
                    idx = int(key) - 1
                    field_name = list(batch.keys())[idx]
                except (ValueError, IndexError):
                    continue

            if isinstance(quote, list):
                valid = [q for q in quote if isinstance(q, str) and len(q) > 5]
                if valid:
                    attribution[field_name] = valid
            elif isinstance(quote, str) and len(quote) > 5:
                attribution[field_name] = [quote]

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
                lines.append(f'    -> "...{q}..."')
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

    # Load model
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
    gen_config["max_new_tokens"] = 2048

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
        attributable = get_attributable_fields(keypoints)
        total = len(attributable)
        found = len(attribution)
        print(f'\n  --- {found}/{total} fields sourced ({elapsed:.1f}s, '
              f'{len(range(0, total, BATCH_SIZE))} batches) ---')

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_attributions, f, indent=2, ensure_ascii=False)
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
