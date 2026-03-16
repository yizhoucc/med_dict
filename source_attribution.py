#!/usr/bin/env python3
"""
Source Attribution (LLM second-pass with KV Cache):
Load the note once into KV cache, then ask per-field questions.

Standalone usage (requires GPU + model):
    python source_attribution.py exp/default_qwen.yaml results/.../progress.json --row 49

Integration: called from run.py after extraction, reusing loaded model + base_cache.
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
    'no labs planned',
    'no procedures planned',
    'no imaging planned',
    'none planned',
    'no new clinical findings',
    'no labs in note',
)


def _is_skip_value(value_str):
    """Check if a value is non-informative for source attribution."""
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


def get_attributable_fields(keypoints):
    """Extract all fields worth attributing from keypoints."""
    attributable = {}
    for prompt_name, fields in keypoints.items():
        if not isinstance(fields, dict):
            continue
        for field_name, field_value in fields.items():
            value_str = _flatten_value(field_value)
            if not _is_skip_value(value_str):
                attributable[field_name] = value_str
    return attributable


# ── per-field attribution via KV cache ─────────────────────────────

_FIELD_QUESTION_TEMPLATE = (
    'I extracted {field_name}: "{value}". '
    'Quote the EXACT short phrase (5-20 words) from the note that is the '
    'most direct evidence. Rules: '
    '1) Prefer the Assessment/Plan section if the answer appears there. '
    '2) If it was INFERRED, quote the logical premise '
    '(e.g. "palliative" comes from "Stage IV" or "metastatic", not unrelated text). '
    '3) If the value says "No X planned" or "None", quote any mention of X, '
    'or reply NOT_IN_NOTE if X is simply absent from the note. '
    'Reply with ONLY the quote, nothing else.'
)


def attribute_single_field(field_name, value_str, model, tokenizer,
                           chat_tmpl, gen_config, base_cache):
    """Ask the LLM where a single field value came from.

    Uses KV cache (note already encoded). Returns the quote string or None.
    """
    from ult import clone_cache, run_model_with_cache_manual

    question = _FIELD_QUESTION_TEMPLATE.format(
        field_name=field_name,
        value=value_str[:150],
    )
    prompt = chat_tmpl.user_assistant(question)
    cache = clone_cache(base_cache)

    output, _ = run_model_with_cache_manual(
        prompt, model, tokenizer, gen_config, cache
    )

    # Clean up the output
    quote = output.strip().strip('"').strip("'").strip()
    # Remove common prefixes the model might add
    for prefix in ['The quote is:', 'Quote:', 'Source:', 'From the note:',
                   'The relevant text is:', 'The exact phrase is:']:
        if quote.lower().startswith(prefix.lower()):
            quote = quote[len(prefix):].strip().strip('"').strip("'").strip()

    # Handle "not in note" responses
    not_in_note_phrases = ['not_in_note', 'not in note', 'not mentioned',
                           'not found', 'n/a', 'none']
    if quote.lower().replace(' ', '_') in not_in_note_phrases or len(quote) < 5:
        return None
    return quote


def attribute_row(note_text, keypoints, model, tokenizer, chat_tmpl,
                  gen_config, base_cache):
    """Run LLM source attribution for all fields in a row using KV cache.

    The base_cache should already contain the encoded note text.

    Returns:
        Dict of field_name -> list of source quotes
    """
    attributable = get_attributable_fields(keypoints)
    if not attributable:
        return {}

    attribution = {}
    for field_name, value_str in attributable.items():
        quote = attribute_single_field(
            field_name, value_str, model, tokenizer,
            chat_tmpl, gen_config, base_cache
        )
        if quote:
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


# ── CLI (standalone testing) ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LLM-based source attribution (KV cache per-field)'
    )
    parser.add_argument('config', help='Path to experiment config YAML')
    parser.add_argument('progress_file', help='Path to progress.json')
    parser.add_argument('--row', type=int, help='Specific row index')
    parser.add_argument('--output', '-o', help='Output JSON file')
    args = parser.parse_args()

    import os
    import torch
    import yaml
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from ult import ChatTemplate, build_base_cache

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load model
    model_cfg = config["model"]
    print("Loading model...")

    hf_token_path = "hf.token"
    if os.path.exists(hf_token_path):
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
    gen_config["max_new_tokens"] = 128  # each answer is just a short quote

    # Load progress
    with open(args.progress_file) as f:
        progress = json.load(f)

    results = progress.get('results', {})

    # Load existing results for resume support
    all_attributions = {}
    if args.output and os.path.exists(args.output):
        try:
            with open(args.output) as f:
                all_attributions = json.load(f)
            print(f'Resuming: loaded {len(all_attributions)} existing rows from {args.output}')
        except (json.JSONDecodeError, IOError):
            pass

    row_keys = [str(args.row)] if args.row is not None else sorted(results.keys(), key=int)
    global_t0 = time.time()
    rows_done_this_run = 0

    for i, row_key in enumerate(row_keys):
        # Skip already completed rows
        if row_key in all_attributions:
            print(f'[{i+1}/{len(row_keys)}] Row {row_key} — already done, skipping')
            continue

        if row_key not in results:
            print(f'Row {row_key} not found', file=sys.stderr)
            continue

        row = results[row_key]
        note_text = row.get('note_text', '')
        keypoints = row.get('keypoints', {})
        coral_idx = row.get('coral_idx', '?')

        print(f'\n{"="*60}')
        print(f'[{i+1}/{len(row_keys)}] ROW {row_key} (coral_idx={coral_idx})')
        print(f'{"="*60}')

        # Build KV cache for this note
        t0 = time.time()
        base_cache = build_base_cache(note_text, model, tokenizer,
                                       chat_tmpl=chat_tmpl)
        cache_time = time.time() - t0

        # Run attribution
        t1 = time.time()
        attribution = attribute_row(
            note_text, keypoints, model, tokenizer,
            chat_tmpl, gen_config, base_cache
        )
        attr_time = time.time() - t1
        total_time = time.time() - t0

        all_attributions[row_key] = attribution
        rows_done_this_run += 1

        output = format_attribution(attribution, keypoints)
        print(output)

        # Stats
        attributable = get_attributable_fields(keypoints)
        total_fields = len(attributable)
        found = len(attribution)
        remaining = len(row_keys) - i - 1
        elapsed = time.time() - global_t0
        avg_per_row = elapsed / rows_done_this_run
        eta = avg_per_row * remaining
        print(f'\n  --- {found}/{total_fields} fields sourced ---')
        print(f'  --- cache: {cache_time:.1f}s + {total_fields} queries: {attr_time:.1f}s = {total_time:.1f}s ---')
        print(f'  --- progress: {len(all_attributions)}/{len(row_keys)} rows done, ETA: {eta/60:.0f}min ---')

        # Incremental save after each row
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(all_attributions, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - global_t0
    print(f'\n{"="*60}')
    print(f'DONE: {len(all_attributions)} rows, {total_elapsed/60:.1f} min total')
    if args.output:
        print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
