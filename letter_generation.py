#!/usr/bin/env python3
"""
Patient letter generation with sentence-level traceability.

Generates a plain-language patient letter from extracted keypoints,
with [source:field_name] tags linking each sentence back to its source fields.
Builds a traceability JSON: sentence -> extraction value -> original note quote.
"""

import json
import re
import time

from ult import clone_cache, run_model_with_cache_manual


def flatten_keypoints(keypoints):
    """Flatten nested keypoints dict into {field_name: value_string}."""
    flat = {}
    for section_name, fields in keypoints.items():
        if isinstance(fields, dict):
            for field_name, field_value in fields.items():
                if isinstance(field_value, list):
                    flat[field_name] = ", ".join(str(v) for v in field_value)
                elif isinstance(field_value, dict):
                    flat[field_name] = ", ".join(
                        f"{k}: {v}" for k, v in field_value.items() if v
                    )
                else:
                    flat[field_name] = str(field_value) if field_value else ""
        else:
            flat[section_name] = str(fields) if fields else ""
    return flat


def generate_tagged_letter(keypoints, model, tokenizer, chat_tmpl,
                           gen_config, base_cache, letter_prompt_template):
    """Generate a tagged patient letter using the LLM.

    Args:
        keypoints: extracted keypoints dict (nested by section)
        model, tokenizer: HF model/tokenizer
        chat_tmpl: ChatTemplate instance
        gen_config: generation config dict (will override max_new_tokens)
        base_cache: KV cache with the note already encoded
        letter_prompt_template: prompt string with {keypoints_json} placeholder

    Returns:
        tagged_text: raw LLM output with [source:X] tags
    """
    flat = flatten_keypoints(keypoints)
    keypoints_json = json.dumps(flat, indent=2, ensure_ascii=False)
    prompt_text = letter_prompt_template.format(keypoints_json=keypoints_json)

    # Build the chat prompt and generate
    chat_prompt = chat_tmpl.user_assistant(prompt_text)
    cache = clone_cache(base_cache)

    letter_config = gen_config.copy()
    letter_config["max_new_tokens"] = letter_config.get("max_new_tokens", 512)

    output, _ = run_model_with_cache_manual(
        chat_prompt, model, tokenizer, letter_config, cache
    )
    return output.strip()


def parse_tagged_letter(tagged_text, keypoints, attribution):
    """Parse a tagged letter into clean text + traceability JSON.

    Args:
        tagged_text: LLM output with [source:field_name] after each sentence
        keypoints: nested keypoints dict
        attribution: {field_name: quote_string} from source_attribution

    Returns:
        dict with:
            - letter_text: clean letter (tags stripped)
            - sentences: list of sentence dicts with traceability info
    """
    flat_kp = flatten_keypoints(keypoints)

    # Split on [source:...] tags, capturing the tag content
    # Pattern: text followed by [source:field1,field2]
    tag_pattern = re.compile(r'\[source:([^\]]+)\]')

    # Find all sentences with their tags
    sentences = []
    # Split the text by the tag pattern
    parts = tag_pattern.split(tagged_text)

    # parts alternates: text, tag_content, text, tag_content, ...
    # Odd indices are tag contents, even indices are text segments
    current_text = ""
    idx = 0

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Text segment - accumulate
            current_text += part
        else:
            # Tag content - this ends the current sentence(s)
            source_fields = [f.strip() for f in part.split(",")]
            text = current_text.strip()
            if text:
                # Build extraction values and note quotes for these fields
                extraction_values = {}
                note_quotes = {}
                for field in source_fields:
                    if field == "none":
                        continue
                    if field in flat_kp:
                        extraction_values[field] = flat_kp[field]
                    if field in attribution:
                        note_quotes[field] = attribution[field]

                sentences.append({
                    "index": idx,
                    "text": text,
                    "source_fields": source_fields,
                    "extraction_values": extraction_values,
                    "note_quotes": note_quotes,
                })
                idx += 1
            current_text = ""

    # Handle any trailing text without a tag
    trailing = current_text.strip()
    if trailing:
        sentences.append({
            "index": idx,
            "text": trailing,
            "source_fields": ["unattributed"],
            "extraction_values": {},
            "note_quotes": {},
        })

    # Build clean letter text (all tags stripped)
    letter_text = tag_pattern.sub("", tagged_text).strip()
    # Collapse multiple spaces/newlines
    letter_text = re.sub(r' +', ' ', letter_text)
    letter_text = re.sub(r'\n{3,}', '\n\n', letter_text)

    return {
        "letter_text": letter_text,
        "sentences": sentences,
    }
