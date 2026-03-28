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


def _clean_keypoints_for_letter(flat):
    """Clean flattened keypoints before sending to LLM for letter generation.

    1. Dedup: if recent_changes ≈ therapy_plan or medication_plan, clear the duplicate.
    2. TNM → plain stage: translate pT2N1M0 etc. to Stage I/II/III/IV.
    3. Strip [REDACTED] from values to prevent leaking into letter.
    """
    # 1. Dedup overlapping fields (substring OR word-overlap > 60%)
    def _similar(a, b):
        if not a or not b:
            return False
        if a in b or b in a:
            return True
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if len(wa) < 3 or len(wb) < 3:
            return False
        return len(wa & wb) / min(len(wa), len(wb)) > 0.6

    rc = flat.get("recent_changes", "").strip()
    tp = flat.get("therapy_plan", "").strip()
    mp = flat.get("medication_plan", "").strip()
    if _similar(rc, tp):
        flat["therapy_plan"] = ""
    if _similar(mp, tp):
        flat["therapy_plan"] = ""
    # Dedup response_assessment ≈ findings (keep response-specific words)
    ra = flat.get("response_assessment", "").strip()
    fi = flat.get("findings", "").strip()
    if ra and fi and _similar(ra, fi):
        _resp_words = {'decrease', 'decreased', 'increase', 'increased', 'improve', 'improved',
                       'stable', 'respond', 'responding', 'progression', 'better', 'worse',
                       'growing', 'shrink', 'smaller', 'larger', 'resolved', 'worsening'}
        ra_sents = [s.strip() for s in re.split(r'[.;]', ra) if s.strip()]
        unique = [s for s in ra_sents if any(w in s.lower().split() for w in _resp_words)]
        flat["response_assessment"] = ". ".join(unique) + "." if unique else ""

    # 2. Receptor pre-translation — append plain-language explanation
    toc = flat.get("Type_of_Cancer", "")
    if toc:
        explanations = []
        if 'triple negative' in toc.lower() or 'TNBC' in toc.upper():
            explanations = ["the cancer cells lack three common receptors (ER, PR, HER2)"]
        else:
            if re.search(r'ER\s*\+|ER\s*positive|HR\s*\+', toc, re.IGNORECASE):
                explanations.append("grows in response to hormones (estrogen)")
            elif re.search(r'ER\s*-|ER\s*negative', toc, re.IGNORECASE):
                explanations.append("does not respond to hormones")
            if re.search(r'HER2\s*\+|HER2\s*positive', toc, re.IGNORECASE):
                explanations.append("has extra HER2 protein")
            elif re.search(r'HER2\s*-|HER2\s*negative|HER2:\s*not tested', toc, re.IGNORECASE):
                explanations.append("does not have extra HER2 protein")
        if explanations:
            flat["Type_of_Cancer"] = toc + " — in plain language: " + "; ".join(explanations)

    # 3. TNM → plain stage
    stage = flat.get("Stage_of_Cancer", "")
    if re.search(r'pT\d|pN[0-9X]|^T\d.*N\d.*M\d', stage):
        stage_lower = stage.lower()
        if "m1" in stage_lower or "metastatic" in stage_lower:
            flat["Stage_of_Cancer"] = "Stage IV (metastatic)"
        elif re.search(r'N[23]', stage):
            flat["Stage_of_Cancer"] = "Stage III"
        elif re.search(r'T[34]', stage):
            flat["Stage_of_Cancer"] = "Stage II-III"
        else:
            flat["Stage_of_Cancer"] = "Early stage (Stage I-II)"

    # 3. Replace [REDACTED] with generic text in all values
    for k, v in flat.items():
        if isinstance(v, str) and "[REDACTED]" in v:
            # Doctor names first: "Dr. [REDACTED]" → "your doctor"
            v = re.sub(r'Dr\.?\s*\[REDACTED\](\s*\[REDACTED\])*', 'your doctor', v)
            # Then general: remaining [REDACTED] → "a specific treatment"
            v = re.sub(r'\[REDACTED\](\s*\[REDACTED\])*', 'a specific treatment', v)
            flat[k] = v

    return flat


_EMOTION_KEYWORDS = [
    "distressed", "anxious", "anxiety", "scared", "fearful", "crying", "tearful",
    "depressed", "depression", "worried", "overwhelmed", "upset", "emotional",
    "frightened", "nervous", "stressed",
]


def generate_tagged_letter(keypoints, model, tokenizer, chat_tmpl,
                           gen_config, base_cache, letter_prompt_template,
                           note_text=""):
    """Generate a tagged patient letter using the LLM.

    Args:
        keypoints: extracted keypoints dict (nested by section)
        model, tokenizer: HF model/tokenizer
        chat_tmpl: ChatTemplate instance
        gen_config: generation config dict (will override max_new_tokens)
        base_cache: KV cache with the note already encoded
        letter_prompt_template: prompt string with {keypoints_json} placeholder
        note_text: original note text for emotion detection

    Returns:
        tagged_text: raw LLM output with [source:X] tags
    """
    flat = flatten_keypoints(keypoints)
    flat = _clean_keypoints_for_letter(flat)

    # Detect patient emotions from note text
    if note_text:
        note_lower = note_text.lower()
        emotions = [kw for kw in _EMOTION_KEYWORDS if kw in note_lower]
        if emotions:
            flat["emotional_context"] = f"Patient appears {', '.join(emotions[:3])}."

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


def _unwrap_json_shell(text):
    """Strip JSON wrapper if LLM wrapped the letter in {"letter": "..."}."""
    stripped = text.strip()
    if not (stripped.startswith('{') and stripped.endswith('}')):
        return text
    # Try standard JSON parse
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, str) and len(v) > 50:
                    return v
    except json.JSONDecodeError:
        pass
    # Regex fallback: LLM used real newlines inside JSON string value
    m = re.search(r'^\{\s*"[^"]+"\s*:\s*"', stripped)
    if m:
        content = stripped[m.end():]
        content = re.sub(r'"\s*\}\s*$', '', content)
        if len(content) > 50:
            content = content.replace('\\n', '\n').replace('\\"', '"')
            return content
    return text


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
    # LLM sometimes wraps output in {"letter": "..."} — unwrap it
    tagged_text = _unwrap_json_shell(tagged_text)

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


def post_check_letter(letter_text):
    """Post-generation checks on letter text. Returns (cleaned_text, warnings)."""
    warnings = []

    # 1. Strip [REDACTED] leaks and fix "Dr. a specific treatment"
    if "[REDACTED]" in letter_text:
        letter_text = re.sub(r'Dr\.?\s*\[REDACTED\](\s*\[REDACTED\])*', 'your doctor', letter_text)
        letter_text = re.sub(r'\[REDACTED\](\s*\[REDACTED\])*', 'a specific treatment', letter_text)
        warnings.append("[POST-LETTER] stripped [REDACTED] from letter")
    if "Dr. a specific treatment" in letter_text:
        letter_text = letter_text.replace("Dr. a specific treatment", "your doctor")
        warnings.append("[POST-LETTER] fixed 'Dr. a specific treatment' → 'your doctor'")

    # 2. Detect TNM staging patterns
    tnm_match = re.search(r'pT\d|pN[0-9X]|stage\s+pT', letter_text, re.IGNORECASE)
    if tnm_match:
        warnings.append(f"[POST-LETTER] WARNING: TNM staging in letter: '{tnm_match.group()}'")

    # 3. Fix receptor contradiction: if letter says both "ER positive" and
    #    "does not respond to hormones" (without receptor-change context), fix it
    has_er_pos = bool(re.search(r'ER\s*positive|ER\+', letter_text, re.IGNORECASE))
    has_wrong = 'does not respond to hormones' in letter_text
    if has_er_pos and has_wrong:
        # Only fix if NOT in receptor-change context ("used to"/"now"/"but now")
        idx = letter_text.find('does not respond to hormones')
        context = letter_text[max(0, idx - 120):idx]
        if 'used to' not in context.lower() and 'now' not in context.lower():
            letter_text = letter_text.replace(
                'does not respond to hormones',
                'responds to hormones (estrogen)',
            )
            warnings.append("[POST-LETTER] fixed receptor contradiction: ER+ but said 'does not respond'")

    # 4. Remove semantically repeated sentences (word overlap > 70%)
    lines = [ln.strip() for ln in letter_text.split('\n') if ln.strip()]
    kept = []
    for ln in lines:
        words = set(ln.lower().split())
        is_dup = False
        if len(words) > 6:
            for prev in kept:
                prev_words = set(prev.lower().split())
                if len(prev_words) > 6:
                    overlap = len(words & prev_words) / min(len(words), len(prev_words))
                    if overlap > 0.7:
                        is_dup = True
                        warnings.append(f"[POST-LETTER] removed duplicate: '{ln[:50]}...'")
                        break
        if not is_dup:
            kept.append(ln)
    letter_text = '\n'.join(kept)

    return letter_text, warnings
