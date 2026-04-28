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


def _count_syllables(word):
    """Estimate syllable count for English word."""
    word = word.lower().strip(".,!?;:'\"")
    if not word:
        return 0
    count = 0
    vowels = "aeiouy"
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text):
    """Compute Flesch-Kincaid Grade Level for text."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 3]
    words = [w for w in re.findall(r"[a-zA-Z']+", text) if len(w) > 0]
    if not sentences or not words:
        return 0.0
    total_syllables = sum(_count_syllables(w) for w in words)
    avg_words_per_sent = len(words) / len(sentences)
    avg_syllables_per_word = total_syllables / len(words)
    grade = 0.39 * avg_words_per_sent + 11.8 * avg_syllables_per_word - 15.59
    return round(max(grade, 0.0), 1)


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

    # 2. Medical term pre-translation in all text fields
    _TERM_MAP = {
        'adenocarcinoma': 'adenocarcinoma (cancer that started in gland cells)',
        'mucinous carcinoma': 'mucinous carcinoma (a type of cancer that makes mucus)',
        'mucinous differentiation': 'mucinous differentiation (a type that makes mucus)',
        'lobular carcinoma': 'lobular carcinoma (cancer that started in the milk-producing glands)',
        'salpingo-oophorectomy': 'surgery to remove the ovaries and fallopian tubes',
        'bilateral salpingo-oophorectomy': 'surgery to remove both ovaries and fallopian tubes',
        'BSO': 'surgery to remove both ovaries and fallopian tubes (BSO)',
    }
    for k, v in flat.items():
        if isinstance(v, str) and v:
            for term, replacement in _TERM_MAP.items():
                if term.lower() in v.lower() and replacement.lower() not in v.lower():
                    v = re.sub(re.escape(term), replacement, v, flags=re.IGNORECASE)
                    flat[k] = v

    # 2c. Drug dedup: remove drugs from current_meds if already in recent_changes/medication_plan
    cm = flat.get("current_meds", "").strip()
    rc = flat.get("recent_changes", "").strip()
    mp = flat.get("medication_plan", "").strip()
    if cm and (rc or mp):
        plan_text = (rc + " " + mp).lower()
        drugs = [d.strip() for d in cm.split(',') if d.strip()]
        kept = []
        for d in drugs:
            # Check if any word of this drug name appears in the plan text
            d_words = d.strip().lower().split()
            if not any(w in plan_text for w in d_words if len(w) > 3):
                kept.append(d)
        if len(kept) < len(drugs):
            flat["current_meds"] = ", ".join(kept) if kept else ""

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

    # 3b. Simplify metastasis sites: if 3+ sites listed, use general term
    for met_field in ("Metastasis", "Distant Metastasis"):
        met = flat.get(met_field, "")
        if met and met.lower().startswith("yes"):
            # Count distinct site words (split by comma, "and", slash)
            sites_part = re.sub(r'^yes\s*[\(\:]?\s*(?:to\s+)?', '', met, flags=re.IGNORECASE)
            sites = [s.strip() for s in re.split(r'[,/]|\band\b', sites_part) if s.strip()]
            if len(sites) >= 3:
                flat[met_field] = "Yes (to multiple sites throughout the body)"

    # 4. Replace [REDACTED] with generic text — or skip field if mostly redacted
    for k, v in flat.items():
        if isinstance(v, str) and "[REDACTED]" in v:
            # Doctor names first: "Dr. [REDACTED]" → "your doctor"
            v = re.sub(r'Dr\.?\s*\[REDACTED\](\s*\[REDACTED\])*', 'your doctor', v)
            # Check if remaining value is mostly [REDACTED] (>50% of words)
            cleaned_test = re.sub(r'\[REDACTED\]', '', v)
            orig_words = len(v.split())
            clean_words = len(cleaned_test.split())
            if orig_words > 0 and clean_words / orig_words < 0.4:
                # Mostly redacted — clear the field so LLM skips it
                flat[k] = ""
            else:
                # Replace remaining [REDACTED] with "a medication"
                v = re.sub(r'\[REDACTED\](\s*\[REDACTED\])*', 'a medication', v)
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

    # Detect patient emotions from note text (with negation filtering)
    if note_text:
        note_lower = note_text.lower()
        _NEG_WORDS = {'no ', 'not ', 'denies ', 'denied ', 'negative ', 'without ', 'absent ', 'none ',
                      'h/o ', 'history of ', 'past medical', 'pmh ', 'hx of ', 'hx '}
        emotions = []
        for kw in _EMOTION_KEYWORDS:
            idx = note_lower.find(kw)
            if idx == -1:
                continue
            # Check preceding 40 chars for negation
            context = note_lower[max(0, idx - 40):idx]
            if any(neg in context for neg in _NEG_WORDS):
                continue
            emotions.append(kw)
        if emotions:
            flat["emotional_context"] = f"Patient appears {', '.join(emotions[:3])}."

    keypoints_json = json.dumps(flat, indent=2, ensure_ascii=False)
    prompt_text = letter_prompt_template.format(keypoints_json=keypoints_json)

    # Build the chat prompt and generate
    chat_prompt = chat_tmpl.user_assistant(prompt_text)
    cache = clone_cache(base_cache)

    letter_config = gen_config.copy()
    letter_config["max_new_tokens"] = letter_config.get("max_new_tokens", 1024)

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
                    # Normalize: try original, then underscore↔space variants
                    field_variants = [field, field.replace('_', ' '), field.replace(' ', '_')]
                    for fv in field_variants:
                        if fv in flat_kp:
                            extraction_values[field] = flat_kp[fv]
                            break
                    for fv in field_variants:
                        if fv in attribution:
                            note_quotes[field] = attribution[fv]
                            break

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

    # Compute metrics
    # 1. Readability
    letter_grade = flesch_kincaid_grade(letter_text)

    # 2. Field coverage: which keypoint fields are referenced in letter
    #    Exclude visit-logistics fields (not meant for patient letter)
    #    and fields cleared by dedup
    _SKIP_COVERAGE = {
        'Patient type', 'second opinion', 'in-person',  # visit metadata
        'Next clinic visit',  # internal scheduling
        'therapy_plan',  # often deduped with medication_plan/recent_changes
    }
    _SKIP_VALUES = {
        'none', 'n/a', '', 'no labs in note.', 'no procedures planned.',
        'no imaging planned.', 'no labs planned.', 'none planned.',
        'not discussed during this visit.', 'not mentioned in note.',
        'not yet on treatment — no response to assess.',
    }
    all_sources = set()
    for s in sentences:
        for f in s["source_fields"]:
            if f not in ("none", "unattributed"):
                all_sources.add(f)
    non_empty_fields = {k for k, v in flat_kp.items()
                        if k not in _SKIP_COVERAGE
                        and isinstance(v, str) and v.strip()
                        and v.strip().lower() not in _SKIP_VALUES}
    covered = all_sources & non_empty_fields
    coverage_pct = round(len(covered) / len(non_empty_fields) * 100) if non_empty_fields else 100

    # 3. Attribution completeness: sentences with both extraction_values AND note_quotes
    n_content = sum(1 for s in sentences if s["source_fields"] not in [["none"], ["unattributed"]])
    n_full_chain = sum(1 for s in sentences
                       if s["source_fields"] not in [["none"], ["unattributed"]]
                       and s.get("extraction_values") and s.get("note_quotes"))
    attr_pct = round(n_full_chain / n_content * 100) if n_content else 100

    return {
        "letter_text": letter_text,
        "sentences": sentences,
        "metrics": {
            "readability_grade": letter_grade,
            "field_coverage_pct": coverage_pct,
            "fields_covered": sorted(covered),
            "fields_missed": sorted(non_empty_fields - all_sources),
            "attribution_complete_pct": attr_pct,
            "sentences_total": len(sentences),
            "sentences_attributed": n_content,
            "sentences_full_chain": n_full_chain,
        },
    }


def post_check_letter(letter_text):
    """Post-generation checks on letter text. Returns (cleaned_text, warnings)."""
    warnings = []

    # 1. Strip [REDACTED] leaks and fix "Dr. a medication"
    if "[REDACTED]" in letter_text:
        letter_text = re.sub(r'Dr\.?\s*\[REDACTED\](\s*\[REDACTED\])*', 'your doctor', letter_text)
        letter_text = re.sub(r'\[REDACTED\](\s*\[REDACTED\])*', 'a medication', letter_text)
        warnings.append("[POST-LETTER] stripped [REDACTED] from letter")
    if "Dr. a medication" in letter_text:
        letter_text = letter_text.replace("Dr. a medication", "your doctor")
        warnings.append("[POST-LETTER] fixed 'Dr. a medication' → 'your doctor'")
    # 2. Detect TNM staging patterns
    tnm_match = re.search(r'pT\d|pN[0-9X]|stage\s+pT', letter_text, re.IGNORECASE)
    if tnm_match:
        warnings.append(f"[POST-LETTER] WARNING: TNM staging in letter: '{tnm_match.group()}'")

    # 2b. Replace chemo regimen abbreviations with plain language
    # Order matters: longer names first to avoid partial matches (e.g., FOLFOXIRI before FOLFOX)
    _CHEMO_ABBREVS = [
        ('FOLFOXIRI', 'a chemotherapy regimen'),
        ('FOLFIRINOX', 'a chemotherapy combination'),
        ('FOLFOX', 'a chemotherapy regimen'),
        ('FOLFIRI', 'a chemotherapy regimen'),
        ('TCHP', 'a combination of chemotherapy and targeted therapy drugs'),
        ('THP', 'targeted therapy with chemotherapy'),
        ('AC-T', 'a chemotherapy regimen'),
        ('AC/T', 'a chemotherapy regimen'),
    ]
    for abbr, plain in _CHEMO_ABBREVS:
        if abbr in letter_text:
            letter_text = letter_text.replace(abbr, plain)
            warnings.append(f"[POST-LETTER] replaced '{abbr}' → '{plain}'")

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

    # 4. Fix "a specific treatment" placeholder — replace with "a medication"
    if "a specific treatment" in letter_text:
        letter_text = letter_text.replace("a specific treatment", "a medication")
        warnings.append("[POST-LETTER] replaced 'a specific treatment' → 'a medication'")

    # 5. LN sanity check: implausible lymph node counts for early stage
    ln_match = re.search(r'(?:spread to|involving|involved)\s+(\d+)\s+lymph nodes', letter_text, re.IGNORECASE)
    if ln_match:
        ln_count = int(ln_match.group(1))
        is_early = bool(re.search(r'early stage|stage I|stage II\b', letter_text, re.IGNORECASE))
        if is_early and ln_count > 10:
            # Replace specific count with generic wording
            letter_text = letter_text.replace(
                ln_match.group(0),
                "spread to some lymph nodes"
            )
            warnings.append(f"[POST-LETTER] fixed implausible LN count ({ln_count}) for early stage")

    # 6. Strip inline dosing details (e.g., "1500mg", "10mg daily", "25 mg/m2")
    dosing_pattern = re.compile(
        r'\s*\d+\.?\d*\s*(?:mg(?:/m[²2])?|mcg|units?|mEq)\s*'
        r'(?:(?:once|twice|three times|per|every|q|BID|TID|QID|daily|weekly|PO|IV)\s*)*'
        r'(?:a\s+day|per\s+day|once\s+daily|twice\s+daily)?',
        re.IGNORECASE
    )
    new_text = dosing_pattern.sub(' ', letter_text)
    if new_text != letter_text:
        letter_text = re.sub(r' {2,}', ' ', new_text)
        # Clean up artifacts: "to ." → ".", "to ," → ","
        letter_text = re.sub(r'\bto\s+([.,])', r'\1', letter_text)
        # Clean up "dose of X to ." → "dose of X."
        letter_text = re.sub(r'(\bto\b)\s*\.', '.', letter_text)
        warnings.append("[POST-LETTER] stripped dosing details from letter")

    # 7. Remove semantically repeated sentences (word overlap > 70%)
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
