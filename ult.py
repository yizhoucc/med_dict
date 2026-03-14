import csv
import re
import torch
import time
import copy
from typing import Dict, List, Tuple, Optional
import nltk
from nltk.corpus import stopwords, words
from collections import Counter
import textwrap
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import pandas as pd
import json
import gc
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

LINE_WIDTH = 140


def load_oncology_whitelist(path="data/oncology_drugs.txt"):
    drugs = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                drugs.add(line.lower())
    return drugs


def load_supportive_whitelist(path="data/supportive_care_drugs.txt"):
    """Load supportive care drug whitelist for filtering supportive_meds."""
    drugs = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                drugs.add(line.lower())
    return drugs


def filter_current_meds(parsed, whitelist):
    val = parsed.get("current_meds", "")
    if isinstance(val, list):
        filtered = [m for m in val if any(d in m.lower() for d in whitelist)]
        parsed["current_meds"] = filtered if filtered else ""
    elif isinstance(val, str):
        if not val or val.lower().startswith("not yet") or val.lower() == "none":
            parsed["current_meds"] = ""
        else:
            meds = [m.strip() for m in val.split(",")]
            filtered = [m for m in meds if any(d in m.lower() for d in whitelist)]
            parsed["current_meds"] = ", ".join(filtered) if filtered else ""
    return parsed


def filter_supportive_meds(parsed, supportive_whitelist, oncology_whitelist):
    """Filter supportive_meds to only keep cancer-treatment-related supportive drugs.

    Keeps a med if it matches either the supportive care whitelist OR the oncology
    whitelist (bone agents like denosumab appear in both contexts).
    """
    val = parsed.get("supportive_meds", "")
    if not val or not isinstance(val, str):
        return parsed
    if val.lower() in ("none", "n/a", ""):
        return parsed
    # Don't filter safe negative phrases (e.g., "None currently being taken.")
    val_clean = val.strip().lower()
    if val_clean.startswith("none") or val_clean.startswith("not ") or val_clean.startswith("no "):
        return parsed

    combined = supportive_whitelist | oncology_whitelist
    # Split by comma, semicolon, or newline
    meds = [m.strip() for m in re.split(r'[,;\n]', val) if m.strip()]
    filtered = [m for m in meds if any(d in m.lower() for d in combined)]
    parsed["supportive_meds"] = ", ".join(filtered) if filtered else ""
    return parsed


class ChatTemplate:
    """Abstraction for model-specific chat formatting.

    Eliminates hardcoded chat template tokens throughout the codebase.
    Supports llama3 and mistral; add new entries to TEMPLATES for other models.
    """

    TEMPLATES = {
        "llama3": {
            "bos": "<|begin_of_text|>",
            "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
            "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
            "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n",
            "turn_end": "<|eot_id|>",
        },
        "mistral": {
            "bos": "<s>",
            "system_start": "[INST] ",
            "user_start": "[INST] ",
            "assistant_start": "",
            "turn_end": " [/INST]",
        },
        "qwen2": {
            "bos": "",
            "system_start": "<|im_start|>system\n",
            "user_start": "<|im_start|>user\n",
            "assistant_start": "<|im_start|>assistant\n",
            "turn_end": "<|im_end|>\n",
        },
    }

    def __init__(self, template_name="llama3"):
        if template_name not in self.TEMPLATES:
            raise ValueError(f"Unknown chat template: {template_name}. Available: {list(self.TEMPLATES.keys())}")
        self.t = self.TEMPLATES[template_name]

    def system_user_assistant(self, system_text, user_text):
        """Full prompt: BOS + system + user + assistant start."""
        return (
            f"{self.t['bos']}{self.t['system_start']}{system_text}{self.t['turn_end']}"
            f"{self.t['user_start']}{user_text}{self.t['turn_end']}"
            f"{self.t['assistant_start']}"
        )

    def user_assistant(self, user_text):
        """Continuation prompt appended to existing cache: user turn + assistant start."""
        return (
            f"{self.t['user_start']}{user_text}{self.t['turn_end']}"
            f"{self.t['assistant_start']}"
        )

def mysave(df, base_filename='output/keysummary', extension = 'csv'):
    
    counter = 1
    output_filename = f"{base_filename}_{counter}{extension}"
    output_dir = os.path.dirname(base_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while os.path.exists(output_filename):
        output_filename = f"{base_filename}_{counter}{extension}"
        counter += 1
    df.to_csv(output_filename, index=True)
    print(f"DataFrame saved to '{output_filename}'")


def clone_cache(cache):
    """Clone KV cache to prevent in-place modification.

    DynamicCache (transformers >= 4.36) can be modified in-place during
    generation, polluting the shared base_cache. Tuple-of-tuples format
    is safe because generate creates new tensors.
    """
    if hasattr(cache, 'get_seq_length'):  # DynamicCache
        return copy.deepcopy(cache)
    return cache  # tuple format is safe


def repair_json_agent(model, tokenizer, broken_json_string: str) -> str:
    """
    Uses an LLM to repair a syntactically incorrect JSON string.
    """
    # 1. Define the repair prompt
    system_prompt = (
        "You are an expert JSON repair utility. "
        "The user will provide a string that is intended to be valid JSON but has syntax errors. "
        "Your task is to fix these errors (e.g., missing commas, unclosed brackets, improper quotes). "
        "You must return *only* the corrected, valid JSON object. "
        "Do not add any explanations, apologies, or conversational text like 'Here is the fixed JSON:'."
    )
    
    user_prompt = f"Fix this broken JSON string:\n\n{broken_json_string}"

    # 2. Format the prompt using the Llama 3.1 chat template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    # apply_chat_template handles all special tokens (BOS, EOS, etc.)
    # add_generation_prompt=True adds the <|start_header_id|>assistant<|end_header_id|>
    # which cues the model to start its response.
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 3. Generate the response
    # We use do_sample=False for a deterministic, non-creative task like fixing syntax.
    outputs = model.generate(
        input_ids,
        max_new_tokens=1024, # Adjust based on your expected JSON size
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    # 4. Decode *only* the new tokens, not the prompt
    # outputs[0] is the full sequence (prompt + generation)
    # input_ids.shape[1] is the length of the prompt
    response_ids = outputs[0][input_ids.shape[1]:]
    repaired_string = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    # 5. Clean up the output to be just the JSON
    # LLMs sometimes wrap JSON in markdown backticks.
    repaired_string = repaired_string.strip().strip("```json").strip("```").strip()
    
    return repaired_string


def robust_json_parser(model, tokenizer, text_to_parse: str):
    """
    Implements the full "try-repair-fallback" logic.
    
    Returns a tuple: (parsed_data, final_string, status)
    - parsed_data: The Python dictionary (or None on failure)
    - final_string: The string that was successfully parsed (or the original broken one on fallback)
    - status: "success", "repaired", or "fallback"
    """
    
    # 1. The "Fast Path" (Try)
    try:
        parsed_json = json.loads(text_to_parse)
        print("✅ Parse Success (Fast Path)")
        return parsed_json, text_to_parse, "success"
    except json.JSONDecodeError as e:
        print(f"⚠️ Parse Failed (Fast Path): {e}. Attempting repair...")
        
        # 2. The "Repair Path" (Repair)
        try:
            # Call your new agent
            repaired_string = repair_json_agent(model, tokenizer, text_to_parse)
            
            # Try to parse the *repaired* string
            parsed_json = json.loads(repaired_string)
            print("✅ Parse Success (Repair Path)")
            return parsed_json, repaired_string, "repaired"
        
        except Exception as e: # Catch JSON errors *and* potential model errors
            print(f"⚠️ Repair Failed: {e}. Using fallback...")
            
            # 3. The "Fallback Path" (Fallback)
            # Return the original broken text for the next agent
            return None, text_to_parse, "fallback"


def run_model(
    prompt_text: str, 
    model, 
    tokenizer, 
    generation_config: Dict, 
    kv_cache: Optional[Tuple[torch.Tensor]] = None
) -> Tuple[str, Tuple[torch.Tensor]]:
    """
    Runs the model with an optional KV cache for efficient, chained generation.
    
    Args:
        prompt_text: The new text to be processed. If a kv_cache is provided,
                     this should *only* be the new text to append.
        model: The loaded Hugging Face CausalLM model.
        tokenizer: The loaded Hugging Face tokenizer.
        generation_config: A dictionary of generation parameters.
        kv_cache: (Optional) The past_key_values from a previous model run.

    Returns:
        A tuple containing:
        - raw_output (str): The newly generated text.
        - new_kv_cache (Tuple[torch.Tensor]): The updated KV cache.
    """
    
    # 1. Tokenize the new input text
    # If kv_cache is provided, prompt_text is *only* the new text.
    # If kv_cache is None, prompt_text is the *full* text.
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # 2. Ensure pad_token_id is set for generation
    if "pad_token_id" not in generation_config:
        generation_config["pad_token_id"] = tokenizer.eos_token_id

    # 3. Run model generation
    with torch.no_grad():
        # Get the attention mask from the inputs
        attention_mask = inputs.get("attention_mask", torch.ones_like(inputs["input_ids"]))

        # Run the model
        # - past_key_values=kv_cache tells the model to use the "memory"
        # - return_dict_in_generate=True gives us a structured output
        #   that includes the new kv_cache
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            return_dict_in_generate=True,  # This is key!
            **generation_config
        )

    # 4. Extract the new, updated KV cache
    # This can be fed back into the function on the next call
    new_kv_cache = outputs.past_key_values

    # 5. Decode the response
    # outputs.sequences contains the *entire* sequence (prompt + new tokens)
    # We slice it from input_length to get *only* the newly generated tokens
    response_tokens = outputs.sequences[0][input_length:]
    raw_output = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    return raw_output.strip(), new_kv_cache


def myprint(original_text):
    wrapped_text = textwrap.fill(original_text, width=LINE_WIDTH)
    print(wrapped_text)


def print_json(data, indent=2):
    """Print JSON with clear structure."""
    if isinstance(data, str):
        data = json.loads(data)
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def txt_to_dict(file_path):
    data_dict = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines) - 1, 2):
            key = lines[i].strip()    # Odd line are key
            value = lines[i + 1].strip()  # Even line are value
            data_dict[key] = value

    return data_dict

        
def clean_model_output(text: str, fix_incomplete=True) -> str:
    """
    Clean up model-generated text with common fixes, not currently using
    """
    if not text:
        return ""
    
    # Basic cleanup
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace
    
    # Fix paragraph spacing
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # Fix punctuation spacing
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
    
    # Remove repetitive patterns (simple version)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if not cleaned_lines or line.strip() != cleaned_lines[-1].strip():
            cleaned_lines.append(line)
    text = '\n'.join(cleaned_lines)
    
    # Handle incomplete sentences
    if fix_incomplete and text and not text.endswith(('.', '!', '?')):
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            # Remove likely incomplete last sentence
            text = '.'.join(sentences[:-1]) + '.'
    
    return text.strip()


def run_model_with_cache_manual(
    prompt_text: str, 
    model, 
    tokenizer, 
    generation_config: Dict, 
    kv_cache: Optional[Tuple[torch.Tensor]] = None
) -> Tuple[str, Tuple[torch.Tensor]]:
    """
    Memory-efficient manual generation with proper cleanup.
    """
    
    # 1. Tokenize the new input text
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    # 2. Setup generation parameters
    max_new_tokens = generation_config.get("max_new_tokens", 256)
    eos_token_id = generation_config.get("eos_token_id", tokenizer.eos_token_id)
    if not isinstance(eos_token_id, (list, set)):
        eos_token_id = [eos_token_id]
    do_sample = generation_config.get("do_sample", False)
    
    # 3. Determine the starting position and build attention mask
    if kv_cache is not None:
        # Support both legacy tuple format and DynamicCache
        if hasattr(kv_cache, 'get_seq_length'):
            past_seq_len = kv_cache.get_seq_length()
        else:
            past_seq_len = kv_cache[0][0].shape[2]
        
        # Build full attention mask ONCE upfront
        attention_mask_cached = torch.ones((1, past_seq_len), dtype=torch.long, device=model.device)
        attention_mask_new = inputs.get("attention_mask", torch.ones_like(input_ids))
        attention_mask = torch.cat([attention_mask_cached, attention_mask_new], dim=1)
    else:
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    
    # 4. First forward pass to process the new prompt
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache,
            use_cache=True
        )
    
    current_cache = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    
    # Clear outputs to free memory
    del outputs
    
    # 5. Pre-allocate list for generated tokens
    generated_tokens = []
    
    # 6. Generation loop
    for step in range(max_new_tokens):
        # Sample next token
        if do_sample:
            temperature = generation_config.get("temperature", 1.0)
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Store the token value
        token_id = next_token.item()
        generated_tokens.append(token_id)
        
        # Check for EOS
        if token_id in eos_token_id:
            break
        
        # Prepare for next iteration - extend attention mask efficiently
        # Instead of concatenating, create a new mask of the right size
        new_length = attention_mask.shape[1] + 1
        new_attention_mask = torch.ones((1, new_length), dtype=torch.long, device=model.device)
        
        # Free old attention mask
        del attention_mask
        attention_mask = new_attention_mask
        
        # Forward pass with the new token
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=current_cache,
                use_cache=True
            )
        
        # Update cache and logits
        current_cache = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        
        # Clean up
        del outputs, next_token
    
    # 7. Decode the generated tokens
    raw_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # 8. Cleanup
    del attention_mask, next_token_logits, generated_tokens
    
    return raw_output.strip(), current_cache


def load_medical_dictionary(dict_path="data/formaldef.txt"):
    """Load medical dictionary from formaldef.txt.
    Returns dict of {term_lower: (original_term, definition)}.
    """
    terms = {}
    if not os.path.exists(dict_path):
        return terms
    with open(dict_path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        term_line = lines[i].strip()
        term = re.sub(r'Listen to pronunciation.*', '', term_line).strip()
        if term and i + 1 < len(lines):
            definition = lines[i + 1].strip()
            if definition and len(term) > 2:
                terms[term.lower()] = (term, definition)
            i += 2
        else:
            i += 1
    return terms


# Terms that are abbreviations or have ambiguous/specialized meanings.
# These are the ones an 8B model is most likely to misunderstand.
# Scored by confusion risk: abbreviations > biomarkers with numbers > ambiguous terms.
INJECT_PRIORITY_TERMS = {
    # Abbreviations - highest priority (model may not know what these stand for)
    "dcis", "idc", "ilc", "fish", "ihc", "suv",
    "brca1", "brca2", "tnbc",
    # Biomarkers with numeric values - model confuses units/meaning
    "ki-67 proliferation index", "ki-67 score", "ki-67",
    "her2", "her2 positive", "her2 negative", "her2/neu",
    "er positive", "er negative",
    "pr positive", "pr negative",
    # Response classification terms - model confuses with common English
    "stable disease", "progressive disease", "partial response", "complete response",
    # Treatment intent terms - model confuses with visit purpose
    "neoadjuvant therapy", "adjuvant therapy", "adjuvant chemotherapy",
    "palliative therapy", "palliative care",
    # Staging/grading - model confuses grade vs stage vs size
    "sentinel lymph node", "sentinel node biopsy",
    "lymphovascular invasion",
    # Procedures model may confuse
    "lumpectomy", "mastectomy",
    "aromatase inhibitor",
    # Receptor/molecular testing
    "estrogen receptor", "progesterone receptor",
    "triple-negative breast cancer", "triple negative breast cancer",
}


def find_relevant_definitions(note_text, dictionary, max_terms=15):
    """Scan note for medical terms and return definitions for terms
    the model is most likely to misunderstand.

    Strategy:
    1. Only match terms in INJECT_PRIORITY_TERMS (curated high-confusion list)
    2. Prefer longer/more specific term matches over short ones
    3. Cap at max_terms to control context size (~50 tokens per definition)

    Returns list of (term, definition) tuples.
    """
    if not dictionary:
        return []

    note_lower = note_text.lower()
    matched = []

    for priority_term in INJECT_PRIORITY_TERMS:
        if priority_term in note_lower and priority_term in dictionary:
            term_orig, definition = dictionary[priority_term]
            matched.append((term_orig, definition))

    # Deduplicate: if both "ki-67" and "ki-67 score" matched, keep only the longer one
    matched.sort(key=lambda x: -len(x[0]))
    seen_short = set()
    deduped = []
    for term, defn in matched:
        short = term.lower().split()[0]  # first word
        if short not in seen_short:
            deduped.append((term, defn))
            seen_short.add(short)

    return deduped[:max_terms]


def format_definitions_context(definitions):
    """Format matched definitions into a context string for injection."""
    if not definitions:
        return ""
    lines = ["Medical term definitions for reference:"]
    for term, defn in definitions:
        # Truncate long definitions to ~150 chars
        short_defn = defn[:200].rsplit(' ', 1)[0] if len(defn) > 200 else defn
        lines.append(f"- {term}: {short_defn}")
    return "\n".join(lines)


def build_base_cache(text, model, tokenizer, definitions_context="", chat_tmpl=None):
    """
    Build a KV cache from a base prompt containing the note text.
    Returns the base_cache for reuse across multiple extraction tasks.

    Args:
        text: The medical note text
        model: The loaded model
        tokenizer: The loaded tokenizer
        definitions_context: Optional medical term definitions to inject
        chat_tmpl: ChatTemplate instance (defaults to llama3)
    """
    if chat_tmpl is None:
        chat_tmpl = ChatTemplate("llama3")

    defs_section = ""
    if definitions_context:
        defs_section = f"\n\n{definitions_context}\n"

    system_msg = (
        f"You are a medical data extraction expert. You will be given a long medical note. "
        f"Your task is to answer a series of questions about it, one by one. "
        f"You MUST respond with valid JSON only. Match the exact schema provided in each task. "
        f"No markdown backticks, no explanations, no text before or after the JSON object."
        f"{defs_section}"
    )
    user_msg = (
        f"Here is the medical note:\n\n"
        f"--- BEGIN NOTE ---\n{text}\n--- END NOTE ---"
        f"\n\nI will now ask you to extract specific sections. "
        f"Please wait for my first extraction task."
    )
    base_prompt = (
        chat_tmpl.system_user_assistant(system_msg, user_msg)
        + '{"status": "Understood. I have read the note and am ready."}'
        + chat_tmpl.t['turn_end']
    )

    with torch.no_grad():
        inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True
        )
        base_cache = outputs.past_key_values
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()

    return base_cache


def try_parse_json(text):
    """Try to parse text as JSON with common cleanup.

    Strips markdown backticks, fixes single quotes, removes trailing commas.
    Returns parsed dict on success, None on failure.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()
    # Strip markdown code fences
    cleaned = cleaned.strip("```json").strip("```").strip()

    # Try direct parse first
    try:
        result = json.loads(cleaned)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    # Try fixing common issues
    try:
        fixed = cleaned.replace("'", '"')
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        result = json.loads(fixed)
        return result if isinstance(result, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting JSON from surrounding text
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return result if isinstance(result, dict) else None
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def extract_schema_from_prompt(prompt_text):
    """Extract the JSON schema example from a prompt string.

    Looks for the {...} block that serves as the expected output format.
    """
    # Find JSON-like blocks in the prompt
    matches = list(re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', prompt_text, re.DOTALL))
    if matches:
        # Return the last match (usually the schema is at the end of the prompt)
        return matches[-1].group()
    return "{}"


def extract_schema_keys(prompt_text):
    """Extract expected JSON keys from the schema in a prompt.

    Returns a set of key names, or empty set if parsing fails.
    """
    schema_str = extract_schema_from_prompt(prompt_text)
    parsed = try_parse_json(schema_str)
    if parsed and isinstance(parsed, dict):
        return set(parsed.keys())
    # Fallback: regex extract quoted keys before colons
    keys = re.findall(r'"([^"]+)"\s*:', schema_str)
    return set(keys) if keys else set()


def extract_and_verify(prompts, model, tokenizer, gen_config, base_cache, verify=True, chat_tmpl=None, **kwargs):
    """
    Extract keypoints with agentic self-correction:
    1. Extract answer from model
    2. Format repair: if not valid JSON, ask model to reformat
    3. Faithfulness verify: check if answer is supported by source text
       - If not faithful, re-extract with guidance

    Args:
        prompts: dict of {key: task_prompt_text}
        model: the loaded model
        tokenizer: the loaded tokenizer
        gen_config: generation config dict
        base_cache: pre-computed KV cache from build_base_cache()
        verify: whether to run faithfulness verification (default True)
        chat_tmpl: ChatTemplate instance (defaults to llama3)

    Returns:
        dict of {key: extracted_value (dict if JSON parseable, string otherwise)}
    """
    if chat_tmpl is None:
        chat_tmpl = ChatTemplate("llama3")

    keypoints = {}

    for key, task in prompts.items():
        key_start = time.time()
        repaired = False
        re_extracted = False

        # --- Step 1: Extract ---
        task_prompt = chat_tmpl.user_assistant(task)
        answer, returned_cache = run_model_with_cache_manual(
            task_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
        )
        del returned_cache
        torch.cuda.empty_cache()
        gc.collect()

        # --- Step 2: Format repair ---
        parsed = try_parse_json(answer)
        if parsed is None:
            schema = extract_schema_from_prompt(task)
            repair_prompt = chat_tmpl.user_assistant(
                f"Your previous response was:\n{answer}\n\n"
                f"This is not valid JSON. Reformat it into valid JSON matching this exact schema:\n"
                f"{schema}\n"
                f"Return ONLY the JSON object, nothing else."
            )
            repaired_answer, _ = run_model_with_cache_manual(
                repair_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
            )
            torch.cuda.empty_cache()
            gc.collect()

            parsed = try_parse_json(repaired_answer)
            if parsed is not None:
                answer = repaired_answer
                repaired = True

        # --- Step 3: Faithfulness verify ---
        if verify:
            verification_prompt = chat_tmpl.user_assistant(
                f"CONTEXT: You previously extracted the following information:\n"
                f"--- BEGIN EXTRACTED ---\n{answer}\n--- END EXTRACTED ---\n\n"
                f"Check if every statement in the extraction is strictly supported by "
                f"the original medical note in your context.\n"
                f"Return a JSON object: {{\"faithful\": true}} if fully supported, "
                f"or {{\"faithful\": false, \"issues\": \"describe what is not supported\"}} if not.\n"
                f"Return ONLY the JSON object."
            )
            check_raw, _ = run_model_with_cache_manual(
                verification_prompt, model, tokenizer,
                {"max_new_tokens": 200, "do_sample": False, "eos_token_id": gen_config.get("eos_token_id", tokenizer.eos_token_id)},
                kv_cache=clone_cache(base_cache)
            )
            torch.cuda.empty_cache()
            gc.collect()

            check_parsed = try_parse_json(check_raw)
            is_faithful = True
            issues = ""
            if check_parsed:
                is_faithful = check_parsed.get("faithful", True)
                issues = check_parsed.get("issues", "")
            else:
                # Fallback: look for "false" in the raw output
                is_faithful = "false" not in check_raw.lower()

            if not is_faithful:
                # Re-extract with guidance
                re_extract_prompt = chat_tmpl.user_assistant(
                    f"Your previous extraction had issues: {issues}\n"
                    f"Please re-extract. Here is the original task:\n{task}\n"
                    f"Be very careful to only include information that is explicitly stated in the note. "
                    f"Do not infer or add information."
                )
                answer, _ = run_model_with_cache_manual(
                    re_extract_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
                )
                torch.cuda.empty_cache()
                gc.collect()
                re_extracted = True

                # Format repair again on re-extracted answer
                parsed = try_parse_json(answer)
                if parsed is None:
                    schema = extract_schema_from_prompt(task)
                    repair_prompt = chat_tmpl.user_assistant(
                        f"Your previous response was:\n{answer}\n\n"
                        f"Reformat as valid JSON matching this schema:\n{schema}\n"
                        f"Return ONLY the JSON object."
                    )
                    repaired_answer, _ = run_model_with_cache_manual(
                        repair_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
                    )
                    torch.cuda.empty_cache()
                    gc.collect()
                    new_parsed = try_parse_json(repaired_answer)
                    if new_parsed is not None:
                        parsed = new_parsed
                        answer = repaired_answer

        # --- Step 4: Temporal check (for plan-related keys) ---
        # Verify that plan extractions don't include past/completed items
        temporal_cleaned = False
        plan_keys = {'Therapy_plan', 'Procedure_Plan', 'Imaging_Plan', 'Lab_Plan',
                      'Medication_Plan', 'Medication_Plan_chatgpt'}
        if key in plan_keys and parsed is not None:
            temporal_prompt = chat_tmpl.user_assistant(
                f"Review this extraction for a PLAN section:\n"
                f"--- BEGIN ---\n{answer}\n--- END ---\n\n"
                f"This should contain ONLY current or future plans.\n"
                f"Remove any items that are PAST/COMPLETED (indicated by past tense, "
                f"past dates, 'underwent', 's/p', 'completed', 'had', 'was done').\n"
                f"Keep items that are current ('continue', 'currently on') or future "
                f"('will', 'plan to', 'pending', 'scheduled', 'next due', 'consider').\n"
                f"Return the cleaned JSON with only current/future items. "
                f"If nothing remains, return the appropriate 'None' or 'No ... planned.' value.\n"
                f"Return ONLY the JSON object."
            )
            cleaned_raw, _ = run_model_with_cache_manual(
                temporal_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
            )
            torch.cuda.empty_cache()
            gc.collect()

            cleaned_parsed = try_parse_json(cleaned_raw)
            if cleaned_parsed is not None:
                # Validate: reject leaked artifacts like {"faithful": true}
                # Cleaned result must share at least one key with original extraction
                original_keys = set(parsed.keys()) if isinstance(parsed, dict) else set()
                cleaned_keys = set(cleaned_parsed.keys())
                has_overlap = bool(original_keys & cleaned_keys)
                if has_overlap and cleaned_parsed != parsed:
                    parsed = cleaned_parsed
                    answer = cleaned_raw
                    temporal_cleaned = True

        # --- Step 5: Store result ---
        if parsed is not None:
            keypoints[key] = parsed
        else:
            keypoints[key] = answer

        elapsed = time.time() - key_start
        flags = []
        if repaired:
            flags.append("repaired")
        if re_extracted:
            flags.append("re-extracted")
        if temporal_cleaned:
            flags.append("temporal-cleaned")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"    {key}: {elapsed:.1f}s{flag_str}")

    return keypoints


# --- V2 Pipeline constants ---
PLAN_KEYS = {'Therapy_plan', 'Procedure_Plan', 'Imaging_Plan', 'Lab_Plan',
             'Medication_Plan', 'Medication_Plan_chatgpt'}

# Patterns that G3 should NOT empty — these are valid "nothing here" answers
SAFE_NEGATIVE_PATTERNS = [
    r"^no\s+\w+\s+planned\.?$",         # "No labs planned.", "No imaging planned."
    r"^no\s+\w+\s+in\s+note\.?$",       # "No labs in note."
    r"^none[\s.]?",                       # "None", "None.", "None planned.", "None are planned."
    r"^not\s+discussed",                  # "Not discussed during this visit."
    r"^not\s+yet\s+on\s+treatment",      # "Not yet on treatment..."
    r"^not\s+mentioned",                  # "Not mentioned in note"
    r"^no\s+procedures\s+planned",       # "No procedures planned."
]

# Standard classification values that G3 should preserve (these are category labels, not free text)
CLASSIFICATION_VALUES = {
    "new patient", "follow up", "follow-up", "established patient",
    "curative", "palliative", "adjuvant", "neoadjuvant", "risk reduction",
    "yes", "no", "in-person", "telehealth",
}

# Words indicating the model made a reasonable inference (G3 should preserve these)
INFERENCE_MARKERS = ["approximately", "likely", "possibly", "probably", "estimated",
                     "not sure", "not certain", "uncertain", "equivocal", "originally"]

# Fields where G6 should NOT change the value (categorical classifications, not free text)
G6_NO_MODIFY_FIELDS = {"Patient type", "second opinion", "in-person", "goals_of_treatment"}

# --- POST-PROC: Hard filter for Procedure_Plan contamination ---
# If procedure_plan contains ONLY imaging or radiotherapy terms, empty it.
IMAGING_TERMS = [
    "mri", "ct scan", "ct ", "pet", "pet/ct", "petct", "pet-ct",
    "dexa", "dxa", "bone density", "bone scan",
    "mammogram", "mammography", "ultrasound", "sonogram",
    "x-ray", "xray", "x ray", "echocardiogram", "tte", "echo",
    "brain mri", "mri brain", "mri pelvis", "mri spine",
    "diagnostic mammogram", "bilateral mammogram",
]
RADIOTHERAPY_TERMS = [
    "radiation", "radiotherapy", "xrt",
    "chest wall rt", "whole brain radiation", "sbrt", "imrt",
    "proton therapy", "brachytherapy",
]
# Short "RT" checked with word boundary to avoid matching "port", "report", etc.
_RT_WORD_RE = re.compile(r'\brt\b', re.IGNORECASE)

def _is_radiotherapy(text_lower):
    """Check if text is about radiotherapy (not port placement, etc.)."""
    return any(t in text_lower for t in RADIOTHERAPY_TERMS) or bool(_RT_WORD_RE.search(text_lower))

def filter_procedure_plan(parsed, gate_log):
    """Remove imaging and radiotherapy items from procedure_plan."""
    val = parsed.get("procedure_plan", "")
    if not val or not isinstance(val, str):
        return parsed
    val_lower = val.strip().lower()
    if val_lower in ("no procedures planned.", "no procedures planned", "none", "none planned."):
        return parsed
    # Split by comma/semicolon, check each item
    items = [item.strip() for item in re.split(r'[,;]', val) if item.strip()]
    kept = []
    removed = []
    for item in items:
        item_lower = item.lower()
        is_imaging = any(t in item_lower for t in IMAGING_TERMS)
        is_rt = _is_radiotherapy(item_lower)
        if is_imaging or is_rt:
            removed.append(item)
        else:
            kept.append(item)
    if removed:
        if kept:
            parsed["procedure_plan"] = ", ".join(kept)
        else:
            parsed["procedure_plan"] = "No procedures planned."
        gate_log.append(f"      [POST-PROC-FILTER] removed from procedure_plan: {removed}")
    return parsed

def _is_safe_negative(text):
    """Check if text is a valid 'nothing here' response that G3 should preserve."""
    if not text or not isinstance(text, str):
        return False
    text_clean = text.strip().lower()
    return any(re.match(p, text_clean) for p in SAFE_NEGATIVE_PATTERNS)

def _is_classification_or_inference(text):
    """Check if text is a standard classification value or contains inference markers."""
    if not text or not isinstance(text, str):
        return False
    text_clean = text.strip().lower()
    if text_clean in CLASSIFICATION_VALUES:
        return True
    return any(marker in text_clean for marker in INFERENCE_MARKERS)

VAGUE_TERMS = ["staging workup", "new symptoms", "will start medications",
               "further workup", "appropriate treatment", "as above",
               "as discussed", "per discussion"]


def _has_vague_terms(text):
    """Check if text contains any vague terms that need specificity improvement."""
    text_lower = text.lower()
    return any(term in text_lower for term in VAGUE_TERMS)


def extract_and_verify_v2(prompts, model, tokenizer, gen_config, base_cache, verify=True, chat_tmpl=None, oncology_whitelist=None, gate_config=None, supportive_whitelist=None):
    """
    V2 extraction pipeline with 6 independent gates.
    Each gate fixes one specific issue (trim, don't redo).

    Gates:
      1. FORMAT   - Parse JSON, LLM reformat if needed
      2. SCHEMA   - Validate keys match expected schema
      3. FAITHFUL - Trim unsupported claims (not re-extract)
      4. TEMPORAL - Remove past/completed items from plan keys
      5. SPECIFIC - Replace vague language with specifics
      6. SEMANTIC - Check each value answers its field's question

    Args:
        prompts: dict of {key: task_prompt_text}
        model: the loaded model
        tokenizer: the loaded tokenizer
        gen_config: generation config dict
        base_cache: pre-computed KV cache from build_base_cache()
        verify: whether to run faithfulness/temporal/specificity gates (default True)
        chat_tmpl: ChatTemplate instance (defaults to llama3)
        gate_config: optional dict with gate behavior overrides:
            preserve_negatives: bool - G3 won't empty safe negative phrases (default False)
            skip_empty_fields: bool - G4/G6 won't add content to empty fields (default False)

    Returns:
        dict of {key: extracted_value (dict if JSON parseable, string otherwise)}
    """
    if chat_tmpl is None:
        chat_tmpl = ChatTemplate("llama3")
    if gate_config is None:
        gate_config = {}

    keypoints = {}

    for key, task in prompts.items():
        key_start = time.time()
        flags = []
        gate_log = []  # per-gate detailed log

        # --- Step 1: Extract ---
        task_prompt = chat_tmpl.user_assistant(task)
        cache_for_extract = clone_cache(base_cache)
        answer, returned_cache = run_model_with_cache_manual(
            task_prompt, model, tokenizer, gen_config, kv_cache=cache_for_extract
        )
        del returned_cache, cache_for_extract
        torch.cuda.empty_cache()
        gc.collect()

        gate_log.append(f"      [EXTRACT] raw={answer[:200]}")

        # --- Gate 1: FORMAT ---
        parsed = try_parse_json(answer)
        if parsed is None:
            schema = extract_schema_from_prompt(task)
            repair_prompt = chat_tmpl.user_assistant(
                f"Your previous response was:\n{answer}\n\n"
                f"This is not valid JSON. Reformat into valid JSON matching this exact schema:\n"
                f"{schema}\n"
                f"Return ONLY the JSON object."
            )
            repaired_answer, _ = run_model_with_cache_manual(
                repair_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
            )
            torch.cuda.empty_cache()
            gc.collect()

            parsed = try_parse_json(repaired_answer)
            if parsed is not None:
                answer = repaired_answer
                flags.append("format-repaired")
                gate_log.append(f"      [G1-FORMAT] repaired -> keys={list(parsed.keys())}")
            else:
                gate_log.append(f"      [G1-FORMAT] repair FAILED, raw={repaired_answer[:150]}")
        else:
            gate_log.append(f"      [G1-FORMAT] ok, keys={list(parsed.keys())}")

        # --- Gate 2: SCHEMA ---
        if parsed is not None:
            expected_keys = extract_schema_keys(task)
            if expected_keys and not (expected_keys & set(parsed.keys())):
                actual_keys = list(parsed.keys())
                schema_fix_prompt = chat_tmpl.user_assistant(
                    f"Your JSON has incorrect keys.\n"
                    f"Expected keys: {sorted(expected_keys)}\n"
                    f"Your keys: {actual_keys}\n"
                    f"Rewrite using the correct keys from the schema. Keep your extracted content.\n"
                    f"Return ONLY the JSON object."
                )
                fixed_raw, _ = run_model_with_cache_manual(
                    schema_fix_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
                )
                torch.cuda.empty_cache()
                gc.collect()

                fixed_parsed = try_parse_json(fixed_raw)
                if fixed_parsed is not None and (expected_keys & set(fixed_parsed.keys())):
                    parsed = fixed_parsed
                    answer = fixed_raw
                    flags.append("schema-fixed")
                    gate_log.append(f"      [G2-SCHEMA] fixed: {actual_keys} -> {list(fixed_parsed.keys())}")
                else:
                    gate_log.append(f"      [G2-SCHEMA] fix FAILED, expected={sorted(expected_keys)}, got={actual_keys}")
            else:
                gate_log.append(f"      [G2-SCHEMA] ok")

        # --- Gate 3: FAITHFULNESS (trim mode) ---
        if verify and parsed is not None:
            original_keys = set(parsed.keys())
            before_values = {k: str(v)[:80] for k, v in parsed.items()}
            faith_prompt = chat_tmpl.user_assistant(
                f"Review this extraction against the original medical note in your context:\n"
                f"--- BEGIN ---\n{answer}\n--- END ---\n\n"
                f"For each key-value pair, check: is the value explicitly stated or directly "
                f"inferable from the note?\n"
                f"Rules:\n"
                f"- KEEP the value if it is supported by the note, even if the wording differs slightly.\n"
                f"- KEEP the value if it is a reasonable clinical summary of what the note says.\n"
                f"- ONLY replace a value with an empty string if it contains information that "
                f"clearly CONTRADICTS the note or is completely fabricated (not mentioned at all).\n"
                f"- Watch for GENERIC BOILERPLATE that is NOT in the note: phrases like "
                f"\"tolerating therapy well\", \"symptoms improved\", \"patient is doing well\" "
                f"must be EMPTY if the note does not explicitly say this.\n"
                f"- When in doubt, KEEP the original value.\n"
                f"- Do NOT remove any keys. Return ALL original keys.\n"
                f"Return ONLY the JSON object with all original keys preserved."
            )
            cleaned_raw, _ = run_model_with_cache_manual(
                faith_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
            )
            torch.cuda.empty_cache()
            gc.collect()

            cleaned_parsed = try_parse_json(cleaned_raw)
            if cleaned_parsed is not None:
                cleaned_keys = set(cleaned_parsed.keys())
                if original_keys & cleaned_keys:
                    # Restore any keys that were dropped by the LLM
                    missing_keys = original_keys - cleaned_keys
                    for mk in missing_keys:
                        cleaned_parsed[mk] = parsed[mk]
                    if missing_keys:
                        flags.append(f"faith-restored-{len(missing_keys)}keys")
                    # Log per-field changes
                    changed_fields = []
                    emptied_fields = []
                    for k in original_keys:
                        old_val = parsed.get(k)
                        new_val = cleaned_parsed.get(k)
                        if old_val != new_val:
                            old_s = str(old_val)[:60]
                            new_s = str(new_val)[:60]
                            changed_fields.append(k)
                            if not new_val or new_val == "":
                                emptied_fields.append(k)
                            gate_log.append(f"      [G3-FAITH] {k}: \"{old_s}\" -> \"{new_s}\"")
                    if changed_fields:
                        flags.append("faith-trimmed")
                        if emptied_fields:
                            gate_log.append(f"      [G3-FAITH] EMPTIED: {emptied_fields}")
                    else:
                        gate_log.append(f"      [G3-FAITH] no changes (all supported)")

                    # Protect safe negatives from being emptied
                    if gate_config.get("preserve_negatives") and emptied_fields:
                        for k in emptied_fields:
                            orig_val = str(parsed.get(k, ""))  # pre-G3 value (full, not truncated)
                            if _is_safe_negative(orig_val):
                                cleaned_parsed[k] = parsed[k]  # restore original
                                gate_log.append(f"      [G3-PROTECT] {k}: restored safe negative \"{orig_val[:60]}\"")

                    # If G3 emptied ALL fields and none were safe negatives,
                    # G3 is likely being too aggressive — revert to original extraction
                    if gate_config.get("preserve_negatives") and emptied_fields:
                        still_empty = [k for k in emptied_fields
                                       if not cleaned_parsed.get(k) or cleaned_parsed.get(k) == ""]
                        if still_empty and len(still_empty) == len(original_keys):
                            # G3 emptied everything — revert
                            for k in still_empty:
                                cleaned_parsed[k] = parsed[k]
                                gate_log.append(f"      [G3-REVERT] {k}: reverted (G3 emptied all fields)")

                    # Also revert individual fields that had classification/inference values
                    # G3 should not empty "New patient", "Approximately Stage II", "Not sure", etc.
                    if gate_config.get("preserve_negatives") and emptied_fields:
                        for k in emptied_fields:
                            if cleaned_parsed.get(k) == "" or not cleaned_parsed.get(k):
                                orig_val = str(parsed.get(k, ""))
                                if _is_classification_or_inference(orig_val):
                                    cleaned_parsed[k] = parsed[k]
                                    gate_log.append(f"      [G3-REVERT-INFER] {k}: restored classification/inference \"{orig_val[:60]}\"")

                    # Protect Type_of_Cancer from G3 removing receptor status
                    # If G3 emptied or trimmed Type_of_Cancer, and original had receptor info, restore
                    if gate_config.get("preserve_negatives") and "Type_of_Cancer" in changed_fields:
                        orig_toc = str(parsed.get("Type_of_Cancer", ""))
                        new_toc = str(cleaned_parsed.get("Type_of_Cancer", ""))
                        has_receptor = any(r in orig_toc.lower() for r in ["er+", "er-", "pr+", "pr-", "her2", "triple negative"])
                        if has_receptor and (not new_toc or len(new_toc) < len(orig_toc) * 0.5):
                            cleaned_parsed["Type_of_Cancer"] = parsed["Type_of_Cancer"]
                            gate_log.append(f"      [G3-PROTECT-RECEPTOR] Type_of_Cancer: restored \"{orig_toc[:60]}\"")

                    parsed = cleaned_parsed
                    answer = json.dumps(cleaned_parsed, ensure_ascii=False)
                else:
                    gate_log.append(f"      [G3-FAITH] REJECTED (no key overlap), kept original")
            else:
                gate_log.append(f"      [G3-FAITH] parse FAILED, kept original")

        # --- Gate 4: TEMPORAL (plan keys only) ---
        if verify and parsed is not None and key in PLAN_KEYS:
            # Skip G4 if all fields are empty (nothing to filter)
            all_empty_g4 = all(
                not v or v == "" or (isinstance(v, str) and _is_safe_negative(v))
                for v in parsed.values()
            )
            if gate_config.get("skip_empty_fields") and all_empty_g4:
                gate_log.append(f"      [G4-TEMPORAL] skipped (all fields empty/negative)")
            else:
                original_keys = set(parsed.keys())
                before_g4 = {k: str(v)[:80] for k, v in parsed.items()}
                temporal_prompt = chat_tmpl.user_assistant(
                    f"Review this extraction for a PLAN section:\n"
                    f"--- BEGIN ---\n{answer}\n--- END ---\n\n"
                    f"Remove any PAST/COMPLETED items (past tense, past dates, "
                    f"\"underwent\", \"s/p\", \"completed\").\n"
                    f"Keep only CURRENT (\"continue\", \"currently on\") and FUTURE "
                    f"(\"will\", \"plan to\", \"pending\") items.\n"
                    f"Return the cleaned JSON. If nothing remains, use the appropriate \"None\" value.\n"
                    f"Return ONLY the JSON object."
                )
                cleaned_raw, _ = run_model_with_cache_manual(
                    temporal_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
                )
                torch.cuda.empty_cache()
                gc.collect()

                cleaned_parsed = try_parse_json(cleaned_raw)
                if cleaned_parsed is not None:
                    cleaned_keys = set(cleaned_parsed.keys())
                    if original_keys & cleaned_keys:
                        if cleaned_parsed != parsed:
                            flags.append("temporal-cleaned")
                            for k in original_keys:
                                if parsed.get(k) != cleaned_parsed.get(k):
                                    gate_log.append(f"      [G4-TEMPORAL] {k}: \"{before_g4.get(k,'')}\" -> \"{str(cleaned_parsed.get(k,''))[:80]}\"")
                        else:
                            gate_log.append(f"      [G4-TEMPORAL] no changes")
                        parsed = cleaned_parsed
                        answer = cleaned_raw
                    else:
                        gate_log.append(f"      [G4-TEMPORAL] REJECTED (no key overlap)")
                else:
                    gate_log.append(f"      [G4-TEMPORAL] parse FAILED")

        # --- Gate 5: SPECIFICITY (conditional) ---
        if verify and parsed is not None and _has_vague_terms(answer):
            original_keys = set(parsed.keys())
            before_g5 = {k: str(v)[:80] for k, v in parsed.items()}
            specificity_prompt = chat_tmpl.user_assistant(
                f"This extraction contains vague language. Replace vague terms with SPECIFIC "
                f"details from the original medical note:\n"
                f"- \"staging workup\" → list the actual tests ordered "
                f"(e.g., \"CT chest/abd/pelvis, bone scan, MRI brain\")\n"
                f"- \"new symptoms\" → describe the specific symptoms\n"
                f"- \"will start medications\" → name the specific drugs and doses\n"
                f"- \"as above\" or \"as discussed\" → state what was actually discussed\n\n"
                f"Original extraction:\n{answer}\n\n"
                f"Return the improved JSON with specific details. Return ONLY the JSON object."
            )
            improved_raw, _ = run_model_with_cache_manual(
                specificity_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
            )
            torch.cuda.empty_cache()
            gc.collect()

            improved_parsed = try_parse_json(improved_raw)
            if improved_parsed is not None:
                improved_keys = set(improved_parsed.keys())
                if original_keys & improved_keys:
                    if improved_parsed != parsed:
                        flags.append("specificity-improved")
                        for k in original_keys:
                            if parsed.get(k) != improved_parsed.get(k):
                                gate_log.append(f"      [G5-SPECIFIC] {k}: \"{before_g5.get(k,'')}\" -> \"{str(improved_parsed.get(k,''))[:80]}\"")
                    else:
                        gate_log.append(f"      [G5-SPECIFIC] no changes")
                    parsed = improved_parsed
                    answer = improved_raw
                else:
                    gate_log.append(f"      [G5-SPECIFIC] REJECTED (no key overlap)")
            else:
                gate_log.append(f"      [G5-SPECIFIC] parse FAILED")

        # --- Gate 6: SEMANTIC RELEVANCE ---
        if verify and parsed is not None:
            # Skip G6 if all fields are empty/safe-negative (nothing to validate)
            all_empty_g6 = all(
                not v or v == "" or (isinstance(v, str) and _is_safe_negative(v))
                for v in parsed.values()
            )
            if gate_config.get("skip_empty_fields") and all_empty_g6:
                gate_log.append(f"      [G6-SEMANTIC] skipped (all fields empty/negative)")
            else:
                # Extract the field descriptions from the prompt schema
                schema_str = extract_schema_from_prompt(task)
                original_keys_g6 = set(parsed.keys())
                before_g6 = {k: str(v)[:80] for k, v in parsed.items()}
                semantic_prompt = chat_tmpl.user_assistant(
                    f"Check if each value in this extraction actually answers the question asked by its field.\n\n"
                    f"The field definitions are:\n{schema_str}\n\n"
                    f"The extraction is:\n{answer}\n\n"
                    f"For each field, check: does the value answer what the field is asking for?\n"
                    f"Common errors to catch:\n"
                    f"- goals_of_treatment should be the INTENT of treatment (curative, palliative, risk reduction), "
                    f"NOT the purpose of this visit (follow-up, discuss options)\n"
                    f"- response_assessment should be how the cancer is CURRENTLY responding (imaging, labs, exam), "
                    f"NOT future plans (will start, plan to, she will have)\n"
                    f"- current_meds should be medications the patient is CURRENTLY taking, "
                    f"NOT medications being discussed or planned\n"
                    f"- Nutrition referral means a REFERRAL TO a nutritionist/dietitian. "
                    f"General diet advice from the oncologist (e.g., 'eat calcium-rich food', 'take Vitamin D') is NOT a nutrition referral — keep it as 'None'\n\n"
                    f"If a value does not answer its field's question, replace it with the correct answer "
                    f"from the original medical note. If no correct answer exists in the note, use an appropriate "
                    f"default (e.g., 'Not yet on treatment — no response to assess.' for response_assessment "
                    f"when no treatment has started).\n"
                    f"If all values correctly answer their fields, return the JSON unchanged.\n"
                    f"Return ONLY the JSON object with all keys preserved."
                )
                semantic_raw, _ = run_model_with_cache_manual(
                    semantic_prompt, model, tokenizer, gen_config, kv_cache=clone_cache(base_cache)
                )
                torch.cuda.empty_cache()
                gc.collect()

                semantic_parsed = try_parse_json(semantic_raw)
                if semantic_parsed is not None:
                    semantic_keys = set(semantic_parsed.keys())
                    if original_keys_g6 & semantic_keys:
                        # Restore any dropped keys
                        for mk in original_keys_g6 - semantic_keys:
                            semantic_parsed[mk] = parsed[mk]

                        # Protect empty/negative fields from being filled with new content
                        if gate_config.get("skip_empty_fields"):
                            for k in original_keys_g6:
                                old_val = parsed.get(k)
                                new_val = semantic_parsed.get(k)
                                old_is_empty = not old_val or old_val == ""
                                if old_is_empty and new_val and new_val != "":
                                    semantic_parsed[k] = old_val  # keep empty
                                    gate_log.append(f"      [G6-PROTECT] {k}: blocked fill into empty field")

                        # Protect classification fields from G6 modification
                        if gate_config.get("preserve_negatives"):
                            for k in original_keys_g6:
                                if k in G6_NO_MODIFY_FIELDS and parsed.get(k) != semantic_parsed.get(k):
                                    semantic_parsed[k] = parsed[k]  # keep original
                                    gate_log.append(f"      [G6-PROTECT-CLASS] {k}: blocked modification of classification field")

                        changed = False
                        for k in original_keys_g6:
                            if parsed.get(k) != semantic_parsed.get(k):
                                changed = True
                                gate_log.append(f"      [G6-SEMANTIC] {k}: \"{before_g6.get(k,'')}\" -> \"{str(semantic_parsed.get(k,''))[:80]}\"")
                        if changed:
                            flags.append("semantic-fixed")
                        else:
                            gate_log.append(f"      [G6-SEMANTIC] ok (all values answer their fields)")
                        parsed = semantic_parsed
                        answer = json.dumps(semantic_parsed, ensure_ascii=False)
                    else:
                        gate_log.append(f"      [G6-SEMANTIC] REJECTED (no key overlap)")
                else:
                    gate_log.append(f"      [G6-SEMANTIC] parse FAILED")

        # --- POST-1: Strip extra keys ---
        if parsed is not None and expected_keys:
            extra = set(parsed.keys()) - expected_keys
            if extra:
                for k in extra:
                    del parsed[k]
                gate_log.append(f"      [POST-KEYS] stripped: {extra}")
                flags.append("stripped")

        # --- POST-2: Filter current_meds by whitelist ---
        if parsed is not None and key == "Current_Medications" and oncology_whitelist:
            before = parsed.get("current_meds", "")
            parsed = filter_current_meds(parsed, oncology_whitelist)
            after = parsed.get("current_meds", "")
            if str(before) != str(after):
                gate_log.append(f"      [POST-MEDS] {before} -> {after}")
                flags.append("meds-filtered")

        # --- POST-3: Filter supportive_meds by whitelist ---
        if parsed is not None and key == "Treatment_Changes" and supportive_whitelist and oncology_whitelist:
            before = parsed.get("supportive_meds", "")
            parsed = filter_supportive_meds(parsed, supportive_whitelist, oncology_whitelist)
            after = parsed.get("supportive_meds", "")
            if str(before) != str(after):
                gate_log.append(f"      [POST-SUPP] {before} -> {after}")
                flags.append("supp-filtered")

        # --- POST-4: Filter procedure_plan (remove imaging/radiotherapy) ---
        if parsed is not None and key == "Procedure_Plan":
            filter_procedure_plan(parsed, gate_log)

        # --- Store result ---
        if parsed is not None:
            keypoints[key] = parsed
        else:
            keypoints[key] = answer

        elapsed = time.time() - key_start
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        print(f"    {key}: {elapsed:.1f}s{flag_str}")
        # Print gate details
        for line in gate_log:
            print(line)

    return keypoints
