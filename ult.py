import csv
import re
import torch
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


def build_base_cache(text, model, tokenizer):
    """
    Build a KV cache from a base prompt containing the note text.
    Returns the base_cache for reuse across multiple extraction tasks.
    """
    base_prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"You are a medical data extraction expert. You will be given a long medical note. "
        f"Your task is to answer a series of questions about it, one by one. "
        f"Respond *only* with the valid JSON object requested. Do not add markdown backticks or any other text."
        f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Here is the medical note:\n\n"
        f"--- BEGIN NOTE ---\n{text}\n--- END NOTE ---"
        f"\n\nI will now ask you to extract specific sections. "
        f"Please wait for my first extraction task."
        f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        f"{{\"status\": \"Understood. I have read the note and am ready.\"}}"
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


def extract_and_verify(prompts, model, tokenizer, gen_config, base_cache, verify=True):
    """
    Extract keypoints from a set of prompts using a shared KV cache,
    with optional faithfulness verification.

    Args:
        prompts: dict of {key: task_prompt_text}
        model: the loaded model
        tokenizer: the loaded tokenizer
        gen_config: generation config dict
        base_cache: pre-computed KV cache from build_base_cache()
        verify: whether to run faithfulness verification (default True)

    Returns:
        dict of {key: extracted_value}
    """
    keypoints = {}

    for key, task in prompts.items():
        task_prompt = (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{task}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        )
        answer, returned_cache = run_model_with_cache_manual(
            task_prompt, model, tokenizer, gen_config, kv_cache=base_cache
        )
        del returned_cache
        torch.cuda.empty_cache()
        gc.collect()

        if verify:
            verification_prompt = (
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"CONTEXT: You previously extracted the following information (Initial Answer):\n"
                f"--- BEGIN INITIAL ANSWER ---\n{answer}\n--- END INITIAL ANSWER ---\n\n"
                f"**CRITICAL TASK:** You must now act as a verifier. Review the Initial Answer against the original full medical note (which is stored in your memory/context).\n"
                f"1. **Faithfulness Check:** Check if every statement in the Initial Answer is strictly supported by the original medical note.\n"
                f"2. **Revision:** Generate a **Final Answer** by removing *any* part of the Initial Answer that is not supported by the original medical note. If the Initial Answer is fully supported, the Final Answer should be the same.\n"
                f"Return the result strictly as a JSON object with the key 'final_answer'.\n"
                f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            )
            final_answer_raw, _ = run_model_with_cache_manual(
                verification_prompt, model, tokenizer, gen_config, kv_cache=base_cache
            )
            torch.cuda.empty_cache()
            gc.collect()
            try:
                final_result = json.loads(final_answer_raw.strip().strip("```json").strip("```").strip())
                keypoints[key] = final_result.get('final_answer', answer)
            except json.JSONDecodeError:
                keypoints[key] = answer
        else:
            keypoints[key] = answer

    return keypoints
