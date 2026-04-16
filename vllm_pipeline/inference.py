"""
Inference functions for vLLM pipeline.

Drop-in replacements for ult.py's run_model_with_cache_manual() and build_base_cache().
In vLLM mode, "cache" is just the base prompt string — vLLM handles KV cache internally
via --enable-prefix-caching.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import Dict, Optional, Tuple
from vllm_pipeline.vllm_client import VLLMClient


# Import ChatTemplate from parent
from ult import ChatTemplate


def build_base_prompt(text: str, definitions_context: str = "", chat_tmpl=None) -> str:
    """
    Build the base prompt string for a medical note.
    Equivalent to build_base_cache() but returns a string instead of KV cache.

    Args:
        text: The medical note text
        definitions_context: Optional medical term definitions
        chat_tmpl: ChatTemplate instance

    Returns:
        Base prompt string (system + note + ready acknowledgment)
    """
    if chat_tmpl is None:
        chat_tmpl = ChatTemplate("qwen2")

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
    return base_prompt


def vllm_generate(
    prompt_text: str,
    client: VLLMClient,
    generation_config: Dict,
    base_prompt: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """
    Generate text using vLLM API.
    Drop-in replacement for run_model_with_cache_manual().

    Args:
        prompt_text: The new prompt to append (e.g., task instruction)
        client: VLLMClient instance
        generation_config: Dict with max_new_tokens, do_sample, temperature, etc.
        base_prompt: The base prompt string (acts as "cache" - vLLM handles prefix caching)

    Returns:
        Tuple of (generated_text, base_prompt) — base_prompt unchanged for compatibility
    """
    if base_prompt is not None:
        full_prompt = base_prompt + prompt_text
    else:
        full_prompt = prompt_text

    result = client.generate(full_prompt, generation_config)
    return result, base_prompt
