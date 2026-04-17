"""
vLLM HTTP Client for medical note extraction pipeline.

Wraps vLLM's OpenAI-compatible API for text generation.
Handles prompt construction, generation config mapping, and health checks.
"""

import re
import openai
from typing import Dict, Optional

# Regex to strip Qwen3.5 thinking tags (e.g., <think>\n\n</think>)
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class VLLMClient:
    """Lightweight wrapper around vLLM's OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model_name: Optional[str] = None):
        self.client = openai.OpenAI(base_url=base_url, api_key="not-needed", timeout=120.0)
        self.model_name = model_name or "default"

    def generate(self, full_prompt: str, gen_config: Dict) -> str:
        """
        Send a completion request to vLLM server.

        Args:
            full_prompt: Complete prompt string (already formatted with ChatML tokens)
            gen_config: Dict with keys: max_new_tokens, do_sample, temperature, top_p

        Returns:
            Generated text string (stripped)
        """
        temperature = 0.0
        if gen_config.get("do_sample", False):
            temperature = gen_config.get("temperature", 0.6)

        response = self.client.completions.create(
            model=self.model_name,
            prompt=full_prompt,
            max_tokens=gen_config.get("max_new_tokens", 768),
            temperature=temperature,
            top_p=gen_config.get("top_p", 1.0),
            stop=["<|im_end|>"],
        )
        text = response.choices[0].text.strip()
        # Strip Qwen3.5 thinking tags
        text = _THINK_RE.sub("", text).strip()
        return text

    def health_check(self) -> bool:
        """Check if vLLM server is running."""
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False
