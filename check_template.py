"""Check Qwen chat template vs our implementation - after fix."""
from transformers import AutoTokenizer
import sys
sys.path.insert(0, '.')
from ult import ChatTemplate

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct-AWQ")

system_msg = "You are a medical data extraction expert."
user_msg = "Here is the medical note: EXAMPLE NOTE"
task = "Extract Reason for Visit."
assistant_resp = "Understood."

# Reference: apply_chat_template
messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg},
    {"role": "assistant", "content": assistant_resp},
    {"role": "user", "content": task},
]
ref = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Our code (simulating build_base_cache + user_assistant)
tmpl = ChatTemplate("qwen2")
base_prompt = (
    tmpl.system_user_assistant(system_msg, user_msg)
    + assistant_resp
    + tmpl.t['turn_end']  # The fix
)
continuation = tmpl.user_assistant(task)
ours = base_prompt + continuation

print("=== Reference ===")
print(repr(ref))
print()
print("=== Ours (fixed) ===")
print(repr(ours))
print()
if ref == ours:
    print("IDENTICAL! Fix is correct.")
else:
    print("STILL DIFFERENT!")
    print(f"  ref len={len(ref)}, ours len={len(ours)}")
    for i in range(min(len(ref), len(ours))):
        if ref[i] != ours[i]:
            print(f"  First diff at pos {i}")
            print(f"  ref:  {repr(ref[max(0,i-20):i+20])}")
            print(f"  ours: {repr(ours[max(0,i-20):i+20])}")
            break

# Also check Llama
print("\n\n=== Llama check ===")
tmpl_llama = ChatTemplate("llama3")
base_llama = (
    tmpl_llama.system_user_assistant(system_msg, user_msg)
    + assistant_resp
    + tmpl_llama.t['turn_end']  # turn_end = <|eot_id|>
)
cont_llama = tmpl_llama.user_assistant(task)
full_llama = base_llama + cont_llama
print(repr(full_llama))
