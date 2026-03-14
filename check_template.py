"""Check Qwen chat template vs our implementation."""
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct-AWQ")

system_msg = "You are a medical data extraction expert."
user_msg = "Here is the medical note: EXAMPLE NOTE"
task = "Extract Reason for Visit."
assistant_resp = "Understood."

# What apply_chat_template produces for multi-turn:
messages = [
    {"role": "system", "content": system_msg},
    {"role": "user", "content": user_msg},
    {"role": "assistant", "content": assistant_resp},
    {"role": "user", "content": task},
]
ref = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("=== Reference (apply_chat_template) ===")
print(repr(ref))

# What our code produces:
our_base = (
    "<|im_start|>system\n" + system_msg + "<|im_end|>\n"
    + "<|im_start|>user\n" + user_msg + "<|im_end|>\n"
    + "<|im_start|>assistant\n"
    + assistant_resp
)
our_cont = "<|im_start|>user\n" + task + "<|im_end|>\n" + "<|im_start|>assistant\n"
our_full = our_base + our_cont

print("\n=== Our code output ===")
print(repr(our_full))

# Diff
print("\n=== DIFF ===")
if ref == our_full:
    print("IDENTICAL!")
else:
    print("DIFFERENT!")
    print(f"  ref length: {len(ref)}, ours length: {len(our_full)}")
    for i in range(min(len(ref), len(our_full))):
        if ref[i] != our_full[i]:
            print(f"  First diff at position {i}")
            print(f"  ref context:  {repr(ref[max(0,i-30):i+30])}")
            print(f"  ours context: {repr(our_full[max(0,i-30):i+30])}")
            break
