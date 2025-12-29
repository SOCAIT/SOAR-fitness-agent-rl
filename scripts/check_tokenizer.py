
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen2.5-14B-Instruct"

print(f"Loading tokenizer for {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(f"Pad token: {tokenizer.pad_token}")
print(f"Pad token ID: {tokenizer.pad_token_id}")
print(f"EOS token: {tokenizer.eos_token}")
print(f"EOS token ID: {tokenizer.eos_token_id}")

if tokenizer.pad_token is None:
    print("WARNING: Pad token is None!")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# Check chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(f"\nFormatted prompt:\n{formatted}")

# Check tokenization
tokens = tokenizer(formatted, return_tensors="pt")
print(f"\nTokenized shape: {tokens.input_ids.shape}")
print(f"First 10 tokens: {tokens.input_ids[0][:10]}")

