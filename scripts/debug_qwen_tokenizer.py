
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 1. Create a prompt using chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

print(f"Prompt Text:\n{prompt_text!r}")

# 2. Tokenize the text
tokens = tokenizer(prompt_text, return_tensors="pt")
input_ids = tokens.input_ids[0]

print(f"\nToken IDs: {input_ids}")

# 3. Decode back
decoded = tokenizer.decode(input_ids)
print(f"\nDecoded:\n{decoded!r}")

# 4. Check if special tokens are tokenized as single tokens
# <|im_start|> ID is 151644
# <|im_end|> ID is 151645
print(f"\nChecking for special token IDs...")
if 151644 in input_ids:
    print("Found <|im_start|> (151644)")
else:
    print("MISSING <|im_start|> - Tokenizer split it!")

if 151645 in input_ids:
    print("Found <|im_end|> (151645)")
else:
    print("MISSING <|im_end|> - Tokenizer split it!")

# 5. Check what happens if we don't use chat template but raw strings
raw_text = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
raw_tokens = tokenizer(raw_text, return_tensors="pt").input_ids[0]
print(f"\nRaw Text Token IDs: {raw_tokens}")
if 151644 in raw_tokens:
    print("Found <|im_start|> in raw text")
else:
    print("MISSING <|im_start|> in raw text")

