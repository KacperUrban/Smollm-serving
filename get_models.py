from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.makedirs("tokenizers", exist_ok=True)
os.makedirs("models", exist_ok=True)

checkpoint_s = "HuggingFaceTB/SmolLM2-135M-Instruct"
checkpoint_b = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer_s = AutoTokenizer.from_pretrained(checkpoint_s)
tokenizer_b = AutoTokenizer.from_pretrained(checkpoint_b)

model_s = AutoModelForCausalLM.from_pretrained(checkpoint_s)
model_b = AutoModelForCausalLM.from_pretrained(checkpoint_b)

messages = [{"role" : "user", "content" : "What is the capital of Poland?"}]
input_text_s = tokenizer_s.apply_chat_template(messages, tokenize=False)
input_text_b = tokenizer_b.apply_chat_template(messages, tokenize=False)

print(f"First model - {checkpoint_s}")
print(input_text_s)

inputs_s = tokenizer_s.encode(input_text_s, return_tensors="pt").to(device)
outputs_s = model_s.generate(inputs_s, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer_s.decode(outputs_s[0]))
print("Saving first model ...")
tokenizer_s.save_pretrained("tokenizers/smollm-135M/")
model_s.save_pretrained("models/smollm-135M/")

print(f"Second model - {checkpoint_b}")
print(input_text_b)

inputs_b = tokenizer_b.encode(input_text_b, return_tensors="pt").to(device)
outputs_b = model_b.generate(inputs_b, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer_b.decode(outputs_b[0]))
print("Saving second model ...")
tokenizer_b.save_pretrained("tokenizers/smollm-360M/")
model_b.save_pretrained("models/smollm-360M/")