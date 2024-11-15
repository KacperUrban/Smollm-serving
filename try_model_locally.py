from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

tokenizer_s = AutoTokenizer.from_pretrained("tokenizers/smollm-135M/")
tokenizer_b = AutoTokenizer.from_pretrained("tokenizers/smollm-360M/")

model_s = AutoModelForCausalLM.from_pretrained("models/smollm-135M/").to(device)
model_b = AutoModelForCausalLM.from_pretrained("models/smollm-360M/").to(device)

messages = [{"role" : "user", "content" : "What is the capital of Poland?"}, {"role" : "user", "content" : "What is the capital of France?"}]
input_text_s = tokenizer_s.apply_chat_template(messages, tokenize=False)
input_text_b = tokenizer_b.apply_chat_template(messages, tokenize=False)

print("First model - smollm-135M")
inputs_s = tokenizer_s.encode(input_text_s, return_tensors="pt", padding=True).to(device)
outputs_s = model_s.generate(inputs_s, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer_s.decode(outputs_s[0]))

print("Second model - smollm-360M")
inputs_b = tokenizer_b.encode(input_text_b, return_tensors="pt", padding=True).to(device)
outputs_b = model_b.generate(inputs_b, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
print(tokenizer_b.decode(outputs_b[0]))