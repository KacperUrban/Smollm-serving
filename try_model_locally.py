from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_and_predict(filepath_model: str, filepath_tokenizer: str, messages: list[dict], max_new_tokens: int = 50, 
                     temperature: float = 0.1, top_p: float = 0.9, do_sample: bool = True) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(filepath_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(filepath_model).to(device)

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    outputs = model.generate(inputs, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, do_sample=do_sample)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(response)


if __name__ == '__main__':
    messages = [{"role" : "user", "content" : "What is the OOP paradigm in programming?"}]

    load_and_predict("models/smollm-135M/", "tokenizers/smollm-135M/", messages, 200)
    load_and_predict("models/smollm-360M/", "tokenizers/smollm-360M/", messages, 200)