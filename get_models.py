from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

os.makedirs("tokenizers", exist_ok=True)
os.makedirs("models", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


def download_and_save_model(
    checkpoint: str, messages: list[dict], filepath_model: str, filepath_tokenizer: str
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True
    )
    response = tokenizer.decode(
        outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    print(response)

    print("Saving model and tokenizer...")
    model.save_pretrained(filepath_model)
    tokenizer.save_pretrained(filepath_tokenizer)


if __name__ == "__main__":
    messages = [{"role": "user", "content": "What is the capital of Poland?"}]

    download_and_save_model(
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        messages,
        "models/smollm-135M/",
        "tokenizers/smollm-135M/",
    )
    download_and_save_model(
        "HuggingFaceTB/SmolLM2-360M-Instruct",
        messages,
        "models/smollm-360M/",
        "models/smollm-360M/",
    )
