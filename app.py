from fastapi import FastAPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import asynccontextmanager

device = 'cuda' if torch.cuda.is_available() else 'cpu'
models = {}
tokenizers = {}

def load_model(filepath_model: str, filepath_tokenizer: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(filepath_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(filepath_model)
    return model, tokenizer

@asynccontextmanager
async def lifespan(app):
    print("Loading models and tokenizers...")
    models["smollm-135M"], tokenizers["smollm-135M"] = load_model("models/smollm-135M/", "tokenizers/smollm-135M/")
    models["smollm-360M"], tokenizers["smollm-360M"] = load_model("models/smollm-360M/", "tokenizers/smollm-360M/")
    yield
    print("Removing models and tokenizers...")
    models.clear()
    tokenizers.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/healthcheck")
def healtcheck():
    return {"status" : "ok"}

@app.post("/predict135M")
def predict_135M(messages: list[dict]) -> str:
    model = models["smollm-135M"].to(device)
    tokenizer = tokenizers["smollm-135M"]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    response  = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    _, response = response.split("assistant\n")
    return response


@app.post("/predict360M")
def predict_360M(messages: list[dict]) -> str:
    model = models["smollm-360M"].to(device)
    tokenizer = tokenizers["smollm-360M"]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=50, temperature=0.2, top_p=0.9, do_sample=True)
    response  = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    _, response = response.split("assistant\n")
    return response