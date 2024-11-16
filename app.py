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


app.get("healthcheck/")
def healtcheck():
    return {"status" : "ok"}