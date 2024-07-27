from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

# Define the response model
class QueryResponse(BaseModel):
    response: str

app = FastAPI()

# Load the tokenizer
tokenizer_name = "PY007/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Load the model directly without quantization settings
model_path = "PY007/TinyLlama-1.1B-Chat-v0.1"

# Load the model without any quantization configuration
model = AutoModelForCausalLM.from_pretrained(model_path, revision='main', use_safetensors=False)
model.to("cpu")

def generate_data(query, model, tokenizer):
    inputs = tokenizer(query, return_tensors="pt")
    generation_config = {
        "max_new_tokens": 250,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,
        "temperature": 0.7,
        "top_p": 0.9,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True  # Ensure sampling-based generation
    }
    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, **generation_config)
    text_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text_output[len(query):].strip()

@app.get("/query", response_model=QueryResponse)
async def query_model(query: str):
    response = generate_data(query, model, tokenizer)
    return QueryResponse(response=response)

@app.get("/")
async def root():
    return {"message": "Welcome to the TinyLlama API. Use the /query endpoint to interact with the model."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
