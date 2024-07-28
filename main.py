from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from transformers import logging as transformers_logging
import warnings

# Set the logging level for transformers
transformers_logging.set_verbosity_error()

# Suppress specific warning messages
warnings.filterwarnings("ignore", message="Unused kwargs:.*")
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used when initializing.*")

# Define the response model
class QueryResponse(BaseModel):
    response: str

app = FastAPI()

# Load the model and tokenizer
tokenizer_name = "PY007/TinyLlama-1.1B-Chat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
tokenizer.save_pretrained("./tokenizer")

model = AutoModelForCausalLM.from_pretrained(tokenizer_name)
model.save_pretrained("./model")

def generate_data(query, model, tokenizer):
    inputs = tokenizer(query, return_tensors="pt")
    generation_config = {
        "max_length": 250,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.2,
        "temperature": 0.7,
        "top_p": 0.9,
        "eos_token_id": tokenizer.eos_token_id
    }
    outputs = model.generate(**inputs, **generation_config)
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
