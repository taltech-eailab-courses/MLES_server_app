import time
import torch
from llama_cpp import Llama
from pydantic import BaseModel

llm_prompt_title="TinyLlama Text Generation"
llm_prompt_description="This endpoint uses the TinyLlama model to generate text based on the provided prompt."

class LLAMARequest(BaseModel):
    prompt: str
    max_tokens: int = 100

# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
llm = Llama.from_pretrained(
	repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
	filename="tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    n_ctx=2048,              # Context window size
    seed=42,                # Reproducibility
    verbose=False,
    n_threads=6,          # Adjust based on your CPU
    n_gpu_layers=20       # Set > 0 if using GPU
)

def generate_text(prompt: str, max_tokens: int = 100) -> str:
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
        stop=["Q:", "\n\n"]
    )
    return {"data": output["choices"][0]["text"]}