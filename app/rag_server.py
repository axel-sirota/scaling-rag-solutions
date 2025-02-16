import os
import os
import sys
import re
import time
import uuid
import logging
import threading

# Ensure clearer CUDA errors & disable parallel tokenizer warnings
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import faiss
import uvicorn
import numpy as np

from fastapi import FastAPI, Response
from pydantic import BaseModel
from datasets import load_dataset
from gensim.utils import simple_preprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# ------------------------
# Globals
# ------------------------
LOG_LEVEL = logging.INFO
INDEX_READY = False
faiss_index = None
docs = None

index_lock = threading.Lock()

gpu_models = []
gpu_index = 0  # Round-robin across GPUs

MODEL_MAX_LENGTH = 1024       # DialoGPT context window (small model)
NEW_TOKENS = 150              # We want 150 tokens for the generation tail
PROMPT_MAX_TOKENS = MODEL_MAX_LENGTH - NEW_TOKENS  # 874
K_RETRIEVE = 2                # We'll retrieve top-2 passages from FAISS

# ------------------------
# Logging
# ------------------------
def init_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] [PID %(process)d] [%(levelname)s] %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

init_logging()
logging.info("Main: Logging initialized.")

# ------------------------
# Build FAISS
# ------------------------
def build_faiss_index():
    logging.info("Main: Starting to build FAISS index...")

    # Load dataset passages
    logging.info("Main: Loading dataset passages...")
    dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
    all_docs = [doc["passage"] for doc in dataset][:2000]
    logging.info(f"Main: Loaded {len(all_docs)} passages.")

    # CPU-based embedding
    emb_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emb_model_cpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").cpu()
    emb_model_cpu.eval()

    def clean_and_tokenize(txt: str) -> str:
        txt = re.sub(r"\s+", " ", txt)
        txt = txt.lower()
        txt = re.sub(r"[^a-zA-Z0-9\s]", "", txt)
        return " ".join(simple_preprocess(txt))

    index_dim = 384
    index = faiss.IndexFlatL2(index_dim)

    logging.info(f"Main: Building FAISS index (dim={index_dim})...")
    for i, passage in enumerate(all_docs):
        cleaned = clean_and_tokenize(passage)
        tokens = emb_tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            emb = emb_model_cpu(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()
        index.add(emb)
        if (i + 1) % 1000 == 0:
            logging.debug(f"Main: Indexed {i+1} passages...")

    logging.info(f"Main: Finished building FAISS index with {index.ntotal} vectors.")
    return all_docs, index

# ------------------------
# Load GPU Models
# ------------------------
def load_models_for_gpu(gpu_id: int):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Main: Loading models for GPU {gpu_id} on device {device}...")

    # Generative model
    MODEL_NAME = "microsoft/DialoGPT-small"
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    gen_model.eval()

    # Embedding model
    EMB_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    emb_tokenizer = AutoTokenizer.from_pretrained(EMB_NAME)
    emb_model = AutoModel.from_pretrained(EMB_NAME).to(device)
    emb_model.eval()

    logging.info(f"Main: GPU {gpu_id} models loaded successfully.")
    return {
        "device": device,
        "gen_tokenizer": gen_tokenizer,
        "gen_model": gen_model,
        "emb_tokenizer": emb_tokenizer,
        "emb_model": emb_model,
    }

def clean_text_for_embedding(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt)
    txt = txt.lower()
    txt = re.sub(r"[^a-zA-Z0-9\s]", "", txt)
    return " ".join(simple_preprocess(txt))

# ------------------------
# FastAPI
# ------------------------
app = FastAPI()

@app.get("/")
def health_check():
    global INDEX_READY
    with index_lock:
        if not INDEX_READY:
            return Response(status_code=503, content="Index not ready")
    return {"status": "healthy"}

@app.get("/models_ready")
def models_ready():
    return {
        "num_gpus": len(gpu_models),
        "gpu_ids": list(range(len(gpu_models))),
    }

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    """
    Single-process RAG:
      1) Round-robin pick GPU model.
      2) Embed the query.
      3) Retrieve top-2 from FAISS (K_RETRIEVE=2).
      4) Build prompt. If prompt would exceed 874 tokens, forcibly trim it.
      5) Generate up to NEW_TOKENS=150.
      6) Return only the last NEW_TOKENS tokens from generation.
    """
    global gpu_index
    if len(gpu_models) == 0:
        return Response(status_code=503, content="No GPUs loaded.")

    idx = gpu_index % len(gpu_models)
    gpu_index += 1
    model_data = gpu_models[idx]

    device = model_data["device"]
    query_text = request.query
    job_id = str(uuid.uuid4())
    logging.info(f"Main: RAG job={job_id}, query='{query_text}' -> GPU idx={idx}, device={device}")

    # 1) Embed query
    cleaned_query = clean_text_for_embedding(query_text)
    emb_tok = model_data["emb_tokenizer"]
    emb_model = model_data["emb_model"]

    tokens = emb_tok(
        cleaned_query,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding="longest"
    ).to(device)

    with torch.no_grad():
        query_emb = emb_model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()

    # 2) Retrieve top-2
    dists, idxs = faiss_index.search(query_emb, K_RETRIEVE)
    retrieved_passages = [docs[i] for i in idxs[0]]

    # 3) Build prompt
    context = "\n".join(retrieved_passages)
    prompt = f"Question: {query_text}\nContext: {context}\nAnswer:"
    gen_tok = model_data["gen_tokenizer"]
    gen_model_ref = model_data["gen_model"]

    # We'll forcibly ensure the prompt is at most PROMPT_MAX_TOKENS
    enc = gen_tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=PROMPT_MAX_TOKENS,
        padding="longest"
    )
    # If no attention_mask, create one
    if "attention_mask" not in enc:
        enc["attention_mask"] = torch.ones_like(enc["input_ids"])

    # Move to GPU
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    logging.info(f"Main: Prompt tokens={prompt_len}, leaving {MODEL_MAX_LENGTH - prompt_len} for generation.")

    # 4) Generate
    with torch.no_grad():
        out_ids = gen_model_ref.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=NEW_TOKENS,
            num_beams=1,
            do_sample=False,
            early_stopping=True,
        )
    total_len = out_ids.shape[1]
    generated_len = total_len - prompt_len
    logging.info(f"Main: Generated {generated_len} new tokens (out of {NEW_TOKENS} possible).")

    if generated_len <= 0:
        # No new tokens generated
        final_tokens = out_ids[0, -1:]
    else:
        new_tokens = out_ids[0, prompt_len:]
        # If it somehow generated more than 150, keep last 150
        if new_tokens.shape[0] > NEW_TOKENS:
            new_tokens = new_tokens[-NEW_TOKENS:]
        final_tokens = new_tokens

    final_answer = gen_tok.decode(out_ids[0], skip_special_tokens=True)

    logging.info(f"Main: RAG job={job_id} completed. Returning last {NEW_TOKENS} tokens only.")
    return {"answer": final_answer}

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    # 1) Build the FAISS index
    start_time = time.time()
    with index_lock:
        docs, faiss_index = build_faiss_index()
        INDEX_READY = True
    build_time = time.time() - start_time
    logging.info(f"Main: FAISS index built in {build_time:.2f} seconds.")

    # 2) GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logging.warning("No GPUs found, fallback to CPU.")
        num_gpus = 1

    # 3) Load model sets
    for g in range(num_gpus):
        data = load_models_for_gpu(g)
        gpu_models.append(data)
    logging.info(f"Main: Loaded {len(gpu_models)} GPU model sets.")

    # 4) Start single-process Uvicorn
    logging.info("Main: Starting server on port 8000.")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=1
    )