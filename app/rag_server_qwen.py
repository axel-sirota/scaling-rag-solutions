import os
import sys
import re
import time
import uuid
import logging
import threading

# Ensure clearer CUDA errors & disable parallel tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import faiss
import uvicorn
import numpy as np

from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from datasets import load_dataset
from gensim.utils import simple_preprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

# ------------------------
# Globals
# ------------------------
LOG_LEVEL = logging.DEBUG
INDEX_READY = False
faiss_index = None
docs = None

index_lock = threading.Lock()

gpu_models = []
gpu_index = 0  # Round-robin across GPUs

# Qwen-3-0.6B supports 4096 tokens context
MODEL_MAX_LENGTH = 4096
NEW_TOKENS = 150
PROMPT_MAX_TOKENS = MODEL_MAX_LENGTH - NEW_TOKENS  # 3946
K_RETRIEVE = 2
MODEL_NAME = "Qwen/Qwen3-0.6B"

# ------------------------
# Logging
# ------------------------
def init_logging():
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] [PID %(process)d] [%(levelname)s] %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    root.addHandler(handler)
    logging.debug("Logging initialized.")

init_logging()

# ------------------------
# Build FAISS
# ------------------------
def build_faiss_index():
    logging.info("Building FAISS index...")
    dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
    passages = [doc["passage"] for doc in dataset][:2000]
    logging.debug(f"Loaded {len(passages)} passages for indexing.")

    # CPU embed model
    emb_name = "sentence-transformers/all-MiniLM-L6-v2"
    emb_tokenizer = AutoTokenizer.from_pretrained(emb_name)
    emb_model_cpu = AutoModel.from_pretrained(emb_name).cpu().eval()

    index = faiss.IndexFlatL2(emb_model_cpu.config.hidden_size)
    logging.info(f"IndexFlatL2 created with dimension {emb_model_cpu.config.hidden_size}.")

    def clean_txt(text: str) -> str:
        t = re.sub(r"\s+", " ", text).lower()
        t = re.sub(r"[^a-zA-Z0-9\s]", "", t)
        return " ".join(simple_preprocess(t))

    for i, p in enumerate(passages):
        cleaned = clean_txt(p)
        toks = emb_tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            emb = emb_model_cpu(**toks).last_hidden_state.mean(dim=1).cpu().numpy()
        index.add(emb)
        if (i+1) % 500 == 0:
            logging.debug(f"Indexed {i+1}/{len(passages)} passages.")
    logging.info(f"FAISS index built: total vectors = {index.ntotal}.")
    return passages, index

# ------------------------
# Load GPU Models
# ------------------------
def load_models_for_gpu(gpu_id: int):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading Qwen model on {device}...")

    # Generative Qwen
    gen_tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if gen_tok.pad_token_id is None:
        gen_tok.pad_token_id = gen_tok.eos_token_id
    gen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype="auto",
    ).eval()

    # Embedding model
    emb_name = "sentence-transformers/all-MiniLM-L6-v2"
    emb_tok = AutoTokenizer.from_pretrained(emb_name)
    emb_model = AutoModel.from_pretrained(emb_name).to(device).eval()

    logging.info(f"Models loaded on {device}.")
    return {"device": device, "gen_tokenizer": gen_tok, "gen_model": gen_model,
            "emb_tokenizer": emb_tok, "emb_model": emb_model}

# Text cleaning for embedding

def clean_text_for_embedding(txt: str) -> str:
    t = re.sub(r"\s+", " ", txt).lower()
    t = re.sub(r"[^a-zA-Z0-9\s]", "", t)
    return " ".join(simple_preprocess(t))

# ------------------------
# FastAPI
# ------------------------
app = FastAPI()

@app.get("/")
def health_check():
    logging.debug("Health check called.")
    with index_lock:
        if not INDEX_READY:
            logging.error("Health check failed: index not ready.")
            raise HTTPException(status_code=503, detail="Index not ready")
    logging.debug("Health check passed.")
    return {"status": "healthy"}

@app.get("/models_ready")
def models_ready():
    logging.debug("Models_ready endpoint called.")
    ready = len(gpu_models) > 0
    logging.info(f"Models ready: {ready}, num_gpus={len(gpu_models)}.")
    return {"num_gpus": len(gpu_models), "gpu_ids": list(range(len(gpu_models)))}

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    global gpu_index
    logging.info("/rag called with query: '%s'", request.query)
    with index_lock:
        if not INDEX_READY:
            logging.error("RAG endpoint error: index not ready.")
            raise HTTPException(status_code=503, detail="Index not ready")

    if not gpu_models:
        logging.error("No GPU models loaded.")
        raise HTTPException(status_code=503, detail="Models not loaded")

    idx = gpu_index % len(gpu_models)
    gpu_index_plus = gpu_index + 1
    logging.debug(f"Selecting GPU index {idx} (next will be {gpu_index_plus}).")
    gpu_index = gpu_index_plus

    model_data = gpu_models[idx]
    device = model_data["device"]
    query_text = request.query
    job_id = str(uuid.uuid4())
    logging.info(f"RAG job {job_id}: using device {device}.")

    # Step 1: embed
    cleaned_query = clean_text_for_embedding(query_text)
    emb_tok = model_data["emb_tokenizer"]
    emb_model = model_data["emb_model"]
    tokens = emb_tok(cleaned_query, return_tensors="pt", truncation=True,
                     max_length=512, padding="longest").to(device)
    with torch.no_grad():
        query_emb = emb_model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()
    logging.debug(f"Query embedded: shape {query_emb.shape}.")

    # Step 2: retrieval
    dists, idxs = faiss_index.search(query_emb, K_RETRIEVE)
    logging.debug(f"Retrieved distances: {dists}, indices: {idxs}.")
    retrieved = [docs[i] for i in idxs[0]]
    logging.info(f"Retrieved {len(retrieved)} passages.")
    context = "\n".join(retrieved)

    # Step 3: build prompt & chat template
    gen_tok = model_data["gen_tokenizer"]
    gen_model = model_data["gen_model"]
    prompt = f"Question: {query_text}\nContext: {context}\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    text = gen_tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=True)
    inputs = gen_tok([text], return_tensors="pt",
                      truncation=True, max_length=PROMPT_MAX_TOKENS).to(device)
    logging.debug(f"Prompt tokens: {inputs.input_ids.shape[1]}.")

    # Step 4: generate
    start_gen = time.time()
    out_ids = gen_model.generate(**inputs, max_new_tokens=NEW_TOKENS,
                                 do_sample=False, early_stopping=True)
    gen_time = time.time() - start_gen
    logging.info(f"Generated sequence in {gen_time:.2f}s; total tokens {out_ids.shape[1]}.")

    # isolate new tokens
    new = out_ids[0, inputs.input_ids.shape[1]:].tolist()
    logging.debug(f"New token count: {len(new)}.")
    try:
        idx_end = len(new) - new[::-1].index(gen_tok.eos_token_id)
    except ValueError:
        idx_end = 0
    thinking = gen_tok.decode(new[:idx_end], skip_special_tokens=True).strip()
    content = gen_tok.decode(new[idx_end:], skip_special_tokens=True).strip()
    logging.info(f"RAG job {job_id} answer: {content}")

    return {"answer": content}

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    logging.info("Server starting: building index...")
    start_idx = time.time()
    with index_lock:
        docs, faiss_index = build_faiss_index()
        INDEX_READY = True
    logging.info(f"Index built in {time.time()-start_idx:.2f}s.")

    num_gpus = torch.cuda.device_count() or 1
    logging.info(f"Detected GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu_models.append(load_models_for_gpu(i))
    logging.info(f"Loaded {len(gpu_models)} model(s). Starting server.")
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
