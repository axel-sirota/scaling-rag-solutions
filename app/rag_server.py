import os
import re
import uuid
import sys
import torch
import faiss
import uvicorn
import numpy as np
import multiprocessing as mp
import threading
import atexit
import logging
import re

from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
from gensim.utils import simple_preprocess

# ------------------------
# Global Variables
# ------------------------
INDEX_READY = False
faiss_index = None
docs = None

# For concurrency / safety if needed
index_lock = threading.Lock()

# ------------------------
# Logging Setup
# ------------------------
def init_logging():
    # Remove any existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [PID %(process)d] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # Python 3.8+
    )
    for handler in logging.getLogger().handlers:
        handler.flush = sys.stdout.flush

init_logging()
logging.info("Main process: Logging initialized.")

# ------------------------
# Build FAISS Index (One Time in Main)
# ------------------------
def build_faiss_index():
    """
    Builds the FAISS index (and any text preprocessing) one time in the main process.
    Returns (docs, faiss_index).
    """
    logging.info("Main process: Starting to build FAISS index...")

    # Load dataset passages
    logging.info("Main process: Loading dataset passages...")
    dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
    all_docs = [doc["passage"] for doc in dataset][:10000]
    logging.info(f"Main process: Loaded {len(all_docs)} passages.")

    # For embedding, we only need a tokenizer + model on CPU here
    emb_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emb_model_cpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").cpu()
    emb_model_cpu.eval()

    def clean_and_tokenize(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = simple_preprocess(text)
        return " ".join(tokens)

    index_dim = 384  # Matches embedding dimension from 'all-MiniLM-L6-v2'
    index = faiss.IndexFlatL2(index_dim)

    logging.info(f"Main process: Building FAISS index (dim={index_dim})...")
    for i, passage in enumerate(all_docs):
        cleaned_doc = clean_and_tokenize(passage)
        tokens = emb_tokenizer(cleaned_doc, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            # Inference on CPU
            emb = emb_model_cpu(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()
        index.add(emb)
        if (i + 1) % 1000 == 0:
            logging.debug(f"Main process: Indexed {i+1} passages...")

    logging.info(f"Main process: Finished building FAISS index with {index.ntotal} vectors.")
    return all_docs, index

# ------------------------
# Worker Process Function
# ------------------------
def worker_process(gpu_id, input_queue, output_queue):
    init_logging()
    logging.info(f"Worker on GPU {gpu_id}: Starting process.")

    # Use the assigned GPU if available
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Worker on GPU {gpu_id}: Using device {device}.")

    # Load generative model + tokenizer
    MODEL_NAME = "microsoft/DialoGPT-small"
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    gen_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(device)
    gen_model.eval()

    # Load embedding model + tokenizer (on GPU)
    emb_tokenizer_gpu = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emb_model_gpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    emb_model_gpu.eval()

    logging.info(f"Worker on GPU {gpu_id}: Models loaded successfully.")

    # IMPORTANT: Use the global docs and faiss_index that were built in the main process.
    # Because we're in a separate process, this only works cleanly on Linux with 'fork'.
    global docs, faiss_index

    def clean_and_tokenize(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = simple_preprocess(text)
        return " ".join(tokens)

    # Now we enter the main loop for handling queries
    logging.info(f"Worker on GPU {gpu_id}: Ready to process queries.")
    while True:
        task = input_queue.get()
        if task is None:
            logging.info(f"Worker on GPU {gpu_id}: Received shutdown signal.")
            break

        job_id = task["job_id"]
        query_text = task["query"]
        logging.info(f"Worker on GPU {gpu_id}: Received job {job_id} with query: '{query_text}'")

        # 1. Embed the query
        cleaned_query = clean_and_tokenize(query_text)
        tokens = emb_tokenizer_gpu(cleaned_query, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            query_emb = emb_model_gpu(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()

        # 2. Retrieve top-k passages from FAISS
        k = 5
        distances, indices = faiss_index.search(query_emb, k)
        retrieved_passages = [docs[idx] for idx in indices[0]]

        # 3. Generate an answer
        context = "\n".join(retrieved_passages)
        prompt = f"Question: {query_text}\nContext: {context}\nAnswer:"
        encoded = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)

        with torch.no_grad():
            outputs = gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                num_beams=5,
                early_stopping=True
            )
        generated_answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

        result = {
            "job_id": job_id,
            "query": query_text,
            "retrieved_passages": retrieved_passages,
            "generated_answer": generated_answer,
        }
        output_queue.put(result)

    logging.info(f"Worker on GPU {gpu_id}: Exiting process.")

# ------------------------
# WorkerClient Class
# ------------------------
class WorkerClient:
    def __init__(self, gpu_id, input_queue, output_queue, process):
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.process = process
        self.lock = threading.Lock()

# List of worker clients
workers = []
worker_index = 0
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    logging.warning("No GPUs detected. Falling back to CPU. Creating one worker.")
    num_gpus = 1

logging.info(f"Main process: Detected {num_gpus} GPU(s).")

# ------------------------
# FastAPI App
# ------------------------
app = FastAPI()

@app.get("/")
def health_check():
    """Return 200 only if the index is built (INDEX_READY). Otherwise 503."""
    global INDEX_READY
    with index_lock:
        if not INDEX_READY:
            return Response(status_code=503, content="Index not ready")
    return {"status": "healthy"}

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    global worker_index
    query_text = request.query
    job_id = str(uuid.uuid4())
    logging.info(f"Main process: Received new query with job_id {job_id}: '{query_text}'")

    # Simple round-robin assignment
    worker = workers[worker_index % len(workers)]
    worker_index += 1

    logging.debug(f"Main process: Dispatching job {job_id} to worker on GPU {worker.gpu_id}.")

    with worker.lock:
        worker.input_queue.put({"job_id": job_id, "query": query_text})
        result = worker.output_queue.get()

    logging.info(f"Main process: Job {job_id} completed. Returning result.")
    return result

# ------------------------
# Shutdown Hook
# ------------------------
def shutdown_workers():
    logging.info("Main process: Shutting down all workers...")
    for w in workers:
        logging.info(f"Main process: Sending shutdown signal to worker on GPU {w.gpu_id}...")
        w.input_queue.put(None)
        w.process.join()
    logging.info("Main process: All workers shut down.")

atexit.register(shutdown_workers)

# ------------------------
# Main Entry Point
# ------------------------
if __name__ == "__main__":
    # 1) Build the FAISS index ONCE in the main process
    with index_lock:
        docs, faiss_index = build_faiss_index()
        # Mark index ready
        INDEX_READY = True

    # 2) Spawn one worker per GPU
    logging.info(f"Main process: Launching {num_gpus} workers...")
    for gpu_id in range(num_gpus):
        in_q = mp.Queue()
        out_q = mp.Queue()
        p = mp.Process(target=worker_process, args=(gpu_id, in_q, out_q))
        p.start()
        workers.append(WorkerClient(gpu_id, in_q, out_q, p))

    logging.info("Main process: All workers launched. Starting FastAPI server on port 8000.")
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=300)
