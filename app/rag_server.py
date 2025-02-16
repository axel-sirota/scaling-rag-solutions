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
import time  # NEW: for timing

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

# Track whether each worker (GPU) has finished model loading
WORKER_READY = []

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
    all_docs = [doc["passage"] for doc in dataset][:2000]
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
    """
    Each worker runs in its own process, loads its GPU-specific models,
    then handles inference queries from the input_queue.
    """
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

    logging.info(f"Worker on GPU {gpu_id}: Models loaded successfully. Sending ready signal...")
    # Signal to main that this worker has finished loading
    output_queue.put({"msg_type": "ready", "gpu_id": gpu_id})

    # IMPORTANT: Use the global docs and faiss_index that were built in the main process.
    global docs, faiss_index

    def clean_and_tokenize(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = simple_preprocess(text)
        return " ".join(tokens)

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

# ------------------------
# FastAPI App
# ------------------------
app = FastAPI()

@app.get("/")
def health_check():
    """
    Return 200 only if the FAISS index is built (INDEX_READY). Otherwise 503.
    This does NOT check if all GPU models are loaded – just the index.
    """
    global INDEX_READY
    with index_lock:
        if not INDEX_READY:
            return Response(status_code=503, content="Index not ready")
    return {"status": "healthy"}

@app.get("/models_ready")
def models_ready():
    """
    Returns a dictionary indicating model-loaded status for each GPU.
    Example: { "gpu_0": true, "gpu_1": false, ... }
    """
    global WORKER_READY
    status_dict = {}
    for i, ready in enumerate(WORKER_READY):
        status_dict[f"gpu_{i}"] = ready
    return status_dict

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    """
    Endpoint that accepts a query string and returns:
      - The retrieved top passages
      - The generated answer
    """
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
    start_time = time.time()  # For measuring total load time

    # Initialize worker readiness tracking
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logging.warning("No GPUs detected. Falling back to CPU. Creating one worker.")
        num_gpus = 1

    WORKER_READY = [False] * num_gpus
    logging.info(f"Main process: Detected {num_gpus} GPU(s).")

    # 1) Build the FAISS index ONCE in the main process
    with index_lock:
        docs, faiss_index = build_faiss_index()
        INDEX_READY = True  # Mark index ready

    # 2) Spawn one worker per GPU
    logging.info(f"Main process: Launching {num_gpus} workers...")
    workers = []
    worker_index = 0

    for gpu_id in range(num_gpus):
        in_q = mp.Queue()
        out_q = mp.Queue()
        p = mp.Process(target=worker_process, args=(gpu_id, in_q, out_q))
        p.start()
        workers.append(WorkerClient(gpu_id, in_q, out_q, p))

    # 3) Wait for each worker to signal it has loaded its models
    for w in workers:
        msg = w.output_queue.get()  # Blocking wait for "ready" message
        if msg.get("msg_type") == "ready":
            gpu_id = msg.get("gpu_id")
            WORKER_READY[gpu_id] = True
            logging.info(f"Main process: Worker on GPU {gpu_id} is ready.")

    end_time = time.time()
    total_load_time = end_time - start_time
    logging.info(f"All GPU processes have loaded. Total load time: {total_load_time:.2f} seconds")

    # 4) Start FastAPI server only after all models are loaded
    logging.info("Main process: Starting FastAPI server on port 8000.")
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=600)
