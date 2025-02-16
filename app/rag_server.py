import os
import sys
import re
import time
import uuid
import logging
import atexit
import threading
import multiprocessing as mp

# Disable huggingface tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# If you want to avoid fork issues on Linux, uncomment:
# mp.set_start_method("spawn", force=True)

import torch
import faiss
import uvicorn
import numpy as np

from fastapi import FastAPI, Response
from pydantic import BaseModel
from datasets import load_dataset
from gensim.utils import simple_preprocess
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ------------------------
# Global Config & Variables
# ------------------------
LOG_LEVEL = logging.INFO  # Or logging.DEBUG if you want more verbose
INDEX_READY = False
faiss_index = None
docs = None

index_lock = threading.Lock()

workers = []
worker_index = 0

WORKER_READY = []  # We'll populate once we know the number of GPUs

# ------------------------
# Main Process: Log Setup
# ------------------------
def init_main_logger():
    """
    Prepare the root logger in the main process.
    We'll just attach a single StreamHandler for console output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)

    # Remove any existing handlers to start fresh
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    # Create a stream handler for stdout
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] [PID %(process)d] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

def log_receiver(log_queue):
    """
    Runs in a background thread in the main process.
    Receives logging.LogRecord objects from worker processes and logs them locally.
    """
    root_logger = logging.getLogger()
    while True:
        record = log_queue.get()
        if record is None:
            # This is our shutdown signal
            break
        # Re-emit the record in the main process
        root_logger.handle(record)

class MultiprocessLogHandler(logging.Handler):
    """
    A custom logging handler that sends LogRecords through a multiprocessing.Queue
    to the main process.
    """
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        try:
            # Make a copy of the record just in case
            # Remove any exception info that might not be pickleable
            record.exc_info = None
            record.exc_text = None
            # record.args might be unpickleable sometimes – typically it's fine.

            # Send the record to the queue
            self.log_queue.put(record)
        except Exception:
            pass

def init_worker_logger(log_queue):
    """
    Configure the worker's root logger to send all logs to MultiprocessLogHandler,
    which forwards them to the main process via log_queue.
    """
    worker_logger = logging.getLogger()
    worker_logger.setLevel(LOG_LEVEL)

    # Remove any default or inherited handlers
    for h in worker_logger.handlers[:]:
        worker_logger.removeHandler(h)

    # Add our multiprocess handler
    mp_handler = MultiprocessLogHandler(log_queue)
    worker_logger.addHandler(mp_handler)

# ------------------------
# Pre-download Models in Main
# ------------------------
def preload_models():
    """
    Download/cache all models in the main process so workers can load from disk 
    rather than downloading from the internet (which can cause blocking).
    """
    model_list = [
        "microsoft/DialoGPT-small",             # generative model
        "sentence-transformers/all-MiniLM-L6-v2"  # embedding model
    ]
    for model_name in model_list:
        logging.info(f"Main process: Pre-downloading model/tokenizer '{model_name}'...")
        # For generative
        _ = AutoModelForCausalLM.from_pretrained(model_name)
        _ = AutoTokenizer.from_pretrained(model_name)
        # If it's an embedding model, we also want AutoModel (not AutoModelForCausalLM)
        if "sentence-transformers" in model_name:
            _ = AutoModel.from_pretrained(model_name)
    logging.info("Main process: All required models are now cached locally.")

# ------------------------
# Build FAISS Index
# ------------------------
def build_faiss_index():
    logging.info("Main process: Starting to build FAISS index...")

    # Load dataset passages
    logging.info("Main process: Loading dataset passages...")
    dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
    all_docs = [doc["passage"] for doc in dataset][:2000]
    logging.info(f"Main process: Loaded {len(all_docs)} passages.")

    # For embedding, we only need CPU
    emb_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    emb_model_cpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").cpu()
    emb_model_cpu.eval()
    logging.info("Main process: Loaded embedding model.")

    def clean_and_tokenize(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        tokens = simple_preprocess(text)
        return " ".join(tokens)

    index_dim = 384
    index = faiss.IndexFlatL2(index_dim)

    logging.info(f"Main process: Building FAISS index (dim={index_dim})...")
    for i, passage in enumerate(all_docs):
        cleaned_doc = clean_and_tokenize(passage)
        tokens = emb_tokenizer(cleaned_doc, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            emb = emb_model_cpu(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()
        index.add(emb)
        if (i + 1) % 1000 == 0:
            logging.debug(f"Main process: Indexed {i+1} passages...")

    logging.info(f"Main process: Finished building FAISS index with {index.ntotal} vectors.")
    return all_docs, index

# ------------------------
# Worker Process Function
# ------------------------
def worker_process(gpu_id, input_queue, output_queue, log_queue):
    # 1) Set up logging in the worker to forward everything to main
    init_worker_logger(log_queue)
    logging.info(f"Worker on GPU {gpu_id}: Starting process.")

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.info(f"Worker on GPU {gpu_id}: Using device {device}.")

    # 2) Load generative model + tokenizer
    MODEL_NAME = "microsoft/DialoGPT-small"
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token
    logging.info(f"Worker on GPU {gpu_id}: Loaded generative tokenizer.")
    gen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    gen_model.eval()
    logging.info(f"Worker on GPU {gpu_id}: Loaded generative model.")

    # 3) Load embedding model + tokenizer on GPU
    emb_tokenizer_gpu = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    logging.info(f"Worker on GPU {gpu_id}: Loaded embedding tokenizer.")
    emb_model_gpu = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    emb_model_gpu.eval()
    logging.info(f"Worker on GPU {gpu_id}: Loaded embedding model.")

    logging.info(f"Worker on GPU {gpu_id}: Models loaded successfully. Sending ready signal...")
    output_queue.put({"msg_type": "ready", "gpu_id": gpu_id})

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
        logging.info(f"Worker on GPU {gpu_id}: Cleaned query: '{cleaned_query}'")
        tokens = emb_tokenizer_gpu(cleaned_query, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        logging.info(f"Worker on GPU {gpu_id}: Tokenized query {query_text}, ready to send to embedding model")
        with torch.no_grad():
            query_emb = emb_model_gpu(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()

        # 2. Retrieve top-k passages
        k = 5
        distances, indices = faiss_index.search(query_emb, k)
        retrieved_passages = [docs[idx] for idx in indices[0]]

        # 3. Generate an answer
        context = "\n".join(retrieved_passages)
        logging.info(f"Worker on GPU {gpu_id}: Retrieved passages for query {query_text}, passages are {len(retrieved_passages)}")
        prompt = f"Question: {query_text}\nContext: {context}\nAnswer:"
        encoded = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids)).to(device)
        logging.info(f"Worker on GPU {gpu_id}: Generated prompt for query {query_text}, ready to generate answer")
        with torch.no_grad():
            outputs = gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                num_beams=5,
                early_stopping=True
            )
        generated_answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Worker on GPU {gpu_id}: Generated answer for query {query_text}")

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
# Background Thread to Wait for Worker "ready"
# ------------------------
def watch_workers_loaded(worker_list):
    global WORKER_READY
    for i, w in enumerate(worker_list):
        msg = w.output_queue.get()  # blocking
        if msg.get("msg_type") == "ready":
            gpu_id = msg["gpu_id"]
            WORKER_READY[gpu_id] = True
            logging.info(f"Main process: Worker on GPU {gpu_id} is ready. ({i+1}/{len(worker_list)})")

    logging.info("Main process: All workers have signaled they are loaded.")

# ------------------------
# FastAPI App
# ------------------------
app = FastAPI()

@app.get("/")
def health_check():
    """
    Returns 200 if the index is built; otherwise 503.
    """
    global INDEX_READY
    with index_lock:
        if not INDEX_READY:
            return Response(status_code=503, content="Index not ready")
    return {"status": "healthy"}

@app.get("/models_ready")
def models_ready():
    """
    Returns JSON for each GPU: { "gpu_0": true, "gpu_1": false, ... }
    """
    status = {}
    for gpu_id, is_ready in enumerate(WORKER_READY):
        status[f"gpu_{gpu_id}"] = is_ready
    return status

class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    """
    Round-robin a query to whichever GPU worker is next in line (if ready).
    """
    global worker_index
    query_text = request.query
    job_id = str(uuid.uuid4())
    logging.info(f"Main process: Received new query job_id={job_id}: '{query_text}'")

    if len(workers) == 0:
        return Response(status_code=503, content="No workers available")

    # Round-robin
    worker = workers[worker_index % len(workers)]
    worker_index += 1

    if not WORKER_READY[worker.gpu_id]:
        logging.warning(f"Worker on GPU {worker.gpu_id} not ready yet. Returning 503.")
        return Response(status_code=503, content=f"Worker {worker.gpu_id} not ready")

    # Dispatch the task
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
    # 1) Set up the main logger to console
    init_main_logger()

    # 2) Create a separate queue for cross-process logs
    log_queue = mp.Queue()

    # 3) Start a background thread to consume logs from worker processes
    log_thread = threading.Thread(target=log_receiver, args=(log_queue,), daemon=True)
    log_thread.start()
    logging.info("Main process: Log receiver thread started.")

    # 4) Preload/cache all required models so workers don't block on download
    preload_models()

    # 5) Build the FAISS index in main
    start_time = time.time()
    with index_lock:
        docs, faiss_index = build_faiss_index()
        INDEX_READY = True
    build_time = time.time() - start_time
    logging.info(f"Main process: FAISS index built in {build_time:.2f} seconds")

    # 6) Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logging.warning("No GPUs detected. Falling back to CPU. Creating one worker.")
        num_gpus = 1

    WORKER_READY[:] = [False] * num_gpus
    logging.info(f"Main process: Launching {num_gpus} worker(s)...")

    # 7) Spawn the worker processes
    for gpu_id in range(num_gpus):
        in_q = mp.Queue()
        out_q = mp.Queue()
        p = mp.Process(target=worker_process, args=(gpu_id, in_q, out_q, log_queue))
        p.start()
        workers.append(WorkerClient(gpu_id, in_q, out_q, p))

    logging.info("Main process: All worker processes started.")

    # 8) Wait for each worker to signal "ready"
    watcher_thread = threading.Thread(target=watch_workers_loaded, args=(workers,))
    watcher_thread.daemon = True
    watcher_thread.start()

    # 9) Start the FastAPI/Uvicorn server
    logging.info("Main process: Starting FastAPI server on port 8000.")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=600)
    finally:
        # Send None to log_queue so our log thread exits
        log_queue.put(None)
        log_thread.join()
        logging.info("Main process: Log receiver thread stopped.")
