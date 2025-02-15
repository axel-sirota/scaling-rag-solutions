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

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from datasets import load_dataset
from gensim.utils import simple_preprocess

# --- Logging Setup ---
def init_logging():
    # Remove any existing handlers.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [PID %(process)d] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,  # Requires Python 3.8+
    )
    for handler in logging.getLogger().handlers:
        handler.flush = sys.stdout.flush

# Initialize logging in the main process.
init_logging()
logging.info("Main process: Logging initialized.")

#######################################
# Worker Process Function
#######################################
def worker_process(gpu_id, input_queue, output_queue):
    init_logging()
    logging.info(f"Worker on GPU {gpu_id}: Starting process.")
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info(f"Worker on GPU {gpu_id}: Using device {device}.")

    # Load models and tokenizers
    MODEL_NAME = "microsoft/DialoGPT-small"
    logging.info(f"Worker on GPU {gpu_id}: Loading generative tokenizer...")
    gen_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # (Optional: set pad_token to eos_token to avoid warnings)
    if gen_tokenizer.pad_token is None:
        gen_tokenizer.pad_token = gen_tokenizer.eos_token

    logging.info(f"Worker on GPU {gpu_id}: Loading generative model...")
    gen_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    gen_model.eval()
    logging.info(f"Worker on GPU {gpu_id}: Generative model loaded successfully.")

    logging.info(f"Worker on GPU {gpu_id}: Loading embedding tokenizer...")
    emb_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    logging.info(f"Worker on GPU {gpu_id}: Loading embedding model...")
    emb_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
    emb_model.eval()
    logging.info(f"Worker on GPU {gpu_id}: Embedding model loaded successfully.")

    # Load dataset passages (using a subset for demonstration)
    logging.info(f"Worker on GPU {gpu_id}: Loading dataset passages...")
    dataset = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
    docs = [doc["passage"] for doc in dataset][:10000]
    logging.info(f"Worker on GPU {gpu_id}: Loaded {len(docs)} passages from dataset.")

    def clean_and_tokenize(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        tokens = simple_preprocess(text)
        return ' '.join(tokens)

    # Build FAISS index
    index_dim = 384  # Should match the embedding output dimension.
    logging.info(f"Worker on GPU {gpu_id}: Building FAISS index (dimension: {index_dim})...")
    faiss_index = faiss.IndexFlatL2(index_dim)
    for idx, doc in enumerate(docs):
        cleaned_doc = clean_and_tokenize(doc)
        tokens = emb_tokenizer(cleaned_doc, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            emb = emb_model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()
        faiss_index.add(emb)
        if (idx + 1) % 1000 == 0:
            logging.debug(f"Worker on GPU {gpu_id}: Processed {idx + 1} passages...")
    logging.info(f"Worker on GPU {gpu_id}: FAISS index built with {faiss_index.ntotal} vectors.")

    # Worker loop: process incoming tasks
    logging.info(f"Worker on GPU {gpu_id}: Entering main loop to process queries.")
    while True:
        task = input_queue.get()  # Wait for a task.
        if task is None:
            logging.info(f"Worker on GPU {gpu_id}: Received shutdown signal.")
            break

        job_id = task["job_id"]
        query_text = task["query"]
        logging.info(f"Worker on GPU {gpu_id}: Received job {job_id} with query: '{query_text}'")

        # 1. Embed the query.
        cleaned_query = clean_and_tokenize(query_text)
        tokens = emb_tokenizer(cleaned_query, return_tensors="pt", truncation=True, max_length=512)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            query_emb = emb_model(**tokens).last_hidden_state.mean(dim=1).cpu().numpy()
        logging.debug(f"Worker on GPU {gpu_id}: Query embedding computed.")

        # 2. Retrieve top-k passages from FAISS.
        k = 5
        distances, indices = faiss_index.search(query_emb, k)
        retrieved_passages = [docs[idx] for idx in indices[0]]
        logging.debug(f"Worker on GPU {gpu_id}: FAISS search complete. Retrieved indices: {indices[0]}")
        context = "\n".join(retrieved_passages)

        # 3. Construct prompt and generate an answer.
        prompt = f"Question: {query_text}\nContext: {context}\nAnswer:"
        # Use the tokenizer call to get both input_ids and attention_mask.
        encoded = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encoded["input_ids"].to(device)
        # If no attention mask is present, create one (ones).
        if "attention_mask" not in encoded:
            attention_mask = torch.ones_like(input_ids).to(device)
        else:
            attention_mask = encoded["attention_mask"].to(device)
        logging.debug(f"Worker on GPU {gpu_id}: Prompt constructed (length {input_ids.size(1)} tokens). Generating answer...")

        with torch.no_grad():
            outputs = gen_model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,
                num_beams=5,
                early_stopping=True
            )
        generated_answer = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info(f"Worker on GPU {gpu_id}: Job {job_id} processed. Answer generated.")

        result = {
            "job_id": job_id,
            "query": query_text,
            "retrieved_passages": retrieved_passages,
            "generated_answer": generated_answer,
        }
        output_queue.put(result)
    logging.info(f"Worker on GPU {gpu_id}: Shutting down.")

#########################################
# Main Process: Worker Management
#########################################
class WorkerClient:
    def __init__(self, gpu_id, input_queue, output_queue, process):
        self.gpu_id = gpu_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.process = process
        self.lock = threading.Lock()

# Create one worker per GPU.
workers = []
worker_index = 0
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    logging.warning("No GPUs detected. Falling back to CPU. Creating one worker.")
    num_gpus = 1

logging.info(f"Main process: Launching {num_gpus} worker(s).")
for gpu_id in range(num_gpus):
    in_q = mp.Queue()
    out_q = mp.Queue()
    p = mp.Process(target=worker_process, args=(gpu_id, in_q, out_q))
    p.start()
    workers.append(WorkerClient(gpu_id, in_q, out_q, p))
logging.info(f"Main process: Launched {len(workers)} worker(s).")

#########################################
# FastAPI Application
#########################################
app = FastAPI()

# Health Check Endpoint
@app.get("/")
def health_check():
    return {"status": "healthy"}


class QueryRequest(BaseModel):
    query: str

@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    global worker_index
    query_text = request.query
    job_id = str(uuid.uuid4())
    logging.info(f"Main process: Received new query with job_id {job_id}: '{query_text}'")

    # Round-robin assignment of worker.
    worker = workers[worker_index % len(workers)]
    worker_index += 1
    logging.debug(f"Main process: Dispatching job {job_id} to worker on GPU {worker.gpu_id}.")

    with worker.lock:
        worker.input_queue.put({"job_id": job_id, "query": query_text})
        logging.debug(f"Main process: Job {job_id} sent to worker on GPU {worker.gpu_id}. Waiting for response...")
        result = worker.output_queue.get()
    logging.info(f"Main process: Job {job_id} completed. Returning result.")
    return result

#########################################
# Shutdown Cleanup
#########################################
def shutdown_workers():
    logging.info("Main process: Shutting down all workers...")
    for worker in workers:
        logging.info(f"Main process: Sending shutdown signal to worker on GPU {worker.gpu_id}...")
        worker.input_queue.put(None)
        worker.process.join()
    logging.info("Main process: All workers have been shut down.")

atexit.register(shutdown_workers)

#########################################
# Run the FastAPI Server
#########################################
if __name__ == "__main__":
    logging.info("Main process: Starting FastAPI server on port 8000.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
