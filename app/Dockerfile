FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

# Install SWIG and any other necessary build dependencies.
RUN apt-get update && apt-get install -y swig

# Copy requirements and upgrade pip.
COPY requirements.txt /app
RUN pip install --upgrade pip

# Install system dependencies required for building SWIG and other packages.
RUN apt-get update && apt-get install -y \
    build-essential \
    libpcre3-dev \
    automake \
    libtool \
    wget \
    python3-dev

RUN pip install torch  --index-url https://download.pytorch.org/whl/cu126
# Install Python dependencies.
RUN pip install -r requirements.txt

RUN pip install faiss-cpu
# Copy the rest of your application.
COPY . /app

EXPOSE 8000
CMD ["python", "rag_server.py"]
