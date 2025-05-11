#!/bin/bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 253957294717.dkr.ecr.us-east-1.amazonaws.com
TAG=$(openssl rand -hex 12)
docker build -t scaled-rag-qwen:$TAG .
docker tag scaled-rag-qwen:$TAG 253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:qwen-$TAG
docker tag scaled-rag-qwen:latest 253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:qwen-latest
docker push 253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:qwen-$TAG
docker push 253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:qwen-latest
