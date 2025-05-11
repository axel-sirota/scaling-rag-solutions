#!/bin/bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 253957294717.dkr.ecr.us-east-1.amazonaws.com
TAG=$(openssl rand -hex 12)
docker build -t scaled-rag:$TAG .
docker tag scaled-rag:$TAG scaled-rag:latest
docker tag scaled-rag:$TAG 253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:$TAG
docker tag scaled-rag:latest 253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:latest
docker push 253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:$TAG
docker push 253957294717.dkr.ecr.us-east-1.amazonaws.com/scaled-rag-repo:latest
