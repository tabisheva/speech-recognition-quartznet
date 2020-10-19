#!/bin/bash

docker build --file Dockerfile --tag quartznet-pytorch .
docker run -it --name=quartznet --runtime=nvidia --ipc=host -p 8080:8080 -v ~/:~/ quartznet-pytorch:latest bash