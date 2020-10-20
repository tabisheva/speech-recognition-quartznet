#!/bin/bash

docker build --file Dockerfile --tag quartznet-pytorch .
docker run -it --gpus all --ipc=host -p 8080:8080 -v /home/$USER/ASR/speech-recognition-quartznet/:/home/$USER quartznet-pytorch:latest bash
