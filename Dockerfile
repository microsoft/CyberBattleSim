# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#FROM openvino/onnxruntime_ep_ubuntu20:latest
# mcr.microsoft.com/azureml/onnxruntime:latest-cuda
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

WORKDIR /root
ADD *.sh ./
ADD *.txt ./
ADD *.yml ./
RUN conda env create -f env.yml
RUN conda init bash
ENV PATH /opt/miniconda/envs/cybersim/bin:$PATH
SHELL ["/bin/bash", "-c"]
RUN activate cybersim
RUN pyright
RUN ./createstubs.sh
COPY . .

# To build the docker image:
#   docker build -t cyberbattle:1.1 .
#
# To run
#   docker run -it --rm cyberbattle:1.1 bash
#
# Pushing to private repository
#   docker login -u spinshot-team-token-writer --password-stdin spinshot.azurecr.io
#   docker tag cyberbattle:1.1 spinshot.azurecr.io/cyberbattle:1.1
#   docker push spinshot.azurecr.io/cyberbattle:1.1
