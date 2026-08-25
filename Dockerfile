# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#FROM openvino/onnxruntime_ep_ubuntu20:latest
# mcr.microsoft.com/azureml/onnxruntime:latest-cuda
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

# Install the `uv` Python package manager.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /root
SHELL ["/bin/bash", "-c"]

# Create a self-contained virtual environment with a uv-managed Python 3.10.
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv venv --python 3.10 "$VIRTUAL_ENV"

# Install python dependencies first for better layer caching.
ADD requirements.txt requirements.dev.txt ./
RUN uv pip install -r requirements.txt -r requirements.dev.txt

# Copy the rest of the sources and install the package itself.
COPY . .
RUN uv pip install --no-deps -e .

# Generate type stubs used for development. Full pyright type-checking is run
# separately by CI (see .github/workflows/ci.yml), so we don't repeat the
# (slow, cold-cache) full-codebase analysis during the image build.
RUN ./createstubs.sh

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
