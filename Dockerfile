# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

FROM mcr.microsoft.com/azureml/onnxruntime:latest-cuda
WORKDIR /root
ADD *.sh ./
ADD *.txt ./
ADD *.py ./
RUN export TERM=dumb && ./init.sh -n
# Override conda python 3.7 install
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 2
ENV PATH="/usr/bin:${PATH}"

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
