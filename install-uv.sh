#!/bin/bash

set -ex

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Install the `uv` Python package manager (https://docs.astral.sh/uv/).
curl -LsSf https://astral.sh/uv/install.sh | sh

echo "uv installed. Open a new terminal (or run 'source ~/.local/bin/env') and then run 'bash init.sh'."
