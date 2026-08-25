#!/bin/bash

set -ex

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

pushd "$(dirname "$0")"

# Ensure the `uv` package manager is available.
if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found, installing it..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Create the virtual environment with a uv-managed Python 3.10.
uv venv --python 3.10 .venv

# shellcheck disable=SC1091
source .venv/bin/activate

python --version

# Install the package together with its runtime and dev dependencies.
uv pip install -e ".[dev]"

# Register a Jupyter kernel named `cybersim` used by the notebooks.
python -m ipykernel install --user --name cybersim --display-name cybersim

if [ ""$GITHUB_ACTION"" == "" ] && [ -d ".git" ]; then
  echo 'running under a git enlistment -> configure pre-commit checks on every `git push` to run pyright and co'
  pre-commit install -t pre-push
fi

./createstubs.sh

popd
