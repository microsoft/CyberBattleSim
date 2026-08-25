# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#.SYNOPSIS
# Initialize a dev environment by installing all python dependencies on Windows.
# Not supported anymore: use WSL-based Linux instead on Windows.
param($installJupyterExtensions)

# Install the uv package manager if it is not already available
if (Get-Command uv -ErrorAction SilentlyContinue) {
    Write-Host "uv already installed"
} else {
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

# Create a virtual environment with a uv-managed Python 3.10
uv venv --python 3.10 .venv

# Install the package together with its runtime and dev dependencies
uv pip install -e ".[dev]"

# Register a Jupyter kernel named `cybersim` used by the notebooks
& .venv/Scripts/python.exe -m ipykernel install --user --name cybersim --display-name cybersim

# Setup pre-commit to check every git push
& .venv/Scripts/pre-commit.exe install -t pre-push
