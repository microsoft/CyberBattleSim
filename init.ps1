# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

#.SYNOPSIS
# Initialize a dev environment by installing all python dependencies on Windows.
# Not supported anymore: use WSL-based Linux instead on Windows.
param($installJupyterExtensions)

# Install pip
$pipversion = $(py -m pip --version)
if ($pipversion) {
    Write-Host "pip already installed"
} else {
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    py.exe get-pip.py
}

# Install virtualenv
py -m pip install --user virtualenv

# Create a virtual environment
py.exe -m venv venv

# Install all pip dependencies in the virtual environment
& venv/Scripts/python.exe -m pip install -e .
& venv/Scripts/python.exe -m pip install -e .[dev]

# Setup pre-commit to check every git commit
& venv/Scripts/pre-commit.exe install -t pre-push
