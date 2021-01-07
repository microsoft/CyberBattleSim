#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -e

. ./getpythonpath.sh

# Install python packages
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install wheel
$PYTHON -m pip install -e .
$PYTHON -m pip install -e .[dev]
