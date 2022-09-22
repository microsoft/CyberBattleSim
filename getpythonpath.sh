#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Look for the the required version of python, return its path in $PYTHON
# and version in $PYTHONVER.
#
# Usage: source this file to set the correct default version of Python in the PATH:
#    source getpythonpath.sh
#

PYTHON=`which python`
if [ -z "$PYTHON" ]; then
    PYTHON=`which python3`
fi
if [ -z "$PYTHON" ]; then
    PYTHON=`which python3.9`
fi

if [ -z "$PYTHON" ]; then
    echo "Could not located python interpreter: '$PYTHON'" >&2
    exit -1
fi

PYTHONVER=`$PYTHON --version | cut -d' ' -f2`

if [[ ! "$PYTHONVER" == "3.9."* ]]; then
    echo 'Version ~=3.9 of Python is required' >&2
    exit
else
    echo "Compatible version $PYTHONVER of Python detected at $PYTHON"
fi

PYTHONPATH=$(dirname $PYTHON)

export PATH=$PYTHONPATH:$PATH
