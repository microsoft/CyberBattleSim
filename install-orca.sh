#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

echo 'Installing the Plotly orca dependency for plotly figure export'

SUDO=''
if (( $EUID != 0 )); then
    SUDO='sudo -E'
fi


xargs -a apt-requirements-orca.txt $SUDO apt-get install -y

$SUDO npm install -g --unsafe-perm=true --allow-root electron@6.1.4 orca
