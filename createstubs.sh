#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -e

pushd "$(dirname "$0")"

echo "$(tput setaf 2)Creating type stubs$(tput sgr0)"
createstub() {
    local name=$1
    if [ ! -d "typings/$name" ]; then
        pyright --createstub $name
    else
        echo stub $name already created
    fi
}
param=$1
if [[ $param == "--recreate" ]]; then
    echo 'Deleting typing directory'
    rm -Rf typings/
fi

echo 'Creating stubs'

mkdir -p typings/

createstub pandas
createstub plotly
createstub asciichartpy
createstub boolean


echo 'Typing stub generation completed'

popd
