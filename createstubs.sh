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
createstub progressbar
createstub pytest
createstub setuptools
createstub ordered_set
createstub asciichartpy
createstub networkx
createstub boolean
createstub IPython


if [ ! -d "typings/gym" ]; then
    pyright --createstub gym
    # Patch gym stubs
    echo '    spaces = ...' >> typings/gym/spaces/dict.pyi
    echo '    nvec = ...' >> typings/gym/spaces/space.pyi
    echo '    spaces = ...' >> typings/gym/spaces/space.pyi
    echo '    spaces = ...' >> typings/gym/spaces/tuple.pyi
    echo '    n = ...' >> typings/gym/spaces/multi_binary.pyi
else
    echo stub gym already created
fi


echo 'Typing stub generation completed'

popd
