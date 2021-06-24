#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -e

. ./getpythonpath.sh

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

createstub pandas
createstub plotly
createstub progressbar
createstub pytest
createstub setuptools
createstub ordered_set
createstub asciichartpy

if [ ! -d "typings/gym" ]; then
    pyright --createstub gym
    # Patch gym stubs
    echo '    spaces = ...' >> typings/gym/spaces/dict.pyi
    echo '    nvec = ...' >> typings/gym/spaces/space.pyi
else
    echo stub gym already created
fi

if [ ! -d "typings/IPython" ]; then
    pyright --createstub IPython.core.display
else
    echo stub 'IPython' already created
fi

if [ ! -d "boolean" ]; then
    pyright --createstub boolean
    sed -i '/class BooleanAlgebra(object):/a\    TRUE = ...\n    FALSE = ...' typings/boolean/boolean.pyi
else
    echo stub 'boolean' already created
fi

echo 'Typing stub generation completed'

# Stubs that needed manual patching and that
# were instead checked-in in git
#   pyright --createstub boolean
#   pyright --createstub gym

popd
