#!/bin/bash
# https://github.com/microsoft/pyright/blob/master/docs/ci-integration.md

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

PATH_TO_PYRIGHT=`which pyright`
PYRIGHT_VERSION=@1.1.271

ARGS=$@

vercomp () {
    if [[ $1 == $2 ]]
    then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields in ver1 with zeros
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++))
    do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++))
    do
        if [[ -z ${ver2[i]} ]]
        then
            # fill empty fields in ver2 with zeros
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]}))
        then
            return 1
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]}))
        then
            return 2
        fi
    done
    return 0
}

# Node version check
echo "Checking node version..."
NODE_VERSION=`node -v | cut -d'v' -f2`
MIN_NODE_VERSION="10.15.2"
vercomp $MIN_NODE_VERSION $NODE_VERSION
# 1 == gt
if [[ $? -eq 1 ]]; then
    echo "Node version ${NODE_VERSION} too old, min expected is ${MIN_NODE_VERSION}, run:"
    echo " npm -g upgrade node"
    exit -1
fi

# Do we need to sudo?
echo "Checking node_modules dir..."
NODE_MODULES=`npm -g root`
SUDO="sudo"
if [ -w "$NODE_MODULES" ]; then
    SUDO="" #nop
fi

# If we can't find pyright, install it.
echo "Checking pyright exists..."
if [ -z "$PATH_TO_PYRIGHT" ]; then
    echo "...installing pyright"
    ${SUDO} npm install -g pyright${PYRIGHT_VERSION}
else
    if [ -z "$PYRIGHT_VERSION" ]; then

        # already installed, upgrade to make sure it's current
        # this avoids a sudo on launch if we're already current
        echo "Checking pyright version..."
        CURRENT=`pyright --version | cut -d' ' -f2`
        REMOTE=`npm info pyright version`
        if [ "$CURRENT" != "$REMOTE" ]; then
            echo "...new version of pyright found, upgrading."
            ${SUDO} npm upgrade -g pyright
        fi
    else
        echo "...restoring pyright version ${PYRIGHT_VERSION}"
        ${SUDO} npm install -g pyright${PYRIGHT_VERSION}
    fi
fi

echo "done."
pyright $ARGS
