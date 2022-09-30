#!/bin/bash

set -ex

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

pushd "$(dirname "$0")"

UBUNTU_VERSION=$(lsb_release -rs)

verlte() {
     [  "$1" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

verlte $UBUNTU_VERSION '20' && OLDER_UBUNTU=1 || OLDER_UBUNTU=0

if [ $OLDER_UBUNTU == 1 ]; then
  echo "Old version of Ubuntu detected ($UBUNTU_VERSION)"
fi

if [ ""$AML_CloudName"" != "" ]; then
  echo "Running on AML machine: skipping venv creation by default"
  CREATE_VENV=0
else
  CREATE_VENV=1
fi

ADD_PYTHON39_APTREPO=1

while getopts "ns" opt; do
  case $opt in
    n)
      echo "Skipping venv creation. Parameters: [$OPTARG]"
      CREATE_VENV=0
      ;;
    s)
      echo "Do not register apt repositories needed to install latest version of Python"
      ADD_PYTHON39_APTREPO=0
      ;;
    \?)
      echo "Syntax: init.sh [-n]" >&2
      echo "   -n    skip creation of virtual environment" >&2
      echo "   -s    skip registration of apt repository needed to to install latest version of Python for older versions of Ubuntu (e.g. 16)" >&2
      exit 1
      ;;
  esac
done


SUDO=''
if (( $EUID != 0 )); then
    SUDO='sudo -E'
fi

if [ ! -z "${VIRTUAL_ENV}" ]; then
    echo 'Running under virtual environment, skipping installation of global packages';
else


    if [ "${ADD_PYTHON39_APTREPO}" == "1" ]; then
        echo 'Adding APT repo ppa:deadsnakes/ppa'
        $SUDO apt install software-properties-common -y
        $SUDO apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub
        $SUDO add-apt-repository ppa:deadsnakes/ppa -y
    fi

    $SUDO apt update

    $SUDO apt-get install curl -y

    # install nodejs 12.0 (required by pyright typechecker)
    curl -sL https://deb.nodesource.com/setup_16.x | $SUDO bash -

    # install all apt-get dependencies
    cat apt-requirements.txt | $SUDO  xargs apt-get install -y

    # $SUDO npm -g upgrade node
    $SUDO npm install -g --unsafe-perm=true --allow-root npm



    python_bin=`which python3.9`


    # Make sure that the desired version of python is used
    # in the rest of the script and when calling pyright to
    # generate stubs
    $SUDO update-alternatives --install /usr/bin/python python $python_bin 2
    $SUDO update-alternatives --install /usr/bin/python3 python3 $python_bin 2
    $SUDO update-alternatives --set python $python_bin
    $SUDO update-alternatives --set python3 $python_bin
    update-alternatives --query python
    export PATH="/usr/bin:${PATH}"

    # install pyright
    ./pyright.sh --version

    # installing orca required to export images with plotly
    ./install-orca.sh

    python --version


    if [ "${CREATE_VENV}" == "1" ]; then
        echo 'Install virtualenv'
        python -m pip install --user virtualenv

        # Create a virtual environment
        python -m venv venv

        source venv/bin/activate
    fi
fi


if ./install-pythonpackages.sh; then
    echo 'Package dependencies successfully installed'
else
    echo 'An error occured when installing package dependencies'
    exit 1
fi

if [ "${CREATE_VENV}" == "1" ]; then
  # Register the `venv`` with jupyter
  python -m ipykernel install --user --name=venv
fi

if [ ""$GITHUB_ACTION"" == "" ] && [ -d ".git" ]; then

  echo 'running under a git enlistment -> configure pre-commit checks on every `git push` to run pyright and co'
  pre-commit install -t pre-push
fi

./createstubs.sh

popd
