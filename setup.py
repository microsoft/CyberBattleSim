#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""setup CyberBattle simulator module"""
import os
import setuptools
from typing import List

pwd = os.path.dirname(__file__)


def get_install_requires(requirements_txt) -> List[str]:
    """get the list of requried packages"""
    install_requires = []
    with open(os.path.join(pwd, requirements_txt)) as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):
                install_requires.append(line)
    return install_requires


# main setup kw args
setup_kwargs = {
    'name': 'loonshot-sim',
    'version': '0.1',
    'description': "The simulation and RL code for the S+C loonshot project",
    'author': 'S+C Loonshot Team',
    'author_email': 'scloonshot@Microsoft.com',
    'install_requires': get_install_requires("requirements.txt"),
    'classifiers': [
        'Environment :: Other Environment',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    'zip_safe': True,
    'packages': setuptools.find_packages(exclude=['test_*.py', '*_test.py']),
    'extras_require': {
        'dev': get_install_requires('requirements.dev.txt')
    }
}

if __name__ == '__main__':
    setuptools.setup(**setup_kwargs)
