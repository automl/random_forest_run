#!/bin/bash

set -e
set -x

# Install the environment requirements
sudo apt-get install -y build-essential
python -m pip install --upgrade pip
pip install "numpy<=1.19"
sudo apt-get -qq update
sudo apt-get install -y libboost-all-dev
sudo apt-get install -y swig
sudo gem install coveralls-lcov
sudo apt-get install -y lcov
sudo apt-get install doxygen
pip3 install --user -U pip-tools
