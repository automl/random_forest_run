#!/bin/bash

set -e
set -x

# Install the environment requirements
# order is important so boost is properly recognized by cmake
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential libboost-all-dev swig doxygen git cmake
sudo apt-get install -y swig

# ctest related requirements
sudo apt-get install -y ruby-dev
sudo gem install coveralls-lcov
sudo apt-get install -y lcov

# python package requirements
python3 -m pip install --upgrade pip
pip3 install --user -U pip-tools
