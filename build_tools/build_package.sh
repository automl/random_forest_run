#!/bin/bash

set -e
set -x

# Traditionally the root directory of a github repo
# contains all python requirements to create a distribution
# This file setups the environment and copy over the required
# files to the build/python_package directory that will contain
# the python distribution directory
sudo apt-get install -y build-essential
python -m pip install --upgrade pip
pip install "numpy<=1.19"
sudo apt-get -qq update
sudo apt-get install -y libboost-all-dev
sudo apt-get remove swig
sudo apt-get -y install swig3.0
sudo ln -s /usr/bin/swig3.0 /usr/bin/swig
sudo gem install coveralls-lcov
sudo apt-get install -y lcov
sudo apt-get install doxygen
pip3 install --user -U pip-tools

# Build the package
mkdir build
cd build
cmake .. && make pyrfr_docstrings
cd python_package

# Copy required files, which will be
# available to a docker image that
# will build wheel files
cp ../../pyproject.toml .
cp -r ../../build_tools .
