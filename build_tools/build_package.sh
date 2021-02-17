#!/bin/bash

set -e
set -x

# Traditionally the root directory of a github repo
# contains all python requirements to create a distribution
# This file setups the environment and copy over the required
# files to the build/python_package directory that will contain
# the python distribution directory

# Install the environment
chmod u+x ./build_tools/env.sh
./build_tools/env.sh

# Build the package
mkdir build
cd build
cmake .. && make pyrfr_docstrings
cd python_package

# Copy required files, which will be
# available to a docker image that
# will build wheel files
# This docker image will copy the contents
# of CWD into itself, so any build requirement
# for wheels should be available
# Directory Structure where docker image runs:
# REPO/build/python_package/
cp ../../pyproject.toml .
cp -r ../../build_tools .
# make the test data available for unit test
cp -r ../../test_data_sets .
