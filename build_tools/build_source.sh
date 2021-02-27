#!/bin/bash

set -e
set -x

# Build the package
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

# Also build the distribution
python -m pip install twine
python setup.py sdist -d ../../dist

# Check whether the source distribution will render correctly
twine check ../../dist/*.tar.gz
