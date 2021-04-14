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

# Also build the distribution
python3 -m pip install twine
python3 setup.py sdist -d ../../dist

# Check whether the source distribution will render correctly
twine check ../../dist/*.tar.gz
