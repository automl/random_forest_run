#!/bin/bash

set -e
set -x

# Build the package
chmod u+x build_tools/build_package.sh
./build_tools/build_package.sh

# Also build the distribution
cd build/python_package
python -m pip install twine
python setup.py sdist -d ../../dist

# Check whether the source distribution will render correctly
twine check ../../dist/*.tar.gz
