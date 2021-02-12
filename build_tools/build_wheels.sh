#!/bin/bash

set -e
set -x

# Build the package
mkdir build
cd build
cmake .. && make pyrfr_docstrings
cd python_package

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies
cp ../../pyproject.toml .

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
