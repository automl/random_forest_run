#!/bin/bash

set -e
set -x

# Build the package
./build_package.sh

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies
cp ../../pyproject.toml .

# Also build the distribution
python -m pip install twine
python setup.py sdist

# Check whether the source distribution will render correctly
twine check dist/*.tar.gz
