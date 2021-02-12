#!/bin/bash

set -e
set -x

# Build the package
chmod u+x build_tools/build_package.sh
./build_tools/build_package.sh

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies
# For debug print CWD contents
pwd
sudo apt-get install tree
tree

# Also build the distribution
cd build/python_package
python -m pip install twine
python setup.py sdist -d ../../dist

# Check whether the source distribution will render correctly
twine check ../../dist/*.tar.gz
