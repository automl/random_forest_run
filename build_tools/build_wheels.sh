#!/bin/bash

set -e
set -x

# Build the package
chmod u+x build_tools/build_package.sh
./build_tools/build_package.sh

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies

# Print the directory structure for debug
sudo apt-get install tree
tree

cd build/python_package

# Build many linux wheels using cibuildwheel
# This library will use a docker image from
# quay.io with minimal support. It also handles
# wheel repair (to make sure all needed collaterals
# are included in the wheel)
python -m pip install cibuildwheel
docker --version
python -m cibuildwheel --output-dir ../../wheelhouse
