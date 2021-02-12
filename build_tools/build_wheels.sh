#!/bin/bash

set -e
set -x

# Build the package
chmod u+x build_tools/build_package.sh
./build_tools/build_package.sh

# The version of the built dependencies are specified
# in the pyproject.toml file, while the tests are run
# against the most recent version of the dependencies
pwd
sudo apt-get install tree
tree

cd build/python_package

python -m pip install cibuildwheel
docker --version
ls
python -m cibuildwheel --output-dir wheelhouse
python -m cibuildwheel --output-dir ../../wheelhouse
ls ../../
ls ../../wheelhouse
