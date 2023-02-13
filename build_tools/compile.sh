#!/bin/bash
# The wheels are built in a docker image by `cibuildwheel` seen in release.yml.
# This docker image comes from quay.io with minimal support. It is CentOS

# This scripts install dependancies for compiling the CPP code before finally
# compiling it

set -e  # Immediatly exit on any error
set -x  # Print each command before running

# Build dependancies
pip3 install cmake numpy
pip3 install scipy

# Not sure why this is needed
echo 'echo "pyuic5 $@"' > /usr/local/bin/pyuic5
chmod +x /usr/local/bin/pyuic5

# More build dependancies
yum install -y curl gsl-devel pcre-devel

# Install SWIG
curl -LO https://downloads.sourceforge.net/swig/swig-4.0.2.tar.gz
tar xzvf swig-4.0.2.tar.gz
cd swig-4.0.2
./configure
make
make install
cd ..
rm -rf swig-4.0.2*

# Debug to make sure it's installed
swig -version

# Install the package building dependencies
# -- one line at a time for easy debug --
# yum errors out with not much info if one package installation failed
yum -y install boost
yum -y install boost-thread
yum -y install boost-devel
yum -y install doxygen
yum -y install openssl-devel
yum -y install cmake
yum -y install tree
yum -y install rsync
cmake --version

# After installing the dependencies build the python package
mkdir build
cd build
cmake  .. && make pyrfr_docstrings

# Copy the files for testing
cp -r ../test_data_sets python_package

# Copy from /project/build to /project
rsync -a --delete --exclude '*build*' python_package/ ../

# Wheel building process will create a package from the contents of /project.
# For debug purposes, show the contents of this directory
tree /project
