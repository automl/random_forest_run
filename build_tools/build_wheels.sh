# The wheels are built on a centos docker image
# To comply with many-linux support, we employ
# cibuildwheel which build the wheels in a
# Docker image quay.io with minimal support. Following files are required
# for the pyrfr to be compiled on the desired target
pip3 install cmake numpy==1.11.0 scipy==0.17.0
echo 'echo "pyuic5 $@"' > /usr/local/bin/pyuic5
chmod +x /usr/local/bin/pyuic5
yum install -y curl gsl-devel pcre-devel
curl -LO https://downloads.sourceforge.net/swig/swig-4.0.2.tar.gz
tar xzvf swig-4.0.2.tar.gz
cd swig-4.0.2
./configure
make
make install
cd ..
rm -rf swig-4.0.2*
swig -version

# Install the package building dependencies -- one line at
# a time for easy debug -- yum errors out with not much info
# if one package installation failed
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

# Wheel building process will create a package from
# the contents of /project. For debug purposes, show
# the contents of this directory
tree /project
