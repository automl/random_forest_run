pip3 install cmake numpy==1.11.0 scipy==0.17.0
echo 'echo "pyuic5 $@"' > /usr/local/bin/pyuic5
chmod +x /usr/local/bin/pyuic5
yum install -y curl gsl-devel pcre-devel
curl -LO http://prdownloads.sourceforge.net/swig/swig-3.0.12.tar.gz
tar xzvf swig-3.0.12.tar.gz
cd swig-3.0.12
./configure
make
make install
cd ..
rm -rf swig-3.0.12*
curl -LO https://dl.bintray.com/boostorg/release/1.69.0/source/boost_1_69_0.tar.gz
tar xzvf boost_1_69_0.tar.gz
cd boost_1_69_0
./bootstrap.sh
./b2 --without-atomic --without-chrono --without-container --without-context
 --without-contract --without-coroutine --without-date_time --without-exception
 --without-fiber --without-graph --without-graph_parallel --without-iostreams
 --without-locale --without-log --without-math --without-mpi --without-python
 --without-random --without-regex --without-stacktrace --without-thread
 --without-timer --without-type_erasure --without-wave
 install
cd ..
rm -rf boost_1_69_0*
swig -version
