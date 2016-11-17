#RFR
A extensible C++ library for random forests with Python bindings.

## Requirements

For the C++ library itself, you need no additional libaries, only a C++11 capable compiler.
Technically, you need Boost if you want to compile the unit tests.
The development is done using GCC 4.8 and 6.2.
You probably have to set CMAKE\_CXX\_FLAGS to -std=c++11 when using older compilers.

```
CMAKE
DOXYGEN (if you want docstrings, which you probably do)
SWIG
```


## Installing the Python Bindings
Checkout the repo (and the refactor branch), create a build directory, and build them using the following commands:
```
git checkout git@bitbucket.org:aadfreiburg/random_forest_run.git
cd random_forest_run
git checkout refactor
mkdir build
cd build
cmake ..
make
python setup.py install --user
```
After the installation finishes (hopefully) sucessfully, you can use the library with the pyrfr module.

The above commands work for Python 3.4 and 3.5 on Gentoo and ArchLinux. With Python 2.7, there have been problems reported. Contact me if you experience any irregularities.

##USAGE

For now, the file `./tests/pyrfr_example.py` inside the repository serves as the
only real documentation of the Python bindings besides the docstrings.
