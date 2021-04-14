# RFR

A extensible C++ library for random forests with Python bindings with a BSD3 license.

## Requirements

For the C++ library itself, you need no additional libaries, only a C++11 capable compiler.
Technically, you need Boost if you want to compile the unit tests.
The development is done using GCC 7.2.
You probably have to set CMAKE\_CXX\_FLAGS to -std=c++11 when using older compilers.

```
CMAKE
DOXYGEN (if you want docstrings, which you probably do)
SWIG > 3.0
```


## Installing the Python Bindings
We upload the latest version to PYPI, so you can install it via
```
pip install pyrfr
```
Development is done with Python 3.6-3.9 on Ubuntu and the unittests are executed via github actions.
We do no longer support Python 2. Contact us if you experience any irregularities.

## USAGE

For now, the file `./tests/pyrfr_unit_test_*.py` inside the repository serve as the
only real documentation of the Python bindings besides the docstrings.
