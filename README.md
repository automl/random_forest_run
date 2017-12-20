# RFR

[![Build Status](https://travis-ci.org/automl/random_forest_run.svg?branch=master)](https://travis-ci.org/automl/random_forest_run)
[![Coverage Status](https://coveralls.io/repos/github/automl/random_forest_run/badge.svg?branch=master)](https://coveralls.io/github/automl/random_forest_run?branch=master)

A extensible C++ library for random forests with Python bindings.

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
Development is done with Python 3.6 on ArchLinux, but the unittests run on TravisCI with older version of Python and GCC. There have been problems reported with Python 2.7. Contact us if you experience any irregularities.

## USAGE

For now, the file `./tests/pyrfr_unit_test_*.py` inside the repository serve as the
only real documentation of the Python bindings besides the docstrings.
