#RFR
A extensible C++ library for random forests with Python bindings.

## Requirements

For the C++ library itself, you need no additional libaries, only a C++11 capable compiler.
The development is done using GCC 4.8 and 5.2.

For the Python bindings, you will need

```
numpy
boost (with the optional python module)
```

Any decent version (>1.32 or so) of Boost should do.


## INSTALLING THE PYTHON BINDINGS
The Python bindings are implemented using the boost python library, so you
will need boost installed with the optional python component enabled.

To install the Python module you will need to execute the following commands:

```
git submodule init
git submodule update
python setup.py build --boost-python-lib-name=boost_python-py27
python setup.py install --user
```

The first two will pull another git repository into the directory. It is used
to access numpy arrays easily within C++. The third command has an important
option: "boost-python-lib-name" which depends on your python version and
operating system you use. The last line simply installs the module into your
home directory.

The above commands work for Python 2.7 on Ubuntu 14.04 LTS.
On a recent Arch Linux this line could look like that:

```
python  setup.py build --boost-python-lib-name=boost_python3
```
and would build the module for Python 3. You can find the name of the library
by searching in `/lib/` or `/usr/lib/` for files starting with `libboost_python`

##USAGE

For now, the file `./tests/python_test.py` inside the repository serves as the
only real documentation of the Python bindings.
