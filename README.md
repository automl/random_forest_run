#RFR
A extensible C++ library for random forests with Python bindings.

## Requirements

For the C++ library itself, you need no additional libaries, only a C++11 capable compiler.
The development is done using GCC 4.8 and 5.2.

For the Python bindings, you will need

```
numpy
Cython
```


## Installing the Python Bindings
Simply execute 
```
python setup.py install --user
```
to install it. After the installation finishes (hopefully) sucessfully, you can use the library with the pyrfr module.

The above commands work for Python 3.4 and 3.5 on Gentoo and ArchLinux. With Python 2.7, there have been problems reported. Contact me if you experience any irregularities.

##USAGE

For now, the file `./tests/python_test.py` inside the repository serves as the
only real documentation of the Python bindings.
