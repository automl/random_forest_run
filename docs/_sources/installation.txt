*********
About pyRFR
*********
.. role:: bash(code)
    :language: bash

This package serves as the python interface to RFR, an extensible C++ librarry for random forests.

Requirements
************

For the C++ library itself, you need no additional libaries, only a C++11 capable compiler. The development is done using GCC 4.8 and 5.2. The Python bindings are only tested on Python 3.

For the Python bindings, you will need
        numpy
        Cython

Installation
************
Right now, pyRFR is not available on PyPI. To install the Python bindings, execute the following commands:

	:bash:`git clone git@bitbucket.org:aadfreiburg/random_forest_run.git`

        :bash:`cd random_forest_run`

	:bash:`python setup.py install --user`


Usage
*****
For now, the file
 
	:bash:`./examples/pyrfr_simple_examples.py`

and the other python scritps inside the repository serves as the only real documentation of the Python bindings.
