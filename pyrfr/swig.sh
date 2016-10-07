#TODO add building Doxygen XML description and calling doxy2swig to generate docstrings for sphinx

swig -v -c++ -python -I./../include/  regression.i
c++ -std=c++11 -c regression_wrap.cxx -I ../include/ -I /usr/include/python3.5m/ -I/usr/share/swig/3.0.10/python/ -fPIC
c++ -shared regression_wrap.o -o _regression.so
