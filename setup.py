from setuptools import setup, Extension


rfr = Extension('rfr',
                    include_dirs = ['/usr/local/include', './boost_numpy', './include'],
                    libraries = ['boost_python3'],
                    library_dirs = ['/usr/local/lib'],
                    sources = ['python_module/rfr.cpp'],
                    extra_compile_args = ['-O3', '-std=c++11'])



setup(
    name='pyrfr',
    version='0.0.1',
    author='Stefan Falkner',
    author_email='sfalkner@cs.uni-freiburg.de',
    license='Use as you wish. No guarantees whatsoever.',
    classifiers=['Development Status :: 3 - Alpha'],
    ext_modules=[rfr],
)
