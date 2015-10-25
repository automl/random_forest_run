from distutils.core import setup, Extension
import distutils.command.build
import numpy as np

from Cython.Build import cythonize





include_dirs = ['./include', np.get_include()]


extensions = cythonize(
					[
						Extension('pyrfr.data_containers',
						sources=['pyrfr/data_containers.pyx'],
						language="c++",
						include_dirs=include_dirs,
						extra_compile_args = ['-O2', '-std=c++11'])
					])



setup(
	name='pyrfr',
	version='0.0.2',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 2 - Pre-Alpha'],
	ext_modules=extensions
)
