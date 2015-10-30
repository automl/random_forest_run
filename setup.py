from distutils.core import setup, Extension
import distutils.command.build
import numpy as np

from Cython.Build import cythonize





include_dirs = ['./include', np.get_include()]


extensions = cythonize(
					[
						Extension('pyrfr.regression',
						sources=['pyrfr/regression.pyx'],
						language="c++",
						include_dirs=include_dirs,
						#extra_compile_args = ['-O0','-g', '-std=c++11'])
						extra_compile_args = ['-O2', '-std=c++11'])
					])



setup(
	name='pyrfr',
	version='0.1.0',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 3 - Alpha'],
	ext_modules=extensions
)
