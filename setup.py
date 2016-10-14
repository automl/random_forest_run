from distutils.core import setup, Extension
import distutils.command.build
import numpy as np
from subprocess import call
from Cython.Build import cythonize




include_dirs = ['${CMAKE_SOURCE_DIR}/include', np.get_include()]
extra_compile_args = ['-O2', '-std=c++11']
#extra_compile_args = ['-g', '-std=c++11']


"""
extensions = cythonize(
					[
						Extension('pyrfr.regression',
						sources=['pyrfr/regression.pyx'],
						language="c++",
						include_dirs=include_dirs,
						extra_compile_args = extra_compile_args
						),
						
						Extension('pyrfr.regression32',
						sources=['pyrfr/regression32.pyx'],
						language="c++",
						include_dirs=include_dirs,
						extra_compile_args = extra_compile_args
						)
					])
"""

extensions = [	Extension(
					name = '_regression',
					sources=['pyrfr/regression.i'],
					include_dirs = include_dirs,
					swig_opts=['-c++'] + ['-I{}'.format(s) for s in include_dirs],
					extra_compile_args = extra_compile_args
				),
				Extension(
					name = '_util',
					sources=['pyrfr/util.i'],
					include_dirs = include_dirs,
					swig_opts=['-c++'] + ['-I{}'.format(s) for s in include_dirs],
					extra_compile_args = extra_compile_args
				)
			]


setup(
	name='pyrfr',
	version='0.2.1',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 3 - Alpha'],
	packages=['pyrfr'],
	ext_modules=extensions
)
