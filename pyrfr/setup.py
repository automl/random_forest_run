from distutils.command.build import build
from setuptools.command.install import install
from distutils.core import setup, Extension

import fnmatch
import os




include_files = []
for root, dirnames, filenames in os.walk('include'):
    for filename in filenames:
        include_files.append(os.path.join(root, filename))



class CustomInstall(install):
	def run(self):
		self.run_command('build_ext')
		build.run(self)
		self.do_egg_install()

include_dirs = ['${CMAKE_SOURCE_DIR}/include', './include']
extra_compile_args = ['-O2', '-std=c++11']
#extra_compile_args = ['-g', '-std=c++11', '-O0']



extensions = [	Extension(
					name = '_regression',
					sources=['pyrfr/regression.i'],
					include_dirs = include_dirs,
					swig_opts=['-c++', '-modern', '-features', 'nondynamic'] + ['-I{}'.format(s) for s in include_dirs],
					extra_compile_args = extra_compile_args
				),
				Extension(
					name = '_util',
					sources=['pyrfr/util.i'],
					include_dirs = include_dirs,
					swig_opts=['-c++', '-modern', '-features', 'nondynamic'] + ['-I{}'.format(s) for s in include_dirs],
					extra_compile_args = extra_compile_args
				)
			]

setup(
	name='pyrfr',
	version='${RFR_VERSION_MAJOR}.${RFR_VERSION_MINOR}.${RFR_VERSION_RELEASE}',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 3 - Alpha'],
	packages=['pyrfr'],
	ext_modules=extensions,
	python_requires='>=3',
	package_data = {'pyrfr': ['docstrings.i']},
	data_files = [('include', include_files)],
	cmdclass={'install': CustomInstall}
)
