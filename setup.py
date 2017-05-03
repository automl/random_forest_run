from distutils.command.build import build
from setuptools.command.install import install
from distutils.core import setup, Extension


class CustomInstall(install):
	def run(self):
		self.run_command('build_ext')
		build.run(self)
		self.do_egg_install()

include_dirs = ['${CMAKE_SOURCE_DIR}/include', './include']
extra_compile_args = ['-O2', '-std=c++11']
# extra_compile_args = ['-g', '-std=c++11', '-O0']


extensions = [Extension(
						name='_regression',
						sources=['pyrfr/regression.i'],
						include_dirs=include_dirs,
						swig_opts=['-c++'] + ['-I{}'.format(s) for s in include_dirs],
						extra_compile_args=extra_compile_args
					),
					Extension(
							name='_util',
							sources=['pyrfr/util.i'],
							include_dirs=include_dirs,
							swig_opts=['-c++'] + ['-I{}'.format(s) for s in include_dirs],
							extra_compile_args=extra_compile_args
						)
			]

setup(
	name='pyrfr',
	version='0.4.0',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 3 - Alpha'],
	packages=['pyrfr'],
	ext_modules=extensions,
	cmdclass={'install': CustomInstall}
)
