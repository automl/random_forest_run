from distutils.core import setup, Extension
import distutils.command.build


import numpy


class build(distutils.command.build.build):
	"""Subclass of build to specify the name of the boost-python library
	"""
	user_options = distutils.command.build.build.user_options + [
		('boost-python-lib-name=', None,
		'specify the name of the boost python library'),
		]

	def initialize_options(self, *args, **kwargs):
		self.boost_python_lib_name = None
		distutils.command.build.build.initialize_options(self, *args, **kwargs)


	def finalize_options(self, *args, **kwargs):
		# if no name for the boost_python library has been provided, set the default
		if self.boost_python_lib_name is None:
			self.boost_python_lib_name = 'boost_python'
		distutils.command.build.build.finalize_options(self, *args, **kwargs)

	def run(self, *args, **kwargs):
		# replace the name of boost_python for the linker
		self.distribution.ext_modules[0].libraries = [self.boost_python_lib_name]

		# now run the regular build
		distutils.command.build.build.run(self, *args, **kwargs)


rfr = Extension('rfr',
					include_dirs = ['./boost_numpy', './include', numpy.get_include()],
					library_dirs = ['/usr/local/lib'],
					libraries = ['dummy_string'],	# just in case. This will be replaced later
					sources = ['python_module/rfr.cpp'],
					extra_compile_args = ['-O2', '-std=c++11'],
					extra_link_args = ['-O2'])	# not sure if this actually does anything


setup(
	name='pyrfr',
	version='0.0.1',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 2 - Pre-Alpha'],
	cmdclass={'build': build},
	ext_modules=[rfr],
    install_requires=['numpy']
)
