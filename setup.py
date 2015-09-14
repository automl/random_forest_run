from distutils.core import setup, Extension
import distutils.command.build_ext

import sys



class build_ext(distutils.command.build_ext.build_ext):
	"""Subclass of build subcommand to specify the name of the boost-python
	library
	"""
	user_options = distutils.command.build_ext.build_ext.user_options + [
		('boost-python-lib-name=', None,
		 'specify the name of the boost python library'),
		]

	def initialize_options(self, *args, **kwargs):
		self.boost_python_lib_name = None
		distutils.command.build_ext.build_ext.initialize_options(self, *args, **kwargs)


	def finalize_options(self, *args, **kwargs):
		if self.boost_python_lib_name is None:
			self.boost_python_lib_name = 'boost_python'
		distutils.command.build_ext.build_ext.finalize_options(self, *args, **kwargs)
	
	def run(self, *args, **kwargs):
	# XXX gotta be a better way than globals
	#global BOOST_PYTHON_LIB_NAME
	#BOOST_PYTHON_LIB_NAME = self.boost_python_lib_name
		print(self.boost_python_lib_name)
		#self.libraries = [ (self.boost_python_lib_name,{}) ]
		distutils.command.build_ext.build_ext.run(self, *args, **kwargs)



rfr = Extension('rfr',
					include_dirs = ['./boost_numpy', './include'],
					library_dirs = ['/usr/local/lib'],
					libraries = ['boost_python3'],
					sources = ['python_module/rfr.cpp'],
					extra_compile_args = ['-O3', '-std=c++11'])


setup(
	name='pyrfr',
	version='0.0.1',
	author='Stefan Falkner',
	author_email='sfalkner@cs.uni-freiburg.de',
	license='Use as you wish. No guarantees whatsoever.',
	classifiers=['Development Status :: 2 - Pre-Alpha'],
	cmdclass={'build_ext': build_ext},
	ext_modules=[rfr],
)
