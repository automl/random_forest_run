import cython
from libcpp cimport bool
from libcpp.vector cimport vector


import numpy as np
cimport numpy as np


"""
All the original C++ classes
"""

cdef extern from "rfr/data_containers/data_container_base.hpp" namespace "rfr::data_containers":
	cdef cppclass data_container_base[num_type, response_type, index_type]:
		array_data_container(num_type*, response_type*, index_type*, index_type, index_type)
		index_type num_features()
		index_type num_data_points()
		bool add_data_point(num_type*, index_type, response_type)
		vector[num_type] retrieve_data_point (index_type index)

cdef extern from "rfr/data_containers/array_wrapper.hpp" namespace "rfr::data_containers":
	cdef cppclass array_data_container[num_type, response_type, index_type](data_container_base):
		array_data_container(num_type*, response_type*, index_type*, index_type, index_type)
		index_type num_features()
		index_type num_data_points()
		bool add_data_point(num_type*, index_type, response_type)
		vector[num_type] retrieve_data_point (index_type)





"""
Base classes
"""
cdef class regression_base:
	cdef data_container_base[np.double_t,np.double_t, np.uint8_t] *thisptr
	
	def __dealloc__(self):
		del self.thisptr

	def num_features(self):
		""" the number of features of the data"""
		return self.thisptr.num_features()

	def num_data_points(self):
		""" the number of data points in the container """
		return self.thisptr.num_data_points()

	def retrieve_data_point(self, index):
		""" return a point in the data """
		return self.thisptr.retrieve_data_point(index)



cdef class numpy_container_regression(regression_base):
	""" A data container wrapping three numpy arrays"""

	cdef object features
	cdef object responses
	cdef object types

	
	def __cinit__(self, np.ndarray[np.double_t,ndim=2] feats, np.ndarray[np.double_t,ndim=1] resp, np.ndarray[np.uint8_t] types):
		# store a 'reference' so that the numpy array does not get garbage collected
		self.features = feats
		self.responses= resp
		self.types    = types
		self.thisptr = new array_data_container[np.double_t, np.double_t, np.uint8_t] (&feats[0,0], &resp[0], &types[0], feats.shape[0], feats.shape[1])

	def add_data_point(self, np.ndarray[np.double_t,ndim=1] fs, np.double_t r):
		assert fs.shape[0] == self.features.shape[1]
		
		cdef np.ndarray[np.double_t,ndim=2] feats = np.vstack([self.features, fs])
		cdef np.ndarray[np.double_t,ndim=1] resp  = np.append(self.responses, r)
		cdef np.ndarray[np.uint8_t,ndim=1] types  = self.types
		self.features = feats
		self.responses= resp
		
	
		del self.thisptr
		self.thisptr = new array_data_container[np.double_t, np.double_t, np.uint8_t] (&feats[0,0], &resp[0], &types[0], feats.shape[0], feats.shape[1])
