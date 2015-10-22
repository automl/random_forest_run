import cython

import numpy as np
cimport numpy as np



cdef extern from "rfr/data_containers/array_wrapper.hpp" namespace "rfr::data_containers":
	cdef cppclass array_data_container[num_type, response_type, index_type]:
		array_data_container(num_type*, response_type*, index_type*, index_type, index_type)
		index_type num_features()
		index_type num_data_points()


cdef class numpy_container_regression:
	""" A data container wrapping three numpy arrays"""
	cdef array_data_container[np.double_t, np.double_t, np.uint8_t] *thisptr

	cdef object features
	cdef object responses
	cdef object types

	
	def __cinit__(self, np.ndarray[np.double_t,ndim=2] feats, np.ndarray[np.double_t,ndim=1] resp, np.ndarray[np.uint8_t] types):
		# store a 'reference' so that the numpy array does not get deleted
		self.features = feats
		self.responses= resp
		self.types    = types
		self.thisptr = new array_data_container[np.double_t, np.double_t, np.uint8_t] (&feats[0,0], &resp[0], &types[0], feats.shape[0], feats.shape[1])

	def num_features(self):
		""" the number of features of the data"""
		return self.thisptr.num_features()

	def num_data_points(self):
		""" the number of data points in the container """
		return self.thisptr.num_data_points()

