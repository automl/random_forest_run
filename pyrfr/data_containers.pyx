import cython
from libcpp cimport bool
from libcpp.vector cimport vector


import numpy as np
cimport numpy as np

from type_defs cimport *





"""
Base classes
"""

cdef class regression_base:

	cdef data_container_base[pyrfr_num_t,pyrfr_response_regression_t, pyrfr_index_t] *thisptr

	def __dealloc__(self):
		del self.thisptr

	def num_features(self):
		""" the number of features of the data"""
		return self.thisptr.num_features()

	def num_data_points(self):
		""" the number of data points in the container """
		return self.thisptr.num_data_points()

	def set_type_of_feature(self, pyrfr_index_t fi, pyrfr_index_t ft):
		self.thisptr.set_type_of_feature(fi,ft)

	def get_type_of_feature(self, pyrfr_index_t fi):
		return self.thisptr.get_type_of_feature(fi)

	def retrieve_data_point(self, int index):
		""" return a point in the data """
		while index < 0:
			index += self.thisptr.num_data_points()

		if index > self.thisptr.num_data_points():
			raise ValueError("Supplied index is too large: {} > {}".format(index,self.thisptr.num_data_points()-1))
		return self.thisptr.retrieve_data_point(index)




"""
The actual containers available in the python module
"""

cdef class numpy_container_regression(regression_base):
	""" A data container wrapping three numpy arrays"""

	cdef object features
	cdef object responses
	cdef object types
	
	def __cinit__(self, np.ndarray[pyrfr_num_t,ndim=2] feats, np.ndarray[pyrfr_response_regression_t,ndim=1] resp, np.ndarray[pyrfr_index_t] types):
		""" constructor should make no copy if the data comes in C-contiguous form."""
		# store a 'reference' so that the numpy array does not get garbage collected
		# also, assure the data is contiguous in memory
		self.features = np.ascontiguousarray(feats)
		self.responses= np.ascontiguousarray(resp)
		self.types    = np.ascontiguousarray(types)
		self.thisptr = new array_data_container[pyrfr_num_t,pyrfr_response_regression_t, pyrfr_index_t] (&feats[0,0], &resp[0], &types[0], feats.shape[0], feats.shape[1])
		

	def add_data_point(self, np.ndarray[pyrfr_num_t,ndim=1] fs, pyrfr_response_regression_t r):
		""" adds a data point by creating new python arrays, thus copies the data"""
		assert fs.shape[0] == self.features.shape[1]

		# create new numpy arrays with the extended data
		cdef np.ndarray[np.double_t,ndim=2] feats = np.vstack([self.features, fs])
		cdef np.ndarray[np.double_t,ndim=1] resp  = np.append(self.responses, r)
		cdef np.ndarray[pyrfr_index_t,ndim=1] types  = self.types
		self.features = feats
		self.responses= resp

		# replace the actual C++ container
		del self.thisptr
		self.thisptr = new array_data_container[pyrfr_num_t,pyrfr_response_regression_t, pyrfr_index_t] (&feats[0,0], &resp[0], &types[0], feats.shape[0], feats.shape[1])


cdef class mostly_continuous_data_regression(regression_base):

	def __cinit__(self, int num_features):
		self.thisptr = new mostly_continuous_data[pyrfr_num_t,pyrfr_response_regression_t, pyrfr_index_t] (num_features)

	def add_data_point(self, np.ndarray[pyrfr_num_t,ndim=1] fs, pyrfr_response_regression_t r):
		self.thisptr.add_data_point(&fs[0], fs.shape[0], r)

	def add_data_points(self, np.ndarray[pyrfr_num_t,ndim=2] feats, np.ndarray[pyrfr_response_regression_t,ndim=1] resp):
		for i in range(feats.shape[0]):
			self.thisptr.add_data_point(&feats[i,0], feats.shape[1], resp[i])

