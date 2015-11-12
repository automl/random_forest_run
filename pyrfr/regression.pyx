import cython
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector


import numpy as np
cimport numpy as np

from cpp_classes cimport *


ctypedef np.double_t num_t
ctypedef np.double_t response_t
ctypedef np.uint_t   index_t
ctypedef default_random_engine rng_t
ctypedef tree_base[rng_t, num_t, response_t, index_t] tree_base_t
ctypedef regression_forest[ tree_base_t, rng_t, num_t, response_t, index_t] regression_forest_base_t


ctypedef  k_ary_random_tree[two, binary_split_one_feature_rss_loss[rng_t, num_t, response_t,index_t], rng_t, num_t, response_t, index_t] binary_rss_tree_t
ctypedef  k_ary_random_tree[two, binary_split_one_feature_rss_loss_v2[rng_t, num_t, response_t,index_t], rng_t, num_t, response_t, index_t] binary_rss_tree_v2_t


"""
Base classes
"""
cdef class data_base:
	""" base class for the data container to reuse as much code as possible"""
	cdef data_container_base[num_t,response_t, index_t] *thisptr

	def __dealloc__(self):
		del self.thisptr

	def num_features(self):
		""" the number of features of the data"""
		return self.thisptr.num_features()

	def num_data_points(self):
		""" the number of data points in the container """
		return self.thisptr.num_data_points()

	def set_type_of_feature(self, index_t fi, index_t ft):
		self.thisptr.set_type_of_feature(fi,ft)

	def get_type_of_feature(self, index_t fi):
		return self.thisptr.get_type_of_feature(fi)

	def retrieve_data_point(self, int index):
		""" return a point in the data """
		while index < 0:
			index += self.thisptr.num_data_points()
		# to avoid:
		#		warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
		cdef index_t new_i = <index_t> index
		if new_i > self.thisptr.num_data_points():
			raise ValueError("Supplied index is too large: {} > {}".format(new_i,self.thisptr.num_data_points()-1))
		return self.thisptr.retrieve_data_point(new_i)

	def retrieve_response(self, int index):
		""" return a point in the data """
		while index < 0:
			index += self.thisptr.num_data_points()
		# to avoid:
		#		warning: comparison between signed and unsigned integer expressions [-Wsign-compare]
		cdef index_t new_i = <index_t> index
		if new_i > self.thisptr.num_data_points():
			raise ValueError("Supplied index is too large: {} > {}".format(new_i,self.thisptr.num_data_points()-1))
		return self.thisptr.response(new_i)

	def export_features(self):
		return np.array([ self.thisptr.retrieve_data_point(i) for i in range(self.num_data_points())])

	def export_responses(self):
		return np.array([ self.thisptr.response(i) for i in range(self.num_data_points())])

######################################################
# The data containers available in the python module #
######################################################
cdef class numpy_data_container(data_base):
	""" A data container wrapping three numpy arrays"""

	cdef public object features
	cdef public object responses
	cdef public object types

	def __cinit__(self, np.ndarray[num_t,ndim=2] feats, np.ndarray[response_t,ndim=1] resp, np.ndarray[index_t] types):
		""" constructor should make no copy if the data comes in C-contiguous form."""
		# store a 'reference' so that the numpy array does not get garbage collected
		# also, assure the data is contiguous in memory
		self.features = np.ascontiguousarray(feats)
		self.responses= np.ascontiguousarray(resp)
		self.types    = np.ascontiguousarray(types)
		self.thisptr = new array_data_container[num_t,response_t, index_t] (&feats[0,0], &resp[0], &types[0], feats.shape[0], feats.shape[1])


	def add_data_point(self, np.ndarray[num_t,ndim=1] fs, response_t r):
		""" adds a data point by creating new python arrays, thus copies the data"""
		assert fs.shape[0] == self.features.shape[1]

		# create new numpy arrays with the extended data
		cdef np.ndarray[np.double_t,ndim=2] feats = np.vstack([self.features, fs])
		cdef np.ndarray[np.double_t,ndim=1] resp  = np.append(self.responses, r)
		cdef np.ndarray[index_t,ndim=1] types  = self.types
		self.features = feats
		self.responses= resp

		# replace the actual C++ container
		del self.thisptr
		self.thisptr = new array_data_container[num_t,response_t, index_t] (&feats[0,0], &resp[0], &types[0], feats.shape[0], feats.shape[1])

cdef class mostly_continuous_data_container(data_base):
	""" A python wrapper around the C++ data container for mostly continuous data."""

	def __cinit__(self, int num_features):
		self.thisptr = new mostly_continuous_data[num_t,response_t, index_t] (num_features)

	def add_data_point(self, np.ndarray[num_t,ndim=1] fs, response_t r):
		self.thisptr.add_data_point(&fs[0], fs.shape[0], r)

	def import_numpy_arrays(self, np.ndarray[num_t,ndim=2] feats, np.ndarray[response_t,ndim=1] resp):
		for i in range(feats.shape[0]):
			self.thisptr.add_data_point(&feats[i,0], feats.shape[1], resp[i])



######################
# The actual forests #
######################
cdef class regression_forest_base:
	""" base class providing the basic functionality needed for any of the C++ forest classes"""
	# attributes for the forest parameters
	cdef public index_t num_trees
	cdef public index_t num_data_points_per_tree
	cdef public bool do_bootstrapping

	# attributes for the individual trees
	cdef public index_t max_features
	cdef public index_t max_depth
	cdef public index_t max_num_nodes
	cdef public index_t min_samples_to_split
	cdef public index_t min_samples_in_leaf
	cdef public response_t epsilon_purity

	# to (re)seed the rng
	cdef public index_t seed


	cdef rng_t *rng_ptr


	def __init__(self):
		self.num_trees=10
		self.num_data_points_per_tree = 0
		self.do_bootstrapping = True
		self.max_features = 0
		self.max_depth = 0
		self.max_num_nodes = 0
		self.min_samples_to_split = 2
		self.min_samples_in_leaf = 1
		self.epsilon_purity = 1e-8
		
		self.seed = 0
		self.rng_ptr = new rng_t(42)

	def __dealloc__(self):
		del self.rng_ptr

	cdef forest_options[num_t, response_t, index_t] build_forest_options(self, data_base data):

		cdef tree_options[num_t, response_t, index_t] to
		to.max_features = self.max_features if self.max_features > 0 else data.num_features()
		to.max_depth = self.max_depth
		to.max_num_nodes = self.max_num_nodes
		to.min_samples_to_split = self.min_samples_to_split
		to.min_samples_in_leaf = self.min_samples_in_leaf
		to.epsilon_purity = self.epsilon_purity


		#construct the forest option object
		cdef forest_options[num_t, response_t, index_t] fo

		fo.num_trees=self.num_trees
		fo.num_data_points_per_tree = self.num_data_points_per_tree if self.num_data_points_per_tree > 0 else data.num_data_points()
		fo.do_bootstrapping = self.do_bootstrapping
		fo.tree_opts = to

		#reseed the rng if needed
		if (self.seed > 0):
			self.rng_ptr.seed(self.seed)
			self.seed=0
			
		return(fo)


cdef class binary_rss(regression_forest_base):
	cdef regression_forest[ binary_rss_tree_t, rng_t, num_t, response_t, index_t]* forest_ptr
	
	def __init(self):
		super(binary_rss, self).__init__()
	
	def __dealloc__(self):
		del self.forest_ptr
	
	def fit(self, data_base data):
		del self.forest_ptr
		fo = self.build_forest_options(data)
		self.forest_ptr = new regression_forest[ binary_rss_tree_t, rng_t, num_t, response_t, index_t] (fo)
		self.forest_ptr.fit(deref(data.thisptr), deref(self.rng_ptr))

	def predict(self, np.ndarray[num_t,ndim=1] feats):
		return self.forest_ptr.predict_mean_std(&feats[0])

	def all_leaf_values(self, np.ndarray[num_t, ndim=1] feats):
		return (self.forest_ptr.all_leaf_values(&feats[0]))



cdef class binary_rss_v2(regression_forest_base):
	cdef regression_forest[ binary_rss_tree_v2_t, rng_t, num_t, response_t, index_t]* forest_ptr
	
	def __init(self):
		super(binary_rss, self).__init__()
	
	def __dealloc__(self):
		del self.forest_ptr
	
	def fit(self, data_base data):
		del self.forest_ptr
		fo = self.build_forest_options(data)
		self.forest_ptr = new regression_forest[ binary_rss_tree_v2_t, rng_t, num_t, response_t, index_t] (fo)
		self.forest_ptr.fit(deref(data.thisptr), deref(self.rng_ptr))

	def predict(self, np.ndarray[num_t,ndim=1] feats):
		return self.forest_ptr.predict_mean_std(&feats[0])

