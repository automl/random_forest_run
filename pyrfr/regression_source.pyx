"""
requires these three types to be defined when included:
-------------------------------------------------------

ctypedef np.double_t num_t
ctypedef np.double_t response_t
ctypedef np.uint_t   index_t
"""

import cython
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.vector cimport vector


import numpy as np
cimport numpy as np

from cpp_classes cimport *

ctypedef default_random_engine rng_t
ctypedef tree_base[rng_t, num_t, response_t, index_t] tree_base_t
ctypedef regression_forest[ tree_base_t, rng_t, num_t, response_t, index_t] regression_forest_base_t

ctypedef  k_ary_random_tree[two, binary_split_one_feature_rss_loss[rng_t, num_t, response_t,index_t], rng_t, num_t, response_t, index_t] binary_rss_tree_t
ctypedef  k_ary_random_tree[two, binary_split_one_feature_rss_loss_v2[rng_t, num_t, response_t,index_t], rng_t, num_t, response_t, index_t] binary_rss_tree_v2_t


"""
Base classes:
=============
"""
cdef class data_base:
	""" 
	base class for the data container to reuse as much code as possible
	"""
	cdef data_container_base[num_t,response_t, index_t] *thisptr

	def __dealloc__(self):
		del self.thisptr

	def num_features(self):
		""" 
		the number of features of the data
		"""
		return self.thisptr.num_features()

	def num_data_points(self):
		""" 
		the number of data points in the container 
 		"""
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
		""" 
		return a point in the response data 
		"""
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

"""
 The data containers available in the python module 
"""
cdef class numpy_data_container(data_base):
	""" 
	A data container wrapping three numpy arrays
	"""

	cdef public object features
	cdef public object responses
	cdef public object types

	def __cinit__(self, np.ndarray[num_t,ndim=2] feats, np.ndarray[response_t,ndim=1] resp, np.ndarray[index_t] types):
		""" 
		constructor should make no copy if the data comes in C-contiguous form.
		"""
		# store a 'reference' so that the numpy array does not get garbage collected
		# also, assure the data is contiguous in memory
		self.features = np.ascontiguousarray(feats)
		self.responses= np.ascontiguousarray(resp)
		self.types    = np.ascontiguousarray(types)
			
		
		if (self.features.shape[0] != self.responses.shape[0]):
			print(self.features.shape, self.responses.shape)
			raise ValueError("Number of datapoints and responses are incompatible!")
		
		if (self.features.shape[1] != self.types.shape[0]):
			raise ValueError("Number of features and types are incompatible!")
		
		
		cdef np.ndarray[num_t, ndim=2] f = self.features
		cdef np.ndarray[response_t, ndim=1] r = self.responses
		cdef np.ndarray[index_t, ndim=1] t = self.types 
		
		self.thisptr = new array_data_container[num_t,response_t, index_t] (&f[0,0], &r[0], &t[0], f.shape[0], f.shape[1])


	def add_data_point(self, np.ndarray[num_t,ndim=1] fs, response_t r):
		""" 
		adds a data point by creating new python arrays, thus copies the data
		"""
		if (fs.shape[0] != self.features.shape[1]):
			raise(ValueError, "Wrong number of features supplied.")

		# create new numpy arrays with the extended data
		cdef np.ndarray[num_t,ndim=2] feats = np.vstack([self.features, fs])
		cdef np.ndarray[response_t,ndim=1] resp  = np.append(self.responses, r)
		cdef np.ndarray[index_t,ndim=1] types  = self.types
		self.features = feats
		self.responses= resp

		# replace the actual C++ container
		del self.thisptr
		self.thisptr = new array_data_container[num_t,response_t, index_t] (&feats[0,0], &resp[0], &types[0], feats.shape[0], feats.shape[1])

cdef class mostly_continuous_data_container(data_base):
	""" 
	A python wrapper around the C++ data container for mostly continuous data.
	"""

	def __cinit__(self, int num_features):
		self.thisptr = new mostly_continuous_data[num_t,response_t, index_t] (num_features)

	def add_data_point(self, np.ndarray[num_t,ndim=1] fs, response_t r):
		self.thisptr.add_data_point(&fs[0], fs.shape[0], r)

	def import_numpy_arrays(self, np.ndarray[num_t,ndim=2] feats, np.ndarray[response_t,ndim=1] resp):
		for i in range(feats.shape[0]):
			self.thisptr.add_data_point(&feats[i,0], feats.shape[1], resp[i])

cdef class mostly_continuous_data_with_instances_container(data_base):
	""" 
	A python wrapper around the C++ data container for mostly continuous data with instances.
	"""

	def __cinit__(self, int num_configuration_features, int num_instance_features):
		self.thisptr = new mostly_continuous_data_with_instances[num_t, response_t, index_t] (num_configuration_features, num_instance_features)

	def add_configuration(self, np.ndarray[num_t, ndim=1] c_fs):
		tmpptr = <mostly_continuous_data_with_instances[num_t, response_t, index_t]*> self.thisptr
		return(tmpptr.add_configuration(&c_fs[0], c_fs.shape[0]))
	
	def add_instance(self, np.ndarray[num_t, ndim = 1] i_fs):
		tmpptr = <mostly_continuous_data_with_instances[num_t, response_t, index_t]*> self.thisptr
		return(tmpptr.add_instance(&i_fs[0], i_fs.shape[0]))

	def add_data_point(self, index_t config_index, index_t instance_index, num_t rs):
		tmpptr = <mostly_continuous_data_with_instances[num_t, response_t, index_t]*> self.thisptr
		return(tmpptr.add_data_point(config_index, instance_index, rs))

	def import_instances(self, np.ndarray[num_t, ndim = 2] instances):
		tmpptr = <mostly_continuous_data_with_instances[num_t, response_t, index_t]*> self.thisptr
		for i in range(instances.shape[0]):
			tmpptr.add_instance(&instances[i,0], instances.shape[1])
	
	def import_configurations(self, np.ndarray[num_t, ndim = 2] configurations):
		tmpptr = <mostly_continuous_data_with_instances[num_t, response_t, index_t]*> self.thisptr
		for i in range(configurations.shape[0]):
			tmpptr.add_configuration(&configurations[i,0], configurations.shape[1])
	
	def add_data_points(self,  np.ndarray[index_t, ndim = 2] config_instance_pairs, np.ndarray[response_t, ndim = 1] responses):
		if (config_instance_pairs.shape[0] != responses.shape[0]):
			raise ValueError("Number of configuration-instance-pairs and response values do not match!")
		if (config_instance_pairs.shape[1] != 2):
			raise ValueError("The config - instance pairs matrix has to be have two columns!")
		
		for i in range(responses.shape[0]):
			self.add_data_point(config_instance_pairs[i][0], config_instance_pairs[i][1], responses[i])
	
	def num_instances(self):
		tmpptr = <mostly_continuous_data_with_instances[num_t, response_t, index_t]*> self.thisptr
		return(tmpptr.num_instances())

	def get_instance_set(self):
		tmpptr = <mostly_continuous_data_with_instances[num_t, response_t, index_t]*> self.thisptr
		return(tmpptr.get_instance_set())

"""
The actual forests:
===================
"""

cdef class regression_forest_base:
	""" base class providing the basic functionality needed for any of
	the C++ forest classes
	"""

	# attributes for the forest parameters
	cdef public index_t num_trees
	"""Sets the number of trees in the forest (Default 10)."""
	
	cdef public index_t num_data_points_per_tree
	""" Determines how many data points are used for each tree. 

	If set to zero the whole data will be used for each tree, otherwise
	this sets the size of the (sub)sample drawn from the data. Note, if
	do_bootstrapping = False, this number must not exceed the number of
	data points in the data container. 
	"""
	
	cdef public bool do_bootstrapping
	""" Whether to sample from the data with (True) or without (False)
	replacement (Default: True).
	"""

	cdef public bool compute_oob_error
	""" For an unbiased estimate of the generalization error, the forest
	can compute the out-of-bag error on the fly if. There is an additional
	overhead depending on the data set size and your bootstrapping settings.
	This bool turns that feature on/off.
	"""

	#attributes for the individual trees.

	cdef public index_t max_features
	""" how many (randomly selected) features are considered for easch split (Default: 0 - all features)."""
	cdef public index_t max_depth
	""" Limits the depth of the trees (Default: 0 - *virtually* no restriction)."""
	cdef public index_t max_num_nodes
	""" Limits the number of nodes (internal and leafs) of each tree (Default: 0 - *virtually* no restriction)"""
	cdef public index_t min_samples_to_split
	""" The minimal number of samples considered to be split (Default: 2)"""
	cdef public index_t min_samples_in_leaf
	""" The smallest number of samples allowed in a leaf (Default: 2)"""
	cdef public response_t epsilon_purity
	""" A small float that specifies when two feature values are consider equal (Default: 1e-8)"""

	cdef public index_t seed
	""" Set this to anything other than zero to reseed the random number generator."""

	cdef rng_t *rng_ptr

	def __init__(self):
		self.num_trees=10
		self.num_data_points_per_tree = 0
		self.do_bootstrapping = True
		self.compute_oob_error = False
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

	cdef recover_settings_from_forest_options(self, forest_options[num_t, response_t, index_t] fo):
		self.num_trees = fo.num_trees
		self.num_data_points_per_tree = fo.num_data_points_per_tree
		self.do_bootstrapping = fo.do_bootstrapping
		self.compute_oob_error = fo.compute_oob_error
				
		self.max_features = fo.tree_opts.max_features
		self.max_depth = fo.tree_opts.max_depth
		self.max_num_nodes = fo.tree_opts.max_num_nodes
		self.min_samples_to_split = fo.tree_opts.min_samples_to_split
		self.min_samples_in_leaf = fo.tree_opts.min_samples_to_split
		self.epsilon_purity = fo.tree_opts.epsilon_purity

	cdef forest_options[num_t, response_t, index_t] build_forest_options(self, data_base data):
				
		cdef tree_options[num_t, response_t, index_t] to
		to.max_features = self.max_features if self.max_features > 0 else data.num_features()
		to.max_depth = self.max_depth if self.max_depth > 0 else 2*data.num_data_points()+1
		to.max_num_nodes = self.max_num_nodes if self.max_num_nodes > 0 else 2*data.num_data_points()+1
		to.min_samples_to_split = self.min_samples_to_split
		to.min_samples_in_leaf = self.min_samples_in_leaf
		to.epsilon_purity = self.epsilon_purity

		#construct the forest option object.
		cdef forest_options[num_t, response_t, index_t] fo

		fo.num_trees=self.num_trees
		fo.num_data_points_per_tree = self.num_data_points_per_tree if self.num_data_points_per_tree > 0 else data.num_data_points()
		fo.do_bootstrapping = self.do_bootstrapping
		fo.compute_oob_error = self.compute_oob_error
		fo.tree_opts = to

		#reseed the rng if needed.
		if (self.seed > 0):
			self.rng_ptr.seed(self.seed)
			self.seed=0
			
		return(fo)
	

cdef class binary_rss(regression_forest_base):
	"""
	The random forest regressor. Builds a forest of trees from the training set (X,y) and predicts the regression targets for X. The splits are made by calculating the residual sum of squares and taking the min.
	"""
	cdef regression_forest[ binary_rss_tree_t, rng_t, num_t, response_t, index_t]* forest_ptr
	
	def __init(self):
		super(binary_rss, self).__init__()
	
	def __dealloc__(self):
		del self.forest_ptr

	def save_to_binary_file(self, filename):
		# todo: make sure directory exists and all permissions are OK
		self.forest_ptr.save_to_binary_file(filename)
	
	def load_from_binary_file(self, filename):
		""" Simple wrapper around the C++ deserialization function.
		
		Beware: the recovered settings for the forest might not be the
		same in the python world. In particular, settings like
		num_data_points_per_tree = 0 have no equivalent in the C++ world, but
		are merely a convenience setting for pyrfr. As the serialization only
		knows about the C++ settings, the values used there are recovered. In
		the example of num_data_points_per_tree, the restored value equals the 
		number of data points available at training.		
		"""
		del self.forest_ptr
		self.forest_ptr = new regression_forest[ binary_rss_tree_t, rng_t, num_t, response_t, index_t] ()
		self.forest_ptr.load_from_binary_file(filename)
		self.recover_settings_from_forest_options(self.forest_ptr.get_forest_options())

	def __reduce__(self):
		d = {}
		d['str_representation'] = self.forest_ptr.save_into_string()
		return (binary_rss, (), d)
	
	def __setstate__(self, d):
		del self.forest_ptr
		self.forest_ptr = new regression_forest[ binary_rss_tree_t, rng_t, num_t, response_t, index_t] ()
		self.forest_ptr.load_from_string(d['str_representation'])
		self.recover_settings_from_forest_options(self.forest_ptr.get_forest_options())

	def fit(self, data_base data):
		""" The fit method.

		:param data: a regression data container with the input data
		:type data: pyrfr.regression.data_base
		"""
		del self.forest_ptr
		fo = self.build_forest_options(data)
		self.forest_ptr = new regression_forest[ binary_rss_tree_t, rng_t, num_t, response_t, index_t] (fo)
		self.forest_ptr.fit(deref(data.thisptr), deref(self.rng_ptr))

	def predict(self, np.ndarray[num_t,ndim=1] feats):
		""" The basic prediction method.

		:param feats: feature vector
		:type feats: 1d numpy array of doubles

		:returns: a tuple containing the mean and the variance prediction
		"""
		return self.forest_ptr.predict_mean_var(&feats[0])
	
	def batch_predictions(self, np.ndarray[num_t, ndim=2] features):
		""" predicting several inputs simultaniously
		
		:param features: several features in a matrix where each row is a valid feature vector
		:type features: 2d numpy array of doubles
		
		:returns: tuple holding of the mean predictions and corresponding standard deviations
		"""
	
		means = np.zeros(features.shape[0], dtype=np.float)
		vrs   = np.zeros(features.shape[0], dtype=np.float)
		
		for i in range(features.shape[0]):
			means[i], vrs[i] = self.forest_ptr.predict_mean_var(&features[i,0])
		return (means, vrs)


	def all_leaf_values(self, np.ndarray[num_t, ndim=1] feats):
		"""
		helper function to get all the repsonses from each tree

		:param feats: feature vector
		:type feats: 1d numpy array of doubles

		:returns: a nested list with the responses of the leafs x falls into from each tree.
		"""
		return (self.forest_ptr.all_leaf_values(&feats[0]))

	def save_latex_representation(self, pattern):
		self.forest_ptr.save_latex_representation(pattern)

	def quantile_prediction(self, np.ndarray[num_t,ndim=1] feats, np.ndarray[num_t,ndim=1] alphas):
		"""
		Quantile regression forest as explained in "Quantile Regression Forests" by Nicolai Meinhausen.

 		:param feats: feature vector to condition on
 		:type feats: 1d numpy array of doubles
		:param alphas: requested quantiles to estimate
		:type alphas: 1d numpy array of doubles

		:returns: the alhpa quantiles for the response at the given feature vector
		"""

		leaf_values = self.all_leaf_values(feats)
		
		# compute the weights for each leaf value
		weights = map(lambda v: [1./(len(v)*len(leaf_values))]*len(v), leaf_values)
			
		# flatten the nested lists using list addition
		weights = np.array(sum(weights,[]))
		values = np.array(sum(leaf_values,[]))

		# sort them according to the response values
		sort_indices = np.argsort(values)
		weights = weights[sort_indices]
		values = values[sort_indices]

		# find the indices where the quantiles would have to be inserted
		cum_weights = np.cumsum(weights)
		alpha_indices = np.searchsorted (cum_weights, alphas)

		# for now just return the value at that point. One could do a linear
		# interpolation, but that should be good enough for now
		quantiles = values[np.minimum(alpha_indices, values.shape[0]-1)]

		return(quantiles)

	def covariance (self, np.ndarray[num_t, ndim=1] f1, np.ndarray[num_t, ndim=1] f2):
		return(self.forest_ptr.covariance(&f1[0], &f2[0]))


	def induced_partitionings(self, pcs):
		return([self.forest_ptr.partition_of_tree(i, pcs) for i in range(self.num_trees)])


	def predict_marginalized_over_instances(self, np.ndarray[num_t, ndim=1] feats, data_container):
		if not isinstance(data_container, mostly_continuous_data_with_instances_container):
			raise TypeError("Data container has to be contain instances!")
		
		cdef vector[num_t] instances
		
		instances = data_container.get_instance_set()
		
		return (self.forest_ptr.predict_mean_var_marginalized_over_set( &feats[0], &(instances.data()[0]), data_container.num_instances()))


	def out_of_bag_error(self):
		return(self.forest_ptr.out_of_bag_error())

cdef class binary_rss_v2(regression_forest_base):
	""" test class for benchmarks! ***DO NOT USE***"""
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
		return self.forest_ptr.predict_mean_var(&feats[0])

	def save_latex_representation(self, pattern):
		self.forest_ptr.save_latex_representation(pattern)

