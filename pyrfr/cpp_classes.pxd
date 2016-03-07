import cython
from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string

import numpy as np
cimport numpy as np


cdef extern from *:
	ctypedef void* two "2"


"""
----------------
Data Containers
----------------
"""

cdef extern from "rfr/data_containers/data_container_base.hpp" namespace "rfr::data_containers":
	cdef cppclass data_container_base[num_type, response_type, index_type]:
		array_data_container(num_type*, response_type*, index_type*, index_type, index_type)
		index_type num_features()
		index_type num_data_points()
		index_type get_type_of_feature (index_type)
		void add_data_point(num_type*, index_type, response_type) except +
		void set_type_of_feature (index_type, index_type) except +
		vector[num_type] retrieve_data_point (index_type index)
		response_type response(index_type)

cdef extern from "rfr/data_containers/array_wrapper.hpp" namespace "rfr::data_containers":
	cdef cppclass array_data_container[num_type, response_type, index_type](data_container_base[num_type, response_type, index_type]):
		array_data_container(num_type*, response_type*, index_type*, index_type, index_type)
		index_type num_features()
		index_type num_data_points()
		index_type get_type_of_feature (index_type)
		void add_data_point(num_type*, index_type, response_type) except +
		void set_type_of_feature (index_type, index_type) except +
		vector[num_type] retrieve_data_point (index_type)

cdef extern from "rfr/data_containers/mostly_continuous_data_container.hpp" namespace "rfr::data_containers":
	cdef cppclass mostly_continuous_data[num_type, response_type, index_type](data_container_base[num_type, response_type, index_type]):
		mostly_continuous_data (index_type num_f)
		index_type num_features()
		index_type num_data_points()
		index_type get_type_of_feature (index_type)
		void add_data_point(num_type*, index_type, response_type) except +
		void set_type_of_feature (index_type, index_type) except +
		void set_type_of_response (index_type, index_type) except +
		vector[num_type] retrieve_data_point (index_type)

cdef extern from "rfr/data_containers/mostly_continuous_data_with_instances_container.hpp" namespace "rfr::data_containers":
	cdef cppclass mostly_continuous_data_with_instances[num_type, response_type, index_type](data_container_base[num_type, response_type, index_type]):
		mostly_continuous_data_with_instances (index_type, index_type)
		index_type num_features()
		index_type num_data_points()
		index_type num_configurations()
		index_type num_instances()
		index_type get_type_of_feature (index_type)
		void add_data_point(index_type, index_type, response_type) except +
		index_type add_configuration (num_type*, index_type) except +
		index_type add_instance (num_type*, index_type) except +
		void set_type_of_feature (index_type, index_type) except +
		void set_type_of_response (index_type, index_type) except +
		vector[num_type] retrieve_data_point (index_type)
		void check_consistency() except +



cdef extern from "<random>" namespace "std":
	cdef cppclass default_random_engine:
		default_random_engine(int)
		void seed(int)
		

########################
#      Splits          #
########################
cdef extern from "rfr/splits/split_base.hpp" namespace "rfr::splits":
	cdef cppclass k_ary_split_base[k, rng_type, num_type, response_type,index_type]:
		pass

cdef extern from "rfr/splits/binary_split_one_feature_rss_loss.hpp" namespace "rfr::splits":
	cdef cppclass binary_split_one_feature_rss_loss[rng_type, num_type, response_type,index_type](k_ary_split_base[two, rng_type, num_type, response_type,index_type]):
		pass

cdef extern from "rfr/splits/binary_split_one_feature_rss_lossv2.hpp" namespace "rfr::splits":
	cdef cppclass binary_split_one_feature_rss_loss_v2[rng_type, num_type, response_type,index_type](k_ary_split_base[two, rng_type, num_type, response_type,index_type]):
		pass


########################
#       Nodes          #
########################
cdef extern from "rfr/nodes/k_ary_node.hpp" namespace "rfr::nodes":
	cdef cppclass  k_ary_node[k, split_type, rng_type, num_type, response_type, index_type]:
		pass


########################
#       Trees          #
########################
cdef extern from "rfr/trees/tree_options.hpp" namespace "rfr::trees":
	cdef cppclass tree_options[num_type, response_type, index_type]:
		tree_options()
		index_type max_features
		index_type max_depth
		index_type min_samples_to_split
		index_type min_samples_in_leaf
		index_type max_num_nodes
		response_type epsilon_purity
		
cdef extern from "rfr/trees/tree_base.hpp" namespace "rfr::trees":
	cdef cppclass tree_base[rng_type, num_type, response_type, index_type]:
		pass

cdef extern from "rfr/trees/k_ary_tree.hpp" namespace "rfr::trees":
	cdef cppclass k_ary_random_tree[k, split_type, rng_type, num_type, response_type, index_type](tree_base[rng_type, num_type, response_type, index_type]):
		pass


########################
#       Forests        #
########################
cdef extern from "rfr/forests/forest_options.hpp" namespace "rfr::forests":
	cdef cppclass forest_options[num_type, response_type, index_type]:
		forest_options()

		index_type num_trees
		index_type num_data_points_per_tree
		bool do_bootstrapping
		tree_options[num_type,response_type,index_type] tree_opts

cdef extern from "rfr/forests/regression_forest.hpp" namespace "rfr::forests":
	cdef cppclass regression_forest[ tree_type, rng_type, num_type, response_type, index_type]:
		regression_forest()
		regression_forest(forest_options[num_type, response_type, index_type])
		void save_latex_representation(const char*)
		void save_to_binary_file(const string)
		void load_from_binary_file(const string)
		void fit(data_container_base[num_type, response_type, index_type] &data, rng_type &rng)
		pair[num_type, num_type] predict_mean_var( num_type * )
		vector[ vector[num_type] ] all_leaf_values (num_type * )
		forest_options[num_type, response_type, index_type] get_forest_options()
		num_type covariance (num_type*, num_type*)
		string save_into_string()
		void load_from_string( string)
