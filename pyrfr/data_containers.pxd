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
		index_type get_type_of_feature (index_type)
		bool add_data_point(num_type*, index_type, response_type)
		bool set_type_of_feature (index_type, index_type)
		vector[num_type] retrieve_data_point (index_type index)

cdef extern from "rfr/data_containers/array_wrapper.hpp" namespace "rfr::data_containers":
	cdef cppclass array_data_container[num_type, response_type, index_type](data_container_base[num_type, response_type, index_type]):
		array_data_container(num_type*, response_type*, index_type*, index_type, index_type)
		index_type num_features()
		index_type num_data_points()
		index_type get_type_of_feature (index_type)
		bool add_data_point(num_type*, index_type, response_type)
		bool set_type_of_feature (index_type, index_type)
		vector[num_type] retrieve_data_point (index_type)


cdef extern from "rfr/data_containers/mostly_continuous_data_container.hpp" namespace "rfr::data_containers":
	cdef cppclass mostly_continuous_data[num_type, response_type, index_type](data_container_base[num_type, response_type, index_type]):
		mostly_continuous_data (index_type num_f)
		index_type num_features()
		index_type num_data_points()
		index_type get_type_of_feature (index_type)
		bool add_data_point(num_type*, index_type, response_type)
		bool set_type_of_feature (index_type, index_type)
		vector[num_type] retrieve_data_point (index_type)


