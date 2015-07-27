#ifndef RFR_BINARY_NODES_HPP
#define RFR_BINARY_NODES_HPP

#include <vector>
#include <utility>	// for std::pair

#include "data_containers/data_container_base.hpp"
#include "splits/split_base.hpp"
#include "nodes/temporary_node.hpp"


#include <iostream>



namespace rfr{


template <typename index_type = unsigned int>
struct leaf_node_data{
	std::vector<index_type> data_indices;
};

template <typename split_type, typename num_type = float, typename index_type = unsigned int>
struct internal_node_data{
	index_type left_child_index;
	index_type right_child_index;
	split_type split;
};



template < typename split_type, typename num_type = float, typename index_type = unsigned int>
class binary_node{
  private:
	index_type parent_index;
	bool is_leaf;
	union{
		leaf_node_data<index_type> leaf;
		internal_node_data<split_type, num_type, index_type> internal_node;
	};

  public:

	/** \brief If the temporary node should be split further, this member turns this node into an internal node.
	* 
	*
	* \return A pair of new temporary nodes that will become its children
	*/ 
	std::pair<rfr::temporary_node<num_type, index_type>, rfr::temporary_node<num_type, index_type> >
		make_internal_node(rfr::temporary_node<num_type, index_type> tmp_node, std::vector<index_type> features_to_try){
		
	}
	
	
	/** \brief Member function that turns this node into a leaf node based on a temporary node.
	*
	*  
	*
	* \param tmp_node the internal representation for a temporary node
	* \return The data in a 2d 'array' ready to be used by the data container classes
	*
	*/
	void make_leaf_node(rfr::temporary_node<num_type, index_type> tmp_node){
		
	}

};



}
#endif
