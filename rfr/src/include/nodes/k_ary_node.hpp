#ifndef RFR_BINARY_NODES_HPP
#define RFR_BINARY_NODES_HPP

#include <vector>
#include <array>

#include "data_containers/data_container_base.hpp"
#include "data_containers/data_container_utils.hpp"
#include "splits/split_base.hpp"
#include "nodes/temporary_node.hpp"


#include <iostream>



namespace rfr{


template <typename index_type = unsigned int>
struct leaf_node_data{
	std::vector<index_type> data_indices;
};

template <int k, typename split_type, typename num_type = float, typename index_type = unsigned int>
struct internal_node_data{
	std::array<index_type, k> children;
	split_type split;
};


template <int k, typename split_type, typename num_type = float, typename index_type = unsigned int>
class k_ary_node{
  private:
	index_type parent_index;
	bool is_leaf;
	union{
		leaf_node_data<index_type> leaf_data;
		internal_node_data<k, split_type, num_type, index_type> internal_data;
	};

  public:

	// standard constructor to initialize the union
	k_ary_node (): parent_index(0), is_leaf(true),leaf_data(){}
	
	k_ary_node (k_ary_node<k, split_type, num_type, index_type> &the_other){
		//parent_index = the_other.parent_index;
		//is_leaf = the_other.is_leaf;
	}
	
	// empty destructor
	~k_ary_node(){}


	/** \brief If the temporary node should be split further, this member turns this node into an internal node.
	* 
	* 
	* \param tmp_node a temporary_node struct containing all the important information
	* \param data a refernce to the data object that is used
	* \param features_to_try vector of allowed features to be used for this split
	* \param nodes reference to vector containing all processed nodes
	* \param tmp_nodes reference to vector containing all temporary nodes that still have to be checked
	*
	*/ 
	void make_internal_node(rfr::temporary_node<num_type, index_type> tmp_node,const rfr::data_container_base<num_type, index_type> &data,  std::vector<index_type> features_to_try, std::vector<rfr::k_ary_node<k,split_type, num_type, index_type> > &nodes, std::vector<rfr::temporary_node<num_type, index_type> > &tmp_nodes){
		is_leaf = false;
		
		std::array<typename std::vector<index_type>::iterator, k+1> split_indices_it;
		internal_data.split.find_best_split(data, features_to_try, tmp_node.data_indices, split_indices_it);
		
		// create an empty node, and a tmp node for each child
		for (index_type i = 0; i <k; i++){
			tmp_nodes.push_back(rfr::temporary_node<num_type, index_type>(nodes.size(), tmp_node.node_index, tmp_node.node_level+1, split_indices_it[i], split_indices_it[i+1]));
			nodes.push_back(rfr::k_ary_node<k,split_type, num_type, index_type>());
		}	
	}
	
	/** \brief Member function that turns this node into a leaf node based on a temporary node.
	*
	*
	* \param tmp_node the internal representation for a temporary node
	*
	*/
	void make_leaf_node(rfr::temporary_node<num_type, index_type> tmp_node){
		is_leaf = true;
		parent_index = tmp_node.parent_index;
		leaf_data.data_indices.swap(tmp_node.data_indices);
	}	
	
	/** \brief Member that returns the index of the child into which the provided sample falls
	 * 
	 * \param sample a feature vector of the appropriate size (not checked!)
	 *
	 * \return index of the child
	 */
	index_type falls_into_child(double * sample){
		return(internal_data.children[internal_data.split(sample)]);
	}
	
	
	bool is_a_leaf(){return(is_leaf);}
	index_type parent() {return(parent_index);}
	
	void print_info(){
		if (is_leaf){
			std::cout<<"status: leaf\nindices:\n";
			rfr::print_vector(leaf_data.data_indices);
		}
		else{
			std::cout<<"status: internal node\n";
			std::cout<<"children: ";
			for (auto i=0; i < k; i++)
				std::cout<<internal_data.children[i]<<" ";
		}
	}
	
	
};



}
#endif
