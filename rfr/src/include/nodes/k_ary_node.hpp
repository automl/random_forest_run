#ifndef RFR_BINARY_NODES_HPP
#define RFR_BINARY_NODES_HPP

#include <vector>
#include <array>

#include "data_containers/data_container_base.hpp"
#include "data_containers/data_container_utils.hpp"
#include "nodes/temporary_node.hpp"


#include <iostream>



namespace rfr{


template <typename index_type = unsigned int>
struct leaf_node_data{

};

template <int k, typename split_type, typename num_type = float, typename index_type = unsigned int>
struct internal_node_data{

};


/** \brief The node class for regular k-ary trees.
 * 
 * In a regular k-ary tree, every node has either zero (a leaf) or exactly k-children (an internal node).
 * In this case, one can try to gain some speed by replacing variable length std::vectors by std::arrays.
 * 
 */

template <int k, typename split_type, typename num_type = float, typename index_type = unsigned int>
class k_ary_node{
  private:
	index_type parent_index;
	bool is_leaf;

	// for leaf nodes
	std::vector<index_type> data_indices;

	// for internal_nodes
	std::array<index_type, k> children;
	split_type split;
	
  public:
  
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
		parent_index = tmp_node.parent_index;
		std::array<typename std::vector<index_type>::iterator, k+1> split_indices_it;
		split.find_best_split(data, features_to_try, tmp_node.data_indices, split_indices_it);
	
		std::cout<<"children.size() = "<<children.size() <<"\n";
		
		// create an empty node, and a tmp node for each child
		for (index_type i = 0; i < k; i++){
			//print_vector(tmp_node.data_indices);
			std::cout<<"nodes.size = "<<nodes.size();
			std::cout<<"\nnode_index =  "<<tmp_node.node_index;
			std::cout<<"\nnode_level = "<<tmp_node.node_level+1;
			std::cout <<"\nsplit_index = "<< (*split_indices_it[i])<<"\n";
			
			std::cout<<"setting up child["<<i<<"]\n";

			tmp_nodes.emplace_back(nodes.size(), tmp_node.node_index, tmp_node.node_level+1, split_indices_it[i], split_indices_it[i+1]);
			
			children[i] = (index_type) nodes.size() + i;
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
		data_indices.swap(tmp_node.data_indices);
	}	
	
	/** \brief Member that returns the index of the child into which the provided sample falls
	 * 
	 * \param sample a feature vector of the appropriate size (not checked!)
	 *
	 * \return index of the child
	 */
	index_type falls_into_child(double * sample){
		return(children[split(sample)]);
	}
	
	
	bool is_a_leaf(){return(is_leaf);}
	index_type parent() {return(parent_index);}
	
	void print_info(){
		if (is_leaf){
			std::cout<<"status: leaf\nindices:\n";
			rfr::print_vector(data_indices);
		}
		else{
			std::cout<<"status: internal node\n";
			std::cout<<"children: ";
			for (auto i=0; i < k; i++)
				std::cout<<children[i]<<" ";
			std::cout<<std::endl;
		}
	}
	
	
};



}
#endif
