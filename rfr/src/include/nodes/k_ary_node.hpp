#ifndef RFR_BINARY_NODES_HPP
#define RFR_BINARY_NODES_HPP

#include <vector>
#include <deque>
#include <array>
#include <sstream>


#include "data_containers/data_container_base.hpp"
#include "data_containers/data_container_utils.hpp"
#include "nodes/temporary_node.hpp"


#include <iostream>




namespace rfr{



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
	* TODO: Think about taking two iterators for the features to try, so that copying can be avoided 
	* 
	* \param tmp_node a temporary_node struct containing all the important information. It is not changed in this function.
	* \param data a refernce to the data object that is used
	* \param features_to_try vector of allowed features to be used for this split
	* \param nodes reference to vector containing all processed nodes
	* \param tmp_nodes reference to vector containing all temporary nodes that still have to be checked
	*
	*/ 
	void make_internal_node(rfr::temporary_node<num_type, index_type> &tmp_node,
							const rfr::data_container_base<num_type, index_type> &data,
							std::vector<index_type> &features_to_try,
							index_type num_nodes,
							std::deque<rfr::temporary_node<num_type, index_type> > &tmp_nodes){
		is_leaf = false;
		parent_index = tmp_node.parent_index;
		std::array<typename std::vector<index_type>::iterator, k+1> split_indices_it;
		split.find_best_split(data, features_to_try, tmp_node.data_indices, split_indices_it);
	
		// create an empty node, and a tmp node for each child
		for (index_type i = 0; i < k; i++){
			tmp_nodes.emplace_back(num_nodes+i, tmp_node.node_index, tmp_node.node_level+1, split_indices_it[i], split_indices_it[i+1]);
			std::cout<<"..............................\n";
			tmp_nodes.back().print_info();
			std::cout<<"..............................\n";
			children[i] = num_nodes + i;
		}	
	}
	
	/** \brief Member function that turns this node into a leaf node based on a temporary node.
	*
	*
	* \param tmp_node the internal representation for a temporary node. Node that the tmp_node instance is no longer valid after this function has been called!!
	*
	*/
	void make_leaf_node(rfr::temporary_node<num_type, index_type> &tmp_node){
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
	std::array<index_type, k> get_children() {return(children);}
	
	
	
	void print_info(){
		if (is_leaf){
			std::cout<<"status: leaf\nindices: ";
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
	
	void print_info(const rfr::data_container_base<num_type, index_type> &data){
		if (is_leaf){
			std::cout<<"status: leaf\nindices: ";
			for (auto i=0; i< data_indices.size(); i++){
				std::cout<<data_indices[i]<<"("<< data.response(data_indices[i]) <<") ";
			}
			std::cout<<std::endl;
		}
		else{
			std::cout<<"status: internal node\n";
			split.print_info();
			std::cout<<"children: ";
			for (auto i=0; i < k; i++)
				std::cout<<children[i]<<" ";
			std::cout<<std::endl;
		}
	}
	
	std::string latex_representation( int my_index){
		std::stringstream str;
			
		if (is_leaf){
			str << "node [rectangle] { index = " << my_index << "\\\\";			
			for (auto tmp : data_indices){
				str << tmp << "\\\\";
			}
			str << "\b\b}";
			
		}
		else{
			str << "node [circle split] { index = " << my_index << "\\nodepart{lower} {";
			str << split.latex_representation() << "}}";
		}
		return(str.str());
	}
};



}
#endif
