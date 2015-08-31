#ifndef RFR_BINARY_NODES_HPP
#define RFR_BINARY_NODES_HPP

#include <vector>
#include <deque>
#include <array>
#include <tuple>
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
template <int k, typename split_type, typename rng_type, typename num_type = float, typename index_type = unsigned int>
class k_ary_node{
  private:
	index_type parent_index;
	bool is_leaf;

	// for leaf nodes
	std::vector<num_type> response_values;

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
							std::deque<rfr::temporary_node<num_type, index_type> > &tmp_nodes,
							rng_type &rng){
		is_leaf = false;
		response_values.clear();
		parent_index = tmp_node.parent_index;
		std::array<typename std::vector<index_type>::iterator, k+1> split_indices_it;
		split.find_best_split(data, features_to_try, tmp_node.data_indices, split_indices_it,rng);
	
		// create an empty node, and a tmp node for each child
		for (index_type i = 0; i < k; i++){
			tmp_nodes.emplace_back(num_nodes+i, tmp_node.node_index, tmp_node.node_level+1, split_indices_it[i], split_indices_it[i+1]);
			children[i] = num_nodes + i;
		}	
	}
	
	/** \brief  turns this node into a leaf node based on a temporary node.
	*
	*
	* \param tmp_node the internal representation for a temporary node. Node that the tmp_node instance is no longer valid after this function has been called!!
	*
	*/
	void make_leaf_node(rfr::temporary_node<num_type, index_type> &tmp_node,
						const rfr::data_container_base<num_type, index_type> &data){
		is_leaf = true;
		parent_index = tmp_node.parent_index;
		
		response_values.resize(tmp_node.data_indices.size());
		for (size_t i = 0; i < tmp_node.data_indices.size(); i++){
			response_values[i] = data.response(tmp_node.data_indices[i]);
		}
		std::sort(response_values.begin(), response_values.end());
	}	

	
	/** \brief returns the index of the child into which the provided sample falls
	 * 
	 * \param sample a feature vector of the appropriate size (not checked!)
	 *
	 * \return index_type index of the child
	 */
	index_type falls_into_child(num_type * sample){
		// could be removed if performance issues arise here, but that's not very likely
		if (is_leaf)
			return(std::numeric_limits<index_type>::quiet_NaN());
		return(children[split(sample)]);
	}
	

	/** \brief calculate the mean of all response values in this leaf
	 *
	 * \return num_type the mean, or NaN if the node is not a leaf
	 */
	num_type mean(){
		if (! is_leaf)
			return(std::numeric_limits<num_type>::quiet_NaN());
		
		num_type m = 0;
		for (auto v : response_values){
			m += v;
		}
		return(m/((num_type) response_values.size()));
	}


	std::tuple<num_type, num_type, index_type> mean_variance_N (){
		if (! is_leaf)
			return(std::tuple<num_type, num_type, index_type>(	std::numeric_limits<num_type>::quiet_NaN(),
																std::numeric_limits<num_type>::quiet_NaN(),
																0));
		
		num_type sum = 0;
		num_type sum_squared = 0;
		index_type N = response_values.size();
		for (auto v : response_values){
			sum += v;
			sum_squared += v*v;	
		}
		return( std::tuple<num_type, num_type, index_type> (sum/N,
															sum_squared/N - (sum/N)*(sum/N),
															N));
	}




	/** \brief to test whether this node is a leaf */
	bool is_a_leaf(){return(is_leaf);}
	/** \brief get the index of the node's parent */
	index_type parent() {return(parent_index);}
	/** \brief get indices of all children*/
	std::array<index_type, k> get_children() {return(children);}
	/** \brief get reference to the response values*/	
	const std::vector<num_type> & responses(){ return(response_values);}


	/** \brief prints out some basic information abouth the node*/
	void print_info(){
		if (is_leaf){
			std::cout<<"status: leaf\nresponse values: ";
			rfr::print_vector(response_values);
		}
		else{
			std::cout<<"status: internal node\n";
			std::cout<<"children: ";
			for (auto i=0; i < k; i++)
				std::cout<<children[i]<<" ";
			std::cout<<std::endl;
		}
	}


	/** \brief generates a label for the node to be used in the LaTeX visualization*/
	std::string latex_representation( int my_index){
		std::stringstream str;
			
		if (is_leaf){
			str << "{i = " << my_index << ": \\begin{tiny}"<<response_values[0];			
			for (size_t i=1; i<response_values.size(); i++){
				str << "," << response_values[i];
			}
			str << "\\end{tiny}}";
			
		}
		else{
			str << "{ i = " << my_index << "\\nodepart{two} {";
			str << split.latex_representation() << "}},rectangle split,rectangle split parts=2,draw";
			
		}
		return(str.str());
	}
};



}
#endif
