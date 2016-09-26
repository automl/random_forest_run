#ifndef RFR_TEMPORARY_NODE_HPP
#define RFR_TEMPORARY_NODE_HPP

#include <vector>

namespace rfr{ namespace nodes{


template <typename num_t = float, typename index_t = unsigned int>
struct temporary_node{
	index_t node_index;
	index_t parent_index;
	std::vector<index_t> data_indices;
	index_t node_level;

	temporary_node (index_t node_id,
					index_t parent_id,
					index_t node_lvl,
					typename std::vector<index_t>::iterator start,
					typename std::vector<index_t>::iterator end):
		node_index(node_id), parent_index(parent_id), node_level(node_lvl)
	{
		data_indices.assign(start, end);
	}
	
	
	void print_info(){
		std::cout<<"node_index = "<<node_index <<"\n";
		std::cout<<"parent_index = "<<parent_index <<"\n";
		std::cout<<"node_level = "<< node_level<<"\n";
		std::cout<<"data_indices: ";
		print_vector(data_indices);
	}
};

}} // end namespace rfr::nodes
#endif
