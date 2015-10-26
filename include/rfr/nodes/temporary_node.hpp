#ifndef RFR_TEMPORARY_NODE_HPP
#define RFR_TEMPORARY_NODE_HPP

#include <vector>

namespace rfr{ namespace nodes{


template <typename num_type = float, typename index_type = unsigned int>
struct temporary_node{
	index_type node_index;
	index_type parent_index;
	std::vector<index_type> data_indices;
	index_type node_level;

	temporary_node (index_type node_id,
					index_type parent_id,
					index_type node_lvl,
					typename std::vector<index_type>::iterator start,
					typename std::vector<index_type>::iterator end):
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
