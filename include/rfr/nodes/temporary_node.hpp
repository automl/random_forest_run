#ifndef RFR_TEMPORARY_NODE_HPP
#define RFR_TEMPORARY_NODE_HPP

#include <vector>

#include <rfr/splits/split_base.hpp>

namespace rfr{ namespace nodes{

template <typename num_t = float, typename response_t = float, typename index_t = unsigned int>
struct temporary_node{

	typedef rfr::splits::data_info_t<num_t, response_t, index_t> info_t;
	
	index_t node_index;
	index_t parent_index;
	typename std::vector<info_t>::iterator start, end; 
	index_t node_level;

	temporary_node (index_t node_id, index_t parent_id,	index_t node_lvl,
					typename std::vector<info_t>::iterator s,
					typename std::vector<info_t>::iterator e):
						node_index(node_id), parent_index(parent_id), start(s), end(s), node_level(node_lvl)
						{}
	
	void print_info(){
		std::cout<<"node_index = "<<node_index <<"\n";
		std::cout<<"parent_index = "<<parent_index <<"\n";
		std::cout<<"node_level = "<< node_level<<"\n";
		std::cout<<"data_indices: ";
        for (auto &it = start; it!=end; it++)
            std::cout<<(*it)->index;
        std::cout<<std::endl;
	}
};

}} // end namespace rfr::nodes
#endif
