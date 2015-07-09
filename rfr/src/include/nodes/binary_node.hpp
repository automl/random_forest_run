#ifndef RFR_BINARY_NODES_HPP
#define RFR_BINARY_NODES_HPP

#include <vector>

#include "splits/binary_split_base.hpp"



#include <iostream>
#include "splits/binary_split_one_feature_rss_loss.hpp"
#include "data_containers/mostly_continuous_data_container.hpp"


typedef rfr::binary_split_one_feature_rss_loss<rfr::mostly_contiuous_data<> > split_t;

template <typename index_type = unsigned int>
struct leaf_node_data{
	std::vector<index_type> data_indices;
};

template <class split_type, typename num_type = float, typename index_type = unsigned int>
struct internal_node_data{
	index_type left_child_index;
	index_type right_child_index;
	split_type split;
};



template < class split_type, typename num_type = float, typename index_type = unsigned int>
class binary_node{
  private:
	index_type parent_index;
	bool is_leaf;
	union{
		leaf_node_data<index_type> leaf;
		internal_node_data<split_type> internal_node;
	};
};




int main(){
	std::cout<<sizeof(leaf_node_data<>)<<"\n";
	std::cout<<sizeof(internal_node_data<split_t>)<<"\n";
	std::cout<<sizeof(binary_node<split_t, float, unsigned int >)<<"\n";
}



#endif
