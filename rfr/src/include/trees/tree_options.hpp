#ifndef RFR_TREE_OPTIONS_HPP
#define RFR_TREE_OPTIONS_HPP

#include <cmath>
#include <limits>
#include "data_containers/data_container_base.hpp"



namespace rfr{

template <typename num_type = float, typename index_type = unsigned int>
struct tree_options{
    index_type max_features;
    index_type max_depth;
    
    index_type min_samples_to_split;
    index_type min_samples_in_leaf;
    
    index_type max_num_nodes;
    
    num_type epsilon_purity;

    /** Default constructor with no limits on the size of the tree
     * 
     * If nothing is know about the data, this constructor can be used
     * to get a valid setting for the tree_options struct. But beware
     * this setting could lead to a huge tree depending on the amount of
     * data. There is no limit to the size, and nodes are split into pure
     * leafs. For each split, every feature is considered! This not only
     * slows the training down, but also makes this tree deterministic!
     */
    tree_options(){
	max_features =  std::numeric_limits<index_type>::max();
	max_depth = std::numeric_limits<index_type>::max();
	
	min_samples_to_split = 2;
	min_samples_in_leaf = 1;
	
	max_num_nodes = std::numeric_limits<index_type>::max();
	
	epsilon_purity = 1e-10;
    }
    
    /** Constructor that adjusts the number of features considered at each split proportional to the square root of the number of features.
     * 
     */    
    tree_options(rfr::data_container_base &data): tree_options(){
	max_features =  static_cast<int>(std::sqrt(data.num_features()) + 0.5);
    }
    
};

}//namespace rfr
#endif
