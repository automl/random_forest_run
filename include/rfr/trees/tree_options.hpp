#ifndef RFR_TREE_OPTIONS_HPP
#define RFR_TREE_OPTIONS_HPP

#include <cmath>
#include <limits>

#include "cereal/cereal.hpp"


#include "rfr/data_containers/data_container.hpp"

namespace rfr{ namespace trees{

template <typename num_type = float,typename response_type = float, typename index_type = unsigned int>
struct tree_options{
    index_type max_features; 		///< number of features to consider for each split 
	index_type max_depth;			///< maximum depth for the tree
    
    index_type min_samples_to_split;///< minumum number of samples to try splitting
    index_type min_samples_in_leaf;	///< minimum number of samples in a leaf
    num_type   min_weight_in_leaf;	///< minimum total sample weights in a leaf

    index_type max_num_nodes;		///< maxmimum total number of nodes in the tree
    index_type max_num_leaves;		///< maxmimum total number of leaves in the tree
    
    response_type epsilon_purity;	///< minimum difference between two response values to be considered different*/


  	/** serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( max_features, max_depth, min_samples_to_split, min_samples_in_leaf, min_weight_in_leaf, max_num_nodes, epsilon_purity);
	}

    /** (Re)set to default values with no limits on the size of the tree
     * 
     * If nothing is know about the data, this member can be used
     * to get a valid setting for the tree_options struct. But beware
     * this setting could lead to a huge tree depending on the amount of
     * data. There is no limit to the size, and nodes are split into pure
     * leafs. For each split, every feature is considered! This not only
     * slows the training down, but also makes this tree deterministic!
     */
    void set_default_values(){
		max_features =  std::numeric_limits<index_type>::max();
		max_depth = std::numeric_limits<index_type>::max();
	
		min_samples_to_split = 2;
		min_samples_in_leaf = 1;
		min_weight_in_leaf = 1;
	
		max_num_nodes = std::numeric_limits<index_type>::max();
		max_num_leaves = std::numeric_limits<index_type>::max();
	
		epsilon_purity = 1e-10;
    }


    /** Default constructor that initializes the values with their default
     */
    tree_options(){ set_default_values();}
    
    /** Constructor that adjusts the number of features considered at each split proportional to the square root of the number of features.
     * 
     */    
    tree_options (rfr::data_containers::base<num_type, response_type, index_type> &data){
	set_default_values();
	max_features =  static_cast<int>(std::sqrt(data.num_features()) + 0.5);
    }
    
    
    void adjust_limits_to_data (const rfr::data_containers::base<num_type, response_type, index_type> &data){
		max_features = std::min(max_features, data.num_features());
    }
    
};

}}//namespace rfr::trees
#endif
