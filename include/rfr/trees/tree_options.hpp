#ifndef RFR_TREE_OPTIONS_HPP
#define RFR_TREE_OPTIONS_HPP

#include <cmath>
#include <limits>

#include "cereal/cereal.hpp"


#include "rfr/data_containers/data_container.hpp"

namespace rfr{ namespace trees{

template <typename num_t = float,typename response_t = float, typename index_t = unsigned int>
struct tree_options{
  index_t	max_features; 			///< number of features to consider for each split 
  index_t	max_depth;				///< maximum depth for the tree
    
  index_t	min_samples_to_split;	///< minumum number of samples to try splitting
  num_t 	min_weight_to_split;	///< minumum weight of samples to try splitting

  index_t	min_samples_in_leaf;	///< minimum total sample weights in a leaf
  num_t	min_weight_in_leaf;		///< minimum total sample weights in a leaf

  index_t	max_num_nodes;			///< maxmimum total number of nodes in the tree
  index_t	max_num_leaves;			///< maxmimum total number of leaves in the tree
    
  response_t epsilon_purity;		///< minimum difference between two response values to be considered different*/

  num_t life_time; ///< life time of a mondrian tree
  bool hierarchical_smoothing;		///< flag to enable/disable hierachical smoothing for mondrian forests


  /** serialize function for saving forests */
  template<class Archive>
	void serialize(Archive & archive)
	{
		archive( max_features, max_depth, min_samples_to_split, min_weight_to_split, min_samples_in_leaf, min_weight_in_leaf, max_num_nodes, epsilon_purity, /*min_samples_node,*/ life_time, hierarchical_smoothing);
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
    max_features =  std::numeric_limits<index_t>::max();
    max_depth = std::numeric_limits<index_t>::max();

    min_samples_to_split = 2;
    min_samples_in_leaf = 1;
    min_weight_to_split = 2;
    min_weight_in_leaf = 1;
  
    max_num_nodes = std::numeric_limits<index_t>::max();
    max_num_leaves = std::numeric_limits<index_t>::max();
  
    epsilon_purity = 1e-10;
    
    life_time = 1000;
    hierarchical_smoothing = false;
  }


  /** Default constructor that initializes the values with their default
    */
  tree_options(){ set_default_values();}
  
  /** Constructor that adjusts the number of features considered at each split proportional to the square root of the number of features.
    * 
    */    
  tree_options (rfr::data_containers::base<num_t, response_t, index_t> &data){
    set_default_values();
    max_features =  static_cast<int>(std::sqrt(data.num_features()) + 0.5);
  }
    
    
  void adjust_limits_to_data (const rfr::data_containers::base<num_t, response_t, index_t> &data){
		max_features = std::min(max_features, data.num_features());
  }


  void print_info(){
		std::cout<<"max_features        : "<< max_features <<std::endl;
		std::cout<<"max_depth           : "<< max_depth <<std::endl;
		std::cout<<"min_samples_to_split: "<< min_samples_to_split <<std::endl;
		std::cout<<"min_weight_to_split : "<< min_weight_to_split <<std::endl;
		std::cout<<"min_samples_in_leaf : "<< min_samples_in_leaf <<std::endl;
		std::cout<<"min_weight_in_leaf  : "<< min_weight_in_leaf <<std::endl;
		std::cout<<"max_num_nodes       : "<< max_num_nodes <<std::endl;
		std::cout<<"max_num_leaves      : "<< max_num_leaves <<std::endl;
    std::cout<<"epsilon_purity      : "<< epsilon_purity <<std::endl;
    std::cout<<"life_time           : "<< life_time <<std::endl;
    std::cout<<"hierarchical_smoothing: "<< hierarchical_smoothing <<std::endl;
	}
    
};

}}//namespace rfr::trees
#endif
