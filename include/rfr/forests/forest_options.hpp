#ifndef RFR_FOREST_OPTIONS_HPP
#define RFR_FOREST_OPTIONS_HPP


#include "cereal/cereal.hpp"


#include "rfr/data_containers/data_container.hpp"
#include "rfr/trees/tree_options.hpp"


namespace rfr{ namespace forests{

template <typename num_t = float,typename response_t = float, typename index_t = unsigned int>
struct forest_options{
	index_t num_trees;					///< number of trees in the forest
	index_t num_data_points_per_tree;	///< number of datapoints used in each tree

	bool do_bootstrapping;				///< flag to toggle bootstrapping
	bool compute_oob_error;				///< flag to enable/disable computing the out-of-bag error

	rfr::trees::tree_options<num_t,response_t,index_t> tree_opts;	///< the options for each tree

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( num_trees,num_data_points_per_tree, do_bootstrapping, compute_oob_error, tree_opts);
	}


	/** (Re)set to default values for the forest.
	 * 
	 * 
	 */
	void set_default_values(){
		num_trees = 0;
		num_data_points_per_tree = 0;

		do_bootstrapping = true;
		compute_oob_error = false;

	}


	/** adjusts all relevant variables to the data */
	void adjust_limits_to_data (const rfr::data_containers::base<num_t, response_t, index_t> &data){
		num_data_points_per_tree = data.num_data_points();
	}

	/** Default constructor that initializes the values with their default */
	forest_options(){ set_default_values(); tree_opts.set_default_values();}

	/** Constructor to feed in tree values but leave the forest parameters at their default.*/
	forest_options(rfr::trees::tree_options<num_t,response_t,index_t> & to): tree_opts(to) { set_default_values();}

	/** Constructor that adjusts to the data. */   
	forest_options (rfr::trees::tree_options<num_t,response_t,index_t> & to, rfr::data_containers::base<num_t, response_t, index_t> &data): tree_opts(to){
		set_default_values();
		tree_opts.set_default_values();
		adjust_limits_to_data(data);
	}
};

}}//namespace rfr::forests
#endif
