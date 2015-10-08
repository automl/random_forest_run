#ifndef PYRFR_REGRESSION_FOREST_BASE_HPP
#define PYRFR_REGRESSION_FOREST_BASE_HPP


#include <boost/python.hpp>
#include <boost/numpy.hpp>




#include "rfr/forests/forest_options.hpp"
#include "rfr/forests/regression_forest.hpp"
#include "../data_container/container.hpp"



namespace pyrfr{ namespace regression {



template <typename forest_type, typename rng_type, typename num_type, typename response_type, typename index_type>
class regression_forest_base{
  protected:

    rng_type rng;
    forest_type* forest_ptr;


  public:
	// tree parameters
    unsigned int num_trees=10;
    unsigned int seed = 0;
    bool do_bootstrapping = true;
    unsigned int num_data_points_per_tree = 0;
    unsigned int max_num_nodes = 0;
    unsigned int max_depth = 0;
    unsigned int max_features_per_split = 0;
    unsigned int min_samples_to_split = 0;
    unsigned int min_samples_in_leaf = 0;
    response_type epsilon_purity = -1;

	~regression_forest_base(){ delete forest_ptr; forest_ptr=NULL;}


	// construct the forest_option object
	void fit_base(const rfr::data_container_base<num_type, response_type, index_type> &data){

		if (seed > 0) {rng.seed(seed); seed = 0;}

		rfr::forest_options<num_type, response_type, index_type> forest_opts;

		// store all the tree related options
		forest_opts.num_data_points_per_tree = (num_data_points_per_tree > 0)? num_data_points_per_tree : data.num_data_points();
		forest_opts.tree_opts.max_features = (max_features_per_split > 0) ? max_features_per_split : data.num_features();

		if (max_depth > 0) forest_opts.tree_opts.max_depth = max_depth;
		if (min_samples_to_split > 0) forest_opts.tree_opts.min_samples_to_split = min_samples_to_split;
		if (min_samples_in_leaf > 0) forest_opts.tree_opts.min_samples_in_leaf = min_samples_in_leaf;
		if (max_num_nodes >  0) forest_opts.tree_opts.max_num_nodes = max_num_nodes;
		if (epsilon_purity >= 0) forest_opts.tree_opts.epsilon_purity = epsilon_purity;

		// now the forest related options
		forest_opts.num_trees = num_trees;
		forest_opts.do_bootstrapping = do_bootstrapping;
		if (num_data_points_per_tree > 0) forest_opts.num_data_points_per_tree = num_data_points_per_tree;

		delete forest_ptr;
		forest_ptr = new forest_type(forest_opts);

		forest_ptr->fit(data, rng);
	}


	void fit ( pyrfr::data_container::mostly_continuous_data<pyrfr_num_type, pyrfr_response_type_regression, pyrfr_index_type> &data){fit_base(data);}
	void fit ( pyrfr::data_container::numpy_data_container<pyrfr_num_type, pyrfr_response_type_regression, pyrfr_index_type> &data){fit_base(data);}
	void fit ( pyrfr::data_container::numpy_transposed_data_container<pyrfr_num_type, pyrfr_response_type_regression, pyrfr_index_type> &data){fit_base(data);}


    void save_latex_representation(const char * filename_template){
		forest_ptr->save_latex_representation(filename_template);
	}
	
};


}}


#endif
