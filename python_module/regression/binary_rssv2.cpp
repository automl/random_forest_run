#include <iostream>
#include <random>
#include <memory>


#include <boost/python.hpp>
#include <boost/numpy.hpp>


#include "../data_container/mostly_continuous.cpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/forests/regression_forest.hpp"
#include "rfr/forests/forest_options.hpp"


namespace pyrfr{ namespace regression { namespace binary_rss{


typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

//typedef rfr::numpy_transposed_data_container<num_type, response_type, index_type> data_container_type;

typedef rfr::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type_v2;
typedef rfr::k_ary_random_tree<2, split_type_v2, rng_type, num_type, response_type, index_type> tree_type_v2;
typedef rfr::regression_forest< tree_type_v2, rng_type, num_type, response_type, index_type> forest_type_v2;


class binary_rss_v2{
  private:
    

    std::default_random_engine rng;
    forest_type_v2* forest_ptr;


  public:

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



    ~binary_rss_v2(){ delete forest_ptr;}
    
    void fit (pyrfr::data_container::mostly_continuous_data &data){


	if (seed > 0) {rng.seed(seed); seed = 0;}

	// construct the forest_option object
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
	forest_ptr = new forest_type_v2(forest_opts);

	forest_ptr->fit(data, rng);
    }


    boost::numpy::ndarray predict(boost::numpy::ndarray const & data_point){
	check_array<num_type>(data_point, 1);
	std::tuple< num_type, num_type > res = forest_ptr->predict_mean_std(reinterpret_cast<num_type*>(data_point.get_data()));

	boost::python::object tmp = boost::python::make_tuple(std::get<0>(res), std::get<1>(res));

	return(boost::numpy::array(tmp));
	
    }


    void save_latex_representation(const char * filename_template){
	forest_ptr->save_latex_representation(filename_template);
    }

    
};


}}}// of namespace pyrfr::regression::binary_rss
