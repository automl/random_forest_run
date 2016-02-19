#include <cstring>
#include <random>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/forests/regression_forest.hpp"
#include "rfr/forests/forest_options.hpp"



typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

typedef rfr::data_containers::mostly_continuous_data<num_type, response_type, index_type> data_container_type;

typedef rfr::splits::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type;
typedef rfr::nodes::k_ary_node<2, split_type, rng_type, num_type, response_type, index_type> node_type;
typedef rfr::nodes::temporary_node<num_type, index_type> tmp_node_type;

typedef rfr::trees::k_ary_random_tree<2, split_type, rng_type, num_type, response_type, index_type> tree_type;


int main (int argc, char** argv){
    data_container_type data;

    char *filename = (char*) malloc(1024*sizeof(char));

    strcpy(filename, argv[1]);
    strcat(filename, "/hectors_nn_features.csv");
    std::cout<<filename<<"\n";
    data.read_feature_file(filename);

    strcpy(filename, argv[1]);
	strcat(filename, "/hectors_nn_responses.csv");
	std::cout<<filename<<"\n";
    data.read_response_file(filename);
    
    data.set_type_of_feature(21,6);

	std::cout<<data.num_data_points()<<" datapoints with "<<data.num_features()<<" features"<<std::endl;

	rfr::trees::tree_options<num_type, response_type, index_type> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = 14;

	
	rfr::forests::forest_options<num_type, response_type, index_type> forest_opts(tree_opts);

	forest_opts.num_data_points_per_tree = 567;
	forest_opts.num_trees = 256;
	forest_opts.do_bootstrapping = true;

	rfr::forests::regression_forest< tree_type, rng_type, num_type, response_type, index_type> the_forest(forest_opts);
	
	rng_type rng;
	rng.seed(101);

	the_forest.fit(data, rng);
	free(filename);
}
