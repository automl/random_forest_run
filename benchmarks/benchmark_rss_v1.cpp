// g++ -I../include --std=c++11 -O2 -pg -o benchmark_rss_v1 benchmark_rss_v1.cpp 

#include <numeric>
#include <cstring>
#include <random>
#include <algorithm>
#include <functional>
#include <utility>
#include <ctime>


#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/forests/regression_forest.hpp"


typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

typedef rfr::data_containers::mostly_continuous_data<num_type, response_type, index_type> data_type;
typedef rfr::splits::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type;
typedef rfr::nodes::k_ary_node<2, split_type, rng_type, num_type, response_type, index_type> node_type;
typedef rfr::nodes::temporary_node<num_type, index_type> tmp_node_type;
typedef rfr::trees::k_ary_random_tree<2, split_type, rng_type, num_type, response_type, index_type> tree_type;
typedef rfr::forests::regression_forest< tree_type, rng_type, num_type, response_type, index_type> forest_type;



int main (int argc, char** argv){

	if (argc != 5){
		std::cout<<"need arguments: <num features> <num datapoints> <sample size> <num trees>"<<std::endl;
		exit(0);
	}

	index_type num_features = atoi(argv[1]);
	index_type num_data_points = atoi(argv[2]);
	index_type sample_size = atoi(argv[3]);
	index_type num_trees = atoi(argv[4]);

	data_type data (num_features);

	clock_t t;

	rng_type rng;
	std::uniform_real_distribution<response_type> dist1(-1.0,1.0);
	std::uniform_int_distribution<index_type> dist2(0,num_data_points);

	auto random_num = std::bind(dist1, rng);
	auto random_ind = std::bind(dist2, rng);
	
	for (auto i=0u; i < num_data_points; i++){

		num_type feature_vector[num_features];
		std::generate_n(feature_vector, num_features, random_num);
		response_type response = random_num();
		
		data.add_data_point(feature_vector, num_features, response);
	}

	rfr::forests::forest_options<num_type, response_type, index_type> forest_opts;
	forest_opts.adjust_limits_to_data(data);

	forest_opts.num_data_points_per_tree = sample_size;
	forest_opts.do_bootstrapping = true;
	forest_opts.num_trees = num_trees;
	forest_opts.tree_opts.max_features = num_features/2;


	forest_type the_forest(forest_opts);

	the_forest.fit(data, rng);

    return(0);
}
