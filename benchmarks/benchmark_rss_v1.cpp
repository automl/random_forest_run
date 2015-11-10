// g++ -I../include -O3 -pg -o benchmark_sorting -std=c11 benchmark_sorting.cpp 

#include <numeric>
#include <cstring>
#include <random>
#include <algorithm>
#include <functional>
#include <utility>
#include <ctime>


#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"

#include "rfr/forests/regression_forest.hpp"


typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

typedef rfr::data_containers::mostly_continuous_data<num_type, response_type, index_type> data_container_type;

typedef rfr::splits::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type;
typedef rfr::nodes::k_ary_node<2, split_type, rng_type, num_type, response_type, index_type> node_type;
typedef rfr::nodes::temporary_node<num_type, index_type> tmp_node_type;

typedef rfr::trees::k_ary_random_tree<2, split_type, rng_type, num_type, response_type, index_type> tree_type;

typedef rfr::forests::regression_forest< tree_type, rng_type, num_type, response_type, index_type> tree_type;



int main (int argc, char** argv){

	index_type num_features = atoi(argv[1]);
	index_type num_data_points = atoi(argv[2]);
	index_type sample_size = atoi(argv[3]);
	index_type num_trees = atoi(argv[4]);

	data_type data (num_features);

	clock_t t;
	num_type best_loss;

	std::default_random_engine rng;
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

	t


    return(0);
}
