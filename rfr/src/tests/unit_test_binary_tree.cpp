// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test
#include <boost/test/unit_test.hpp>

#include <random>
#include <numeric>
#include <cstring>


#include "data_containers/mostly_continuous_data_container.hpp"
#include "splits/binary_split_one_feature_rss_loss.hpp"
#include "nodes/temporary_node.hpp"
#include "nodes/k_ary_node.hpp"
#include "trees/tree_options.hpp"
#include "trees/k_ary_tree.hpp"

typedef double my_num_type;
typedef unsigned int my_index_type;
typedef std::default_random_engine rng_type;
typedef  rfr::binary_split_one_feature_rss_loss<my_num_type, my_index_type> my_split_type;


typedef rfr::k_ary_node<2, my_split_type, my_num_type, my_index_type> node_type;
typedef rfr::temporary_node<my_num_type, my_index_type> tmp_node_type;

typedef rfr::k_ary_random_tree<2, my_split_type, rng_type, my_num_type, my_index_type> tree_type;

// Test does not actually check the correctness of the split or anything.
// It makes sure everything compiles and  runs
BOOST_AUTO_TEST_CASE( binary_tree_test ){

    rfr::mostly_contiuous_data<my_num_type, my_index_type> data;
    char filename [1024];

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);


    data.set_type_of_feature(1, 3);
    
    rfr::tree_options<my_num_type, my_index_type> tree_opts;
	
	
    tree_opts.max_features = 1;
    tree_opts.max_depth = 3;
	
    rng_type rng_engine;

    for (auto i = 0; i <4; i++){
	tree_type the_tree(&rng_engine);
	the_tree.fit(data, tree_opts);
	
	char filename[100];
	sprintf(filename, "/tmp/tree_%i.tex", i);
	the_tree.save_latex_representation(filename);
    }
	

	
	
	
}
