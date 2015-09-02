// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test
#include <boost/test/unit_test.hpp>

#include <random>


#include "data_containers/mostly_continuous_data_container.hpp"
#include "splits/binary_split_one_feature_rss_loss.hpp"
#include "nodes/temporary_node.hpp"
#include "nodes/k_ary_node.hpp"
#include "trees/tree_options.hpp"
#include "trees/k_ary_tree.hpp"
#include "forests/regression_forest.hpp"
#include "forests/forest_options.hpp"



typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

typedef rfr::mostly_contiuous_data<num_type, response_type, index_type> data_container_type;

typedef rfr::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type;
typedef rfr::k_ary_node<2, split_type, rng_type, num_type, response_type, index_type> node_type;
typedef rfr::temporary_node<num_type, index_type> tmp_node_type;

typedef rfr::k_ary_random_tree<2, split_type, rng_type, num_type, response_type, index_type> tree_type;


BOOST_AUTO_TEST_CASE( data_container_tests ){
	rfr::mostly_contiuous_data<num_type> data;

    char *filename = (char*) malloc(1024*sizeof(char));

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "diabetes_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "diabetes_responses.csv");
    data.read_response_file(filename);

	rfr::forest_options<num_type, response_type, index_type> forest_opts();

	forest_opts.num_data_points_per_tree = 50;
	forest_opts.num_trees = 2;
	forest_opts.do_bootstrapping = true;

	rfr::regression_forest< tree_type, rng_type, num_type, response_type, index_type> the_forest(forest_opts);

    free(filename);
}
