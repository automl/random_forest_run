// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test
#include <boost/test/unit_test.hpp>


#include <numeric>
#include <cstring>


#include "data_containers/mostly_continuous_data_container.hpp"
#include "splits/binary_split_one_feature_rss_loss.hpp"
#include "nodes/temporary_node.hpp"
#include "nodes/binary_node.hpp"


typedef double num_type;


BOOST_AUTO_TEST_CASE( binary_nodes_tests ){

	rfr::mostly_contiuous_data<num_type> data;
    char filename [1024];

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "diabetes_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "diabetes_responses.csv");
    data.read_response_file(filename);


	
}
