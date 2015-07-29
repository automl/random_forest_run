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
#include "nodes/k_ary_node.hpp"


typedef double my_num_type;
typedef unsigned short my_index_type;


typedef rfr::k_ary_node<2, rfr::binary_split_one_feature_rss_loss<my_num_type, my_index_type>, my_num_type, my_index_type> node_type;
typedef rfr::temporary_node<my_num_type, my_index_type> tmp_node_type;

BOOST_AUTO_TEST_CASE( binary_nodes_tests ){

	rfr::mostly_contiuous_data<my_num_type, my_index_type> data;
    char filename [1024];

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);


	std::vector<my_index_type> indices(data.num_data_points());
	std::iota(indices.begin(),indices.end(), 0);

	tmp_node_type tmp_node(0, -1, 0, indices.begin(), indices.end());


	node_type root_node1;
	
	
	root_node1.make_leaf_node(tmp_node);
	root_node1.print_info();
	
	
	std::vector<node_type> nodes (1);
	std::vector<tmp_node_type> tmp_nodes(0);
	
	
	std::vector<my_index_type> features_to_try = {0,1};
	
	
	node_type root_node2;	
	root_node2.make_internal_node(tmp_node, data, features_to_try, nodes, tmp_nodes);
	root_node2.print_info();
	
	
}
