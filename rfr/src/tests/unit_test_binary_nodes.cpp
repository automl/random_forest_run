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
typedef unsigned int my_index_type;


typedef rfr::k_ary_node<2, rfr::binary_split_one_feature_rss_loss<my_num_type, my_index_type>, my_num_type, my_index_type> node_type;
typedef rfr::temporary_node<my_num_type, my_index_type> tmp_node_type;


// Test does not actually check the correctness of the split or anything.
// It makes sure everything compiles and  runs
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
	
	
	std::cout<<"done building first node\n";
	
	std::vector<node_type> nodes;
	nodes.emplace_back();
	
	for (auto n: nodes)
		n.print_info();
	
	std::vector<tmp_node_type> tmp_nodes;
	
	
	std::vector<my_index_type> features_to_try(2, 0);
	std::iota(features_to_try.begin(), features_to_try.end(), 0);
	

	nodes[0].make_internal_node(tmp_node, data, features_to_try, nodes, tmp_nodes);
	nodes[0].print_info();
	
	
	nodes.emplace_back();
	nodes[1].make_leaf_node(tmp_nodes[0]);
	
	
	nodes.emplace_back();
	nodes[2].make_leaf_node(tmp_nodes[1]);	
	
	
	BOOST_REQUIRE(
	
	for (std::vector<tmp_node_type>::iterator tn =  tmp_nodes.begin(); tn != tmp_nodes.end(); tn++){
		tn->print_info();
	}
	
	for (auto n: nodes)
		n.print_info();
	
	
}
