// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test
#include <boost/test/unit_test.hpp>


#include <numeric>
#include <cstring>
#include <vector>
#include <deque>
#include <tuple>



#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"


typedef double num_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

typedef rfr::binary_split_one_feature_rss_loss<rng_type, num_type, index_type> split_type;
typedef rfr::k_ary_node<2, split_type, rng_type, num_type, index_type> node_type;
typedef rfr::temporary_node<num_type, index_type> tmp_node_type;


BOOST_AUTO_TEST_CASE( binary_nodes_tests ){

	rfr::mostly_continuous_data<num_type, index_type> data;
    char filename [1024];

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);

	rng_type rng;

	std::vector<index_type> indices(data.num_data_points());
	std::iota(indices.begin(),indices.end(), 0);

	tmp_node_type tmp_node1(0, -1, 0, indices.begin(), indices.end());

	node_type root_node1;
	
	root_node1.make_leaf_node(tmp_node1, data);
	root_node1.print_info();
	
	
	std::vector<node_type> nodes;
	nodes.emplace_back();
	
	for (auto n: nodes)
		n.print_info();
	
	std::deque<tmp_node_type> tmp_nodes;
	
	tmp_node_type tmp_node2(0, -1, 0, indices.begin(), indices.end());
	tmp_nodes.push_back(tmp_node2);
	
	std::vector<index_type> features_to_try(2, 0);
	std::iota(features_to_try.begin(), features_to_try.end(), 0);
	

	nodes[0].make_internal_node(tmp_nodes.front(), data, features_to_try, nodes.size(), tmp_nodes, rng);
	tmp_nodes.pop_front();
	
	nodes.emplace_back();
	nodes[1].make_leaf_node(tmp_nodes[0], data);
	tmp_nodes.pop_front();
	
	nodes.emplace_back();
	nodes[2].make_leaf_node(tmp_nodes[0], data);
	tmp_nodes.pop_front();
	

	// check calculation of mean and variance

	// first an internal node
	auto info0 = nodes[0].mean_variance_N();
	BOOST_CHECK(isnan(std::get<0>(info0)));
	BOOST_CHECK(isnan(std::get<1>(info0)));
	BOOST_CHECK(std::get<2>(info0) == 0);

	// now the two leaves:
	auto info1 = nodes[1].mean_variance_N();
	BOOST_CHECK_CLOSE(std::get<0>(info1), ((num_type) 5)/3, 1e-10);
	BOOST_CHECK_CLOSE(std::get<1>(info1), ((num_type) 2)/9, 1e-10);
	BOOST_CHECK(std::get<2>(info1) == 60);

	auto info2 = nodes[2].mean_variance_N();
	BOOST_CHECK_CLOSE(std::get<0>(info2), ((num_type) 7)/2, 1e-10);
	BOOST_CHECK_CLOSE(std::get<1>(info2), ((num_type) 1)/4, 1e-10);
	BOOST_CHECK(std::get<2>(info2) == 40);
}
