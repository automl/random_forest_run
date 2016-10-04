#include <boost/test/unit_test.hpp>

#include <numeric>
#include <cstring>
#include <vector>
#include <deque>
#include <tuple>


#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <fstream>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss_v2.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"


typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_type;
typedef rfr::splits::data_info_t<num_t, num_t, index_t> info_t;


typedef rfr::data_containers::mostly_continuous_data<num_t, response_t, index_t> data_container_type;

typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_type> split_type;
typedef rfr::nodes::k_ary_node<2, split_type, num_t, response_t, index_t, rng_type> node_type;
typedef rfr::nodes::temporary_node<num_t, response_t, index_t> tmp_node_type;


BOOST_AUTO_TEST_CASE( binary_nodes_make_leaf ){

	data_container_type data;
    char filename [1024];

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);

	std::vector<info_t > data_info(data.num_data_points());
	BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);
    
    
	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].weight = 1;

	}
  
    rng_type rng;
    
	tmp_node_type tmp_node1(0, 0, 0, data_info.begin(), data_info.end());
    
	node_type root_node1;
	
	root_node1.make_leaf_node(tmp_node1, data);
	
    
    auto infor = root_node1.leaf_statistic();
    BOOST_REQUIRE(root_node1.is_a_leaf());
    BOOST_REQUIRE_EQUAL(infor.sum_of_weights(), 100);
	
    
    
}

BOOST_AUTO_TEST_CASE( binary_nodes_make_internal_node ){

	data_container_type data;
    char filename [1024];

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);

	std::vector<info_t > data_info(data.num_data_points());
	BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);
    
    
	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].weight = 1;
	}
  
    rng_type rng;


	std::vector<node_type> nodes;
	nodes.emplace_back();
	
	for (auto n: nodes)
		n.print_info();
	
	std::deque<tmp_node_type> tmp_nodes;
	
	tmp_node_type tmp_node2(0, 0, 0, data_info.begin(), data_info.end());
	tmp_nodes.push_back(tmp_node2);
	
	std::vector<index_t> features_to_try({0,1});	

    
    BOOST_REQUIRE_EQUAL(nodes.size(), 1);
	nodes[0].make_internal_node(tmp_nodes.front(), data, features_to_try, nodes.size(), tmp_nodes, rng);
	tmp_nodes.pop_front();
	
	nodes.emplace_back();
    BOOST_REQUIRE_EQUAL(nodes.size(), 2);
	nodes[1].make_leaf_node(tmp_nodes[0], data);
	tmp_nodes.pop_front();
	
	nodes.emplace_back();
    BOOST_REQUIRE_EQUAL(nodes.size(), 3);
	nodes[2].make_leaf_node(tmp_nodes[0], data);
	tmp_nodes.pop_front();
	

	// check calculation of mean and variance

	// first an internal node
	auto info0 = nodes[0].leaf_statistic();
	BOOST_CHECK(std::isnan(info0.mean()));
	BOOST_CHECK(std::isnan(info0.variance_population()));
	BOOST_REQUIRE_EQUAL(info0.sum_of_weights(), 0);
    BOOST_REQUIRE(!(nodes[0].is_a_leaf()));
    
    BOOST_REQUIRE_EQUAL(nodes[0].get_split_fraction(0), 0.6);
    BOOST_REQUIRE_EQUAL(nodes[0].get_split_fraction(1), 0.4);
    
    
	// now the two leaves:
    auto info1 = nodes[1].leaf_statistic();
	BOOST_CHECK_CLOSE(info1.mean(), ((num_t) 5)/3, 1e-10);
	BOOST_CHECK_CLOSE(info1.variance_unbiased_frequency(), ((num_t) 40)/177, 1e-10);
    BOOST_REQUIRE_EQUAL(info1.sum_of_weights(), 60);
    BOOST_REQUIRE(nodes[1].is_a_leaf());   
    

    auto info2 = nodes[2].leaf_statistic();
	BOOST_CHECK_CLOSE(info2.mean(), ((num_t) 7)/2, 1e-10);
	BOOST_CHECK_CLOSE(info2.variance_unbiased_frequency(), ((num_t) 10)/39, 1e-10);
	BOOST_REQUIRE_EQUAL(info2.sum_of_weights(), 40);
	BOOST_REQUIRE(nodes[2].is_a_leaf());
	
	
	{
		std::ofstream ofs("test_binary_nodes.xml");
		cereal::XMLOutputArchive oarchive(ofs);
		oarchive(nodes);
	}
	
		
	std::vector<node_type> nodes2;
	{
		std::ifstream ifs("test_binary_nodes.xml");
		cereal::XMLInputArchive iarchive(ifs);
		iarchive(nodes2);
	}
}
