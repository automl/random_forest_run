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

#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"


typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_type;
typedef rfr::splits::data_info_t<num_t, num_t, index_t> info_t;


typedef rfr::data_containers::default_container<num_t, response_t, index_t> data_container_type;

typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_type> split_type;
typedef rfr::nodes::temporary_node<num_t, response_t, index_t> tmp_node_type;

typedef rfr::nodes::k_ary_node_minimal<2, split_type, num_t, response_t, index_t, rng_type> minimal_node_type;
typedef rfr::nodes::k_ary_node_full<2, split_type, num_t, response_t, index_t, rng_type> full_node_type;



data_container_type load_toy_data(){
	data_container_type data(2);
	
    std::string feature_file, response_file;

    feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_features.csv";
    response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_responses.csv";

    data.import_csv_files(feature_file, response_file);
	
	data.set_type_of_feature(1,10);
    return(data);
}


template <typename node_type>
void test_make_internal_node_and_make_leaf_node(){
	auto data = load_toy_data();
	data.set_type_of_feature(1,0);
	std::vector<info_t > data_info(data.num_data_points());
	BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);


	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].prediction_value = data.response(i);
		data_info[i].weight = 1;
	}

	rng_type rng;


	// create an empty node
	std::vector<node_type> nodes;
	nodes.emplace_back();
	BOOST_REQUIRE_EQUAL(nodes.size(), 1);


	// setup a temporary node
	std::deque<tmp_node_type> tmp_nodes;
	tmp_node_type tmp_node2(0, 0, 0, data_info.begin(), data_info.end());
	tmp_nodes.push_back(tmp_node2);


	std::vector<index_t> features_to_try({0,1});

	// actually split the data and remove the tmp_node
	nodes[0].make_internal_node(tmp_nodes.front(), data, features_to_try, nodes.size(), tmp_nodes, 1, 1, rng);
	tmp_nodes.pop_front();

	// turn the first child into a leaf
	nodes.emplace_back();
	BOOST_REQUIRE_CLOSE(tmp_nodes.front().total_weight(), 60, 1e-4);
	nodes[1].make_leaf_node(tmp_nodes[0], data);
	tmp_nodes.pop_front();

	// turn the second child into a leaf
	nodes.emplace_back();
	BOOST_REQUIRE_EQUAL(nodes.size(), 3);
	BOOST_REQUIRE_CLOSE(tmp_nodes.front().total_weight(), 40, 1e-4);
	nodes[2].make_leaf_node(tmp_nodes[0], data);
	tmp_nodes.pop_front();


	// check is_leaf
	BOOST_REQUIRE(!nodes[0].is_a_leaf());
	BOOST_REQUIRE( nodes[1].is_a_leaf());
	BOOST_REQUIRE( nodes[2].is_a_leaf());


	//check children
	auto children = nodes[0].get_children();
	BOOST_REQUIRE_EQUAL(children[0],1);
	BOOST_REQUIRE_EQUAL(children[1],2);

    BOOST_REQUIRE_EQUAL(nodes[0].get_child_index(0),1);
    BOOST_REQUIRE_EQUAL(nodes[0].get_child_index(1),2);

	children = nodes[1].get_children();
	BOOST_REQUIRE_EQUAL(children[0],0);
	BOOST_REQUIRE_EQUAL(children[1],0);

    BOOST_REQUIRE_EQUAL(nodes[1].get_child_index(0),0);
    BOOST_REQUIRE_EQUAL(nodes[1].get_child_index(1),0);

    BOOST_REQUIRE_EQUAL(nodes[0].get_depth(), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].get_depth(), 1);
    BOOST_REQUIRE_EQUAL(nodes[2].get_depth(), 1);

    BOOST_REQUIRE_EQUAL(nodes[0].parent(), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].parent(), 0);
    BOOST_REQUIRE_EQUAL(nodes[2].parent(), 0);


    //check split_fraction
	auto sf = nodes[0].get_split_fractions();
	BOOST_REQUIRE_CLOSE(sf[0], 0.6,1e-6);
	BOOST_REQUIRE_CLOSE(sf[1], 0.4,1e-6);

    BOOST_REQUIRE_EQUAL(nodes[0].get_num_data(), 100);
    BOOST_REQUIRE_EQUAL(nodes[1].get_num_data(), 60);
    BOOST_REQUIRE_EQUAL(nodes[2].get_num_data(), 40);

    sf = nodes[1].get_split_fractions();
	BOOST_REQUIRE(std::isnan(sf[0]));
	BOOST_REQUIRE(std::isnan(sf[1]));

	// check the split info. Here feature 0 is continuous, so no cat split available
    BOOST_REQUIRE_EQUAL(nodes[0].get_feature_index(), 0);
    BOOST_REQUIRE_CLOSE(nodes[0].get_num_split_value(), 60., 1.);
    BOOST_REQUIRE(nodes[0].get_cat_split().empty());

    BOOST_REQUIRE_EQUAL(nodes[1].get_feature_index(), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].get_num_split_value(), 0);
    BOOST_REQUIRE(nodes[1].get_cat_split().empty());

    BOOST_REQUIRE_EQUAL(nodes[2].get_feature_index(), 0);
    BOOST_REQUIRE_EQUAL(nodes[2].get_num_split_value(),0);
    BOOST_REQUIRE(nodes[2].get_cat_split().empty());

    std::vector<num_t> test_vector_1 {50., 0};
    std::vector<num_t> test_vector_2 {70., 0};

    BOOST_REQUIRE_EQUAL(nodes[0].falls_into_child(test_vector_1), 1);
    BOOST_REQUIRE_EQUAL(nodes[0].falls_into_child(test_vector_2), 2);

    BOOST_REQUIRE_EQUAL(nodes[1].falls_into_child(test_vector_1), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].falls_into_child(test_vector_2), 0);

    BOOST_REQUIRE_EQUAL(nodes[2].falls_into_child(test_vector_1), 0);
    BOOST_REQUIRE_EQUAL(nodes[2].falls_into_child(test_vector_2), 0);


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


	// add and remove an additional sample to a node and check that this works
	nodes[2].push_response_value(1,1);
	nodes[2].pop_response_value(1,1);
	info2 = nodes[2].leaf_statistic();
	BOOST_CHECK_CLOSE(info2.mean(), ((num_t) 7)/2, 1e-10);
	BOOST_CHECK_CLOSE(info2.variance_unbiased_frequency(), ((num_t) 10)/39, 1e-10);
	BOOST_REQUIRE_EQUAL(info2.sum_of_weights(), 40);
	BOOST_REQUIRE(nodes[2].is_a_leaf());


	// test serializability
	{
		std::ofstream ofs("/tmp/rfr_test_binary_nodes.xml");
		cereal::XMLOutputArchive oarchive(ofs);
		oarchive(nodes);
	}


	std::vector<node_type> nodes2;
	{
		std::ifstream ifs("/tmp/rfr_test_binary_nodes.xml");
		cereal::XMLInputArchive iarchive(ifs);
		iarchive(nodes2);
	}

	// just for the coverage :)
	for (auto &n: nodes)
		n.print_info();

	// check the predictions of the leaf nodes (skip root note!)
	for (auto i=1u; i< nodes.size(); ++i){
		auto stat1 = nodes[i].leaf_statistic();
		auto stat2 = nodes2[i].leaf_statistic();

		BOOST_REQUIRE_EQUAL(stat1.mean(), stat2.mean());
		BOOST_REQUIRE(stat1.numerically_equal(stat2, 1e-6));

	}
}

template <typename node_type>
void test_make_internal_node_and_make_leaf_node_differing_values(){
	auto data = load_toy_data();
	data.set_type_of_feature(1,0);
	std::vector<info_t > data_info(data.num_data_points());
	BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);


	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].prediction_value = data.response(i)*10;
		data_info[i].weight = 1;
	}

	rng_type rng;


	// create an empty node
	std::vector<node_type> nodes;
	nodes.emplace_back();
	BOOST_REQUIRE_EQUAL(nodes.size(), 1);


	// setup a temporary node
	std::deque<tmp_node_type> tmp_nodes;
	tmp_node_type tmp_node2(0, 0, 0, data_info.begin(), data_info.end());
	tmp_nodes.push_back(tmp_node2);


	std::vector<index_t> features_to_try({0,1});

	// actually split the data and remove the tmp_node
	nodes[0].make_internal_node(tmp_nodes.front(), data, features_to_try, nodes.size(), tmp_nodes, 1, 1, rng);
	tmp_nodes.pop_front();

	// turn the first child into a leaf
	nodes.emplace_back();
	BOOST_REQUIRE_CLOSE(tmp_nodes.front().total_weight(), 60, 1e-4);
	nodes[1].make_leaf_node(tmp_nodes[0], data);
	tmp_nodes.pop_front();

	// turn the second child into a leaf
	nodes.emplace_back();
	BOOST_REQUIRE_EQUAL(nodes.size(), 3);
	BOOST_REQUIRE_CLOSE(tmp_nodes.front().total_weight(), 40, 1e-4);
	nodes[2].make_leaf_node(tmp_nodes[0], data);
	tmp_nodes.pop_front();


	// check is_leaf
	BOOST_REQUIRE(!nodes[0].is_a_leaf());
	BOOST_REQUIRE( nodes[1].is_a_leaf());
	BOOST_REQUIRE( nodes[2].is_a_leaf());


	//check children
	auto children = nodes[0].get_children();
	BOOST_REQUIRE_EQUAL(children[0],1);
	BOOST_REQUIRE_EQUAL(children[1],2);

    BOOST_REQUIRE_EQUAL(nodes[0].get_child_index(0),1);
    BOOST_REQUIRE_EQUAL(nodes[0].get_child_index(1),2);

	children = nodes[1].get_children();
	BOOST_REQUIRE_EQUAL(children[0],0);
	BOOST_REQUIRE_EQUAL(children[1],0);

    BOOST_REQUIRE_EQUAL(nodes[1].get_child_index(0),0);
    BOOST_REQUIRE_EQUAL(nodes[1].get_child_index(1),0);

    BOOST_REQUIRE_EQUAL(nodes[0].get_depth(), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].get_depth(), 1);
    BOOST_REQUIRE_EQUAL(nodes[2].get_depth(), 1);

    BOOST_REQUIRE_EQUAL(nodes[0].parent(), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].parent(), 0);
    BOOST_REQUIRE_EQUAL(nodes[2].parent(), 0);


	//check split_fraction
	auto sf = nodes[0].get_split_fractions();
	BOOST_REQUIRE_CLOSE(sf[0], 0.6,1e-6);
	BOOST_REQUIRE_CLOSE(sf[1], 0.4,1e-6);

	sf = nodes[1].get_split_fractions();
	BOOST_REQUIRE(std::isnan(sf[0]));
	BOOST_REQUIRE(std::isnan(sf[1]));

    // check the split info. Here feature 0 is continuous, so no cat split available
    BOOST_REQUIRE_EQUAL(nodes[0].get_feature_index(), 0);
    BOOST_REQUIRE_CLOSE(nodes[0].get_num_split_value(), 59.4153952, 1e-6);
    BOOST_REQUIRE(nodes[0].get_cat_split().empty());

    BOOST_REQUIRE_EQUAL(nodes[1].get_feature_index(), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].get_num_split_value(), 0);
    BOOST_REQUIRE(nodes[1].get_cat_split().empty());

    BOOST_REQUIRE_EQUAL(nodes[2].get_feature_index(), 0);
    BOOST_REQUIRE_EQUAL(nodes[2].get_num_split_value(),0);
    BOOST_REQUIRE(nodes[2].get_cat_split().empty());

    std::vector<num_t> test_vector_1 {50., 0};
    std::vector<num_t> test_vector_2 {70., 0};

    BOOST_REQUIRE_EQUAL(nodes[0].falls_into_child(test_vector_1), 1);
    BOOST_REQUIRE_EQUAL(nodes[0].falls_into_child(test_vector_2), 2);

    BOOST_REQUIRE_EQUAL(nodes[1].falls_into_child(test_vector_1), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].falls_into_child(test_vector_2), 0);

    BOOST_REQUIRE_EQUAL(nodes[2].falls_into_child(test_vector_1), 0);
    BOOST_REQUIRE_EQUAL(nodes[2].falls_into_child(test_vector_2), 0);



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
	BOOST_CHECK_CLOSE(info1.mean(), ((num_t) 5)/3 * 10, 1e-10);
	BOOST_CHECK_CLOSE(info1.variance_unbiased_frequency(), ((num_t) 40)/177 * 100, 1e-10);
	BOOST_REQUIRE_EQUAL(info1.sum_of_weights(), 60);
	BOOST_REQUIRE(nodes[1].is_a_leaf());


	auto info2 = nodes[2].leaf_statistic();
	BOOST_CHECK_CLOSE(info2.mean(), ((num_t) 7)/2 * 10, 1e-10);
	BOOST_CHECK_CLOSE(info2.variance_unbiased_frequency(), ((num_t) 10)/39 * 100, 1e-10);
	BOOST_REQUIRE_EQUAL(info2.sum_of_weights(), 40);
	BOOST_REQUIRE(nodes[2].is_a_leaf());


	// add and remove an additional sample to a node and check that this works
	nodes[2].push_response_value(1,1);
	nodes[2].pop_response_value(1,1);
	info2 = nodes[2].leaf_statistic();
	BOOST_CHECK_CLOSE(info2.mean(), ((num_t) 7)/2 * 10, 1e-10);
	BOOST_CHECK_CLOSE(info2.variance_unbiased_frequency(), ((num_t) 10)/39 * 100, 1e-10);
	BOOST_REQUIRE_EQUAL(info2.sum_of_weights(), 40);
	BOOST_REQUIRE(nodes[2].is_a_leaf());


	// test serializability
	{
		std::ofstream ofs("/tmp/rfr_test_binary_nodes.xml");
		cereal::XMLOutputArchive oarchive(ofs);
		oarchive(nodes);
	}


	std::vector<node_type> nodes2;
	{
		std::ifstream ifs("/tmp/rfr_test_binary_nodes.xml");
		cereal::XMLInputArchive iarchive(ifs);
		iarchive(nodes2);
	}

	// just for the coverage :)
	for (auto &n: nodes)
		n.print_info();

	// check the predictions of the leaf nodes (skip root note!)
	for (auto i=1u; i< nodes.size(); ++i){
		auto stat1 = nodes[i].leaf_statistic();
		auto stat2 = nodes2[i].leaf_statistic();

		BOOST_REQUIRE_EQUAL(stat1.mean(), stat2.mean());
		BOOST_REQUIRE(stat1.numerically_equal(stat2, 1e-6));

	}
}


template <typename node_type>
void test_make_internal_node_and_make_leaf_node_categorical_values(){
    auto data = load_toy_data();
    data.set_type_of_feature(1,3);
    std::vector<info_t > data_info(data.num_data_points());
    BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);


    for (auto i=0u; i<data.num_data_points(); ++i){
        data_info[i].index=i;
        data_info[i].response = data.response(i);
        data_info[i].prediction_value = data.response(i)*10;
        data_info[i].weight = 1;
    }

    rng_type rng;


    // create an empty node
    std::vector<node_type> nodes;
    nodes.emplace_back();
    BOOST_REQUIRE_EQUAL(nodes.size(), 1);


    // setup a temporary node
    std::deque<tmp_node_type> tmp_nodes;
    tmp_node_type tmp_node2(0, 0, 0, data_info.begin(), data_info.end());
    tmp_nodes.push_back(tmp_node2);

    std::vector<index_t> features_to_try({1, 1});

    // actually split the data and remove the tmp_node
    nodes[0].make_internal_node(tmp_nodes.front(), data, features_to_try, nodes.size(), tmp_nodes, 1, 1, rng);
    tmp_nodes.pop_front();

    // turn the first child into a leaf
    nodes.emplace_back();

    BOOST_REQUIRE_CLOSE(tmp_nodes.front().total_weight(), 70, 1e-4);
    nodes[1].make_leaf_node(tmp_nodes[0], data);
    tmp_nodes.pop_front();

    // turn the second child into a leaf
    nodes.emplace_back();
    BOOST_REQUIRE_EQUAL(nodes.size(), 3);
    BOOST_REQUIRE_CLOSE(tmp_nodes.front().total_weight(), 30, 1e-4);
    nodes[2].make_leaf_node(tmp_nodes[0], data);
    tmp_nodes.pop_front();

    // check is_leaf
    BOOST_REQUIRE(!nodes[0].is_a_leaf());
    BOOST_REQUIRE( nodes[1].is_a_leaf());
    BOOST_REQUIRE( nodes[2].is_a_leaf());

    //check split_fraction
    auto sf = nodes[0].get_split_fractions();


    BOOST_REQUIRE_CLOSE(sf[0], 0.7,1e-6);
    BOOST_REQUIRE_CLOSE(sf[1], 0.3,1e-6);

    sf = nodes[1].get_split_fractions();
    BOOST_REQUIRE(std::isnan(sf[0]));
    BOOST_REQUIRE(std::isnan(sf[1]));

    auto nums_split = nodes[0].get_num_split_value();
    auto cat_split_value = nodes[0].get_cat_split();
    auto value_index = nodes[0].get_feature_index();

    auto nums_split1 = nodes[1].get_num_split_value();
    auto aplit_value1 = nodes[1].get_cat_split();
    auto value_index1 = nodes[1].get_feature_index();

    auto nums_split2 = nodes[2].get_num_split_value();
    auto aplit_value2 = nodes[2].get_cat_split();
    auto value_index2 = nodes[2].get_feature_index();

    // check the split info. Here feature 0 is continuous, so no cat split available
    BOOST_REQUIRE_EQUAL(nodes[0].get_feature_index(), 1);
    BOOST_REQUIRE(std::isnan(nodes[0].get_num_split_value()));
    BOOST_REQUIRE_EQUAL(cat_split_value.size(), 2);
    BOOST_REQUIRE_CLOSE(cat_split_value[0], 0., 1e-6);
    BOOST_REQUIRE_CLOSE(cat_split_value[1], 1., 1e-6);

    BOOST_REQUIRE_EQUAL(nodes[1].get_feature_index(), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].get_feature_index(), 0);
    BOOST_REQUIRE(nodes[1].get_cat_split().empty());

    BOOST_REQUIRE_EQUAL(nodes[2].get_feature_index(), 0);
    BOOST_REQUIRE_EQUAL(nodes[2].get_feature_index(), 0);
    BOOST_REQUIRE(nodes[2].get_cat_split().empty());

    std::vector<num_t> test_vector_1 {0., 1};
    std::vector<num_t> test_vector_2 {0., 2};

    BOOST_REQUIRE_EQUAL(nodes[0].falls_into_child(test_vector_1), 1);
    BOOST_REQUIRE_EQUAL(nodes[0].falls_into_child(test_vector_2), 2);

    BOOST_REQUIRE_EQUAL(nodes[1].falls_into_child(test_vector_1), 0);
    BOOST_REQUIRE_EQUAL(nodes[1].falls_into_child(test_vector_2), 0);

    BOOST_REQUIRE_EQUAL(nodes[2].falls_into_child(test_vector_1), 0);
    BOOST_REQUIRE_EQUAL(nodes[2].falls_into_child(test_vector_2), 0);
}


BOOST_AUTO_TEST_CASE( minimal_node_tests ){
	test_make_internal_node_and_make_leaf_node<minimal_node_type>();
	test_make_internal_node_and_make_leaf_node_differing_values<minimal_node_type>();
    test_make_internal_node_and_make_leaf_node_categorical_values<minimal_node_type>();
}

BOOST_AUTO_TEST_CASE( full_node_tests ){
	test_make_internal_node_and_make_leaf_node<full_node_type>();
	test_make_internal_node_and_make_leaf_node_differing_values<full_node_type>();
    test_make_internal_node_and_make_leaf_node_categorical_values<minimal_node_type>();
}