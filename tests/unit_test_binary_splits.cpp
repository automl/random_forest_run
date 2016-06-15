// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test


#include <cstring>
#include <numeric>
#include <vector>
#include <array>
#include <random>

#include <boost/test/unit_test.hpp>

#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>

#include <fstream>


#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"

typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

typedef rfr::data_containers::mostly_continuous_data<num_type, response_type, index_type> data_container_type;

typedef rfr::splits::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type;

BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_continuous_split_test){
	
	char filename[1000];
	
	// read the test dataset
	data_container_type data;
	
	strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);
	

	std::vector<index_type> indices(data.num_data_points());
	std::iota(indices.begin(), indices.end(), 0);

	std::array<std::vector<index_type>::iterator, 3>indices_split_it;
	std::vector<index_type> features_to_try(1,0);

	rng_type rng;

	split_type split1;
	num_type loss = split1.find_best_split(data, features_to_try, indices, indices_split_it, rng);

	// actual loss independently computed in python
	BOOST_REQUIRE_CLOSE(loss, 23.33333333, 1e-4);
	
	// split criterion has to be in [59, 60) -> see python reference
	num_type split_val = split1.get_num_split_value();

	BOOST_REQUIRE(split_val >=59);
	BOOST_REQUIRE(split_val < 60);
	
	// test the () operator for the trainings data
	std::vector<index_type> operator_test = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	
	for (size_t i=0; i<operator_test.size(); i++){
		num_type tmp_feature_vector[] = {data.feature(0,i), data.feature(1,i)};
		BOOST_REQUIRE(split1(tmp_feature_vector) == operator_test[i]);
	}

}


BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_categorical_split_test){
	
	char filename[1000];
	
	// read the test dataset
	data_container_type data;
	strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);
	
	data.set_type_of_feature(1,10);
	rng_type rng;

	std::vector<index_type> indices(data.num_data_points());
	std::iota(indices.begin(), indices.end(), 0);

	std::array<std::vector<index_type>::iterator, 3>indices_split_it;
	std::vector<index_type> features_to_try(1,1);

	split_type split2;
	num_type loss = split2.find_best_split(data, features_to_try, indices, indices_split_it, rng);

	// actual best split and loss independently computed in python
	BOOST_REQUIRE_CLOSE(loss, 88.57142857, 1e-4);
	auto split_set = split2.get_cat_split_set();
	
	
	BOOST_REQUIRE(split_set[1]);
	BOOST_REQUIRE(split_set[2]);
	BOOST_REQUIRE(!split_set[3]);
	
	
	// test the () operator for the trainings data
	std::vector<index_type> operator_test = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	
	for (size_t i=0; i<operator_test.size(); i++){
		num_type tmp_feature_vector[] = {data.feature(0,i), data.feature(1,i)};
		BOOST_CHECK_MESSAGE(split2(tmp_feature_vector) == operator_test[i],split2(tmp_feature_vector) << "!=" <<  operator_test[i]<<" (index "<<i<<")\n");
	}
}


// check if it finds the best split out of the two above
BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_find_best_split_test){
	
	char filename[1000];
	
	// read the test dataset
	data_container_type data;
	strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);
	
	data.set_type_of_feature(1,10);

	rng_type rng;

	std::vector<index_type> indices(data.num_data_points());
	std::iota(indices.begin(), indices.end(), 0);

	std::array<std::vector<index_type>::iterator, 3>indices_split_it;
	std::vector<index_type> features_to_try({0,1});


	split_type split3;
	num_type loss = split3.find_best_split(data, features_to_try, indices, indices_split_it, rng);
	BOOST_REQUIRE_CLOSE(loss, 23.33333333, 1e-4);
	
	num_type split_val = split3.get_num_split_value();

	BOOST_REQUIRE(split_val >=59);
	BOOST_REQUIRE(split_val < 60);
}


// test serialization
BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_serialization){
	char filename[1000];
	
	// read the test dataset
	data_container_type data;
	
	strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);
	
	data.set_type_of_feature(1,10);

	rng_type rng;

	std::vector<index_type> indices(data.num_data_points());
	std::iota(indices.begin(), indices.end(), 0);

	std::array<std::vector<index_type>::iterator, 3>indices_split_it;
	std::vector<index_type> features_to_try({0,1});


	split_type split4;
	split4.find_best_split(data, features_to_try, indices, indices_split_it, rng);
	
	index_type index4 = split4.get_feature_index();
	auto split_val = split4.get_num_split_value();
	auto split_bits= split4.get_cat_split_set();

	{
		std::ofstream ofs("test.xml");
		cereal::XMLOutputArchive oarchive(ofs);
		oarchive(split4);
	}
	
		
	split_type split5;
	{
		std::ifstream ifs("test.xml");
		cereal::XMLInputArchive iarchive(ifs);
		iarchive(split5);
	}
	
	BOOST_REQUIRE(index4 == split5.get_feature_index());
	BOOST_REQUIRE(split_val == split5.get_num_split_value());
	BOOST_REQUIRE(split_bits ==split5.get_cat_split_set());
	
}

// test binary serialization

BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_binary_serialization){
	char filename[1000];
	
	// read the test dataset
	data_container_type data;
	
	strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);
	
	data.set_type_of_feature(1,10);

	rng_type rng;

	std::vector<index_type> indices(data.num_data_points());
	std::iota(indices.begin(), indices.end(), 0);

	std::array<std::vector<index_type>::iterator, 3>indices_split_it;
	std::vector<index_type> features_to_try({0,1});


	split_type split4;
	split4.find_best_split(data, features_to_try, indices, indices_split_it, rng);
	
	index_type index4 = split4.get_feature_index();
	auto split_val = split4.get_num_split_value();
	auto split_bits= split4.get_cat_split_set();

	{
		std::ofstream ofs("test.bin", std::ios::binary);
		cereal::BinaryOutputArchive oarchive(ofs);
		oarchive(split4);
	}
	
		
	split_type split5;
	{
		std::ifstream ifs("test.bin", std::ios::binary);
		cereal::BinaryInputArchive iarchive(ifs);
		iarchive(split5);
	}
	
	BOOST_REQUIRE(index4 == split5.get_feature_index());
	BOOST_REQUIRE(split_val == split5.get_num_split_value());
	BOOST_REQUIRE(split_bits ==split5.get_cat_split_set());
	
}

