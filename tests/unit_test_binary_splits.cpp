#include <cstring>
#include <numeric>
#include <vector>
#include <array>
#include <random>

#include <boost/test/unit_test.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/bitset.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <sstream>

#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"

typedef double num_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_type;

typedef rfr::data_containers::default_container<num_t, num_t, index_t> data_container_type;
typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, num_t, index_t,rng_type,128> split_type;
typedef rfr::splits::data_info_t<num_t, num_t, index_t> info_t;


template <class T>
void print_vector (T v){
	for (auto e : v)
		std::cout<<e<<", ";
	std::cout<<"\b\b\n";
}


void print_pcs (std::vector<std::vector<num_t> > pcs){
	for (auto i: pcs){
		print_vector(i);
	}
}


data_container_type load_toy_data(){
	data_container_type data(2);
	
    std::string feature_file, response_file;
    
    feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_features.csv";
    response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_responses.csv";

    data.import_csv_files(feature_file, response_file);
	
	data.set_type_of_feature(1,10);

	BOOST_REQUIRE_EQUAL(data.get_type_of_feature(1), 10);
	
    return(data);
}





BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_continuous_split_test){
	
	auto data = load_toy_data();
	
    std::vector<info_t > data_info(data.num_data_points());
	
	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].weight = 1;

	}

	std::array<std::vector<info_t>::iterator, 3> infos_split_it;
	std::vector<index_t> features_to_try(1,0);

	rng_type rng;

	split_type split1;
	num_t loss = split1.find_best_split(data, features_to_try,data_info.begin(), data_info.end(),infos_split_it,1, 1, rng);

	// actual loss independently computed in python
	BOOST_REQUIRE_CLOSE(loss, 23.33333333, 1e-4);
	
	// split criterion has to be in [59, 60) -> see python reference
	num_t split_val = split1.get_num_split_value();

	BOOST_REQUIRE(split_val >=59);
	BOOST_REQUIRE(split_val < 60);
	
	// test the () operator for the trainings data
	std::vector<index_t> operator_test = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	
	for (size_t i=0; i<operator_test.size(); i++){
		std::vector<num_t> tmp_feature_vector ({data.feature(0,i), data.feature(1,i)});
		BOOST_REQUIRE(split1(tmp_feature_vector) == operator_test[i]);
	}
	
	std::vector<std::vector<num_t> > pcs = { {-1000, 1000}, {0,1,2,3,4,5,6,7,8,9}};
	
	auto pcss = split1.compute_subspaces(pcs);
	
	BOOST_REQUIRE(std::find(pcss[0][1].begin(), pcss[0][1].end(), 0) != pcss[0][1].end());


	BOOST_CHECK_EQUAL(pcss[0][0][0], pcs[0][0]);
	BOOST_CHECK_EQUAL(pcss[0][0][1], split_val);
	BOOST_CHECK_EQUAL(pcss[1][0][0], split_val);
	BOOST_CHECK_EQUAL(pcss[1][0][1], pcs[0][1]);


	// make sure that x2 has not been altered
	BOOST_CHECK_EQUAL_COLLECTIONS( pcss[0][1].begin(), pcss[0][1].end(),
									pcs[1].begin(), pcs[1].end());
	
	BOOST_CHECK_EQUAL_COLLECTIONS( pcss[1][1].begin(), pcss[1][1].end(),
									pcs[1].begin(), pcs[1].end());


	split1.print_info();
	
}


BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_categorical_split_test){
	
	auto data = load_toy_data();
	
    
    std::vector<info_t > data_info(data.num_data_points());
	
	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].weight = 1;

	}

	std::array<std::vector<info_t>::iterator, 3> infos_split_it;
	std::vector<index_t> features_to_try(1,1);

	rng_type rng;

	split_type split2;
	num_t loss = split2.find_best_split(data, features_to_try,data_info.begin(), data_info.end(),infos_split_it, 1, 1, rng);

    num_t total_weight = 0;
    for (auto it = infos_split_it[0]; it!=infos_split_it[1]; ++it)
        total_weight += (*it).weight;
    for (auto it = infos_split_it[1]; it!=infos_split_it[2]; ++it)
        total_weight += (*it).weight;
    BOOST_REQUIRE_CLOSE(total_weight, data.num_data_points(), 1e-4);    
    
	// actual best split and loss independently computed in python
	BOOST_REQUIRE_CLOSE(loss, 88.57142857, 1e-6);
	auto split_set = split2.get_cat_split_set();
	std::cout<< split2.get_num_split_value()<<std::endl;
	std::cout<< split_set<<std::endl;
	BOOST_REQUIRE(split_set[0]);
	BOOST_REQUIRE(split_set[1]);
	BOOST_REQUIRE(!split_set[2]);
	
	
	// test the () operator for the trainings data
	std::vector<index_t> operator_test = {0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
	

	// test the pcs splitting
	for (size_t i=0; i<operator_test.size(); i++){
		std::vector<num_t> tmp_feature_vector ({data.feature(0,i), data.feature(1,i)});
		BOOST_CHECK_MESSAGE(split2(tmp_feature_vector) == operator_test[i],split2(tmp_feature_vector) << "!=" <<  operator_test[i]<<" (index "<<i<<")\n");
	}
	
	std::vector<std::vector<num_t> > pcs = { {-1000, 1000}, {0,1,2,3,4,5,6,7,8,9}};
	
	auto pcss = split2.compute_subspaces(pcs);
	
	BOOST_REQUIRE( pcss[0][0][0] == pcs[0][0]);BOOST_REQUIRE( pcss[0][0][1] == pcs[0][1]);
	BOOST_REQUIRE( pcss[1][0][0] == pcs[0][0]);BOOST_REQUIRE( pcss[0][0][1] == pcs[0][1]);
	
	BOOST_REQUIRE(std::find(pcss[0][1].begin(), pcss[0][1].end(), 0) != pcss[0][1].end());
	BOOST_REQUIRE(std::find(pcss[0][1].begin(), pcss[0][1].end(), 1) != pcss[0][1].end());
	BOOST_REQUIRE(std::find(pcss[0][1].begin(), pcss[0][1].end(), 2) == pcss[0][1].end());
	
	BOOST_REQUIRE(std::find(pcss[1][1].begin(), pcss[1][1].end(), 0) == pcss[1][1].end());
	BOOST_REQUIRE(std::find(pcss[1][1].begin(), pcss[1][1].end(), 1) == pcss[1][1].end());
	BOOST_REQUIRE(std::find(pcss[1][1].begin(), pcss[1][1].end(), 2) != pcss[1][1].end());


	// test the can_be_split_function
	
	std::vector<num_t> eevee (data.num_features(), 0);

	BOOST_REQUIRE(split2.can_be_split(eevee));

	eevee[split2.get_feature_index()] = NAN;

	BOOST_REQUIRE(!split2.can_be_split(eevee));

	split2.print_info();
	
}


// check if it finds the best split out of the two above
BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_find_best_split_test){

	auto data = load_toy_data();

    std::vector<info_t > data_info(data.num_data_points());    
	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].weight = 1;
	}

	std::array<std::vector<info_t>::iterator, 3> infos_split_it;
	std::vector<index_t> features_to_try({0,1});

	rng_type rng;

	split_type split3;
	num_t loss = split3.find_best_split(data, features_to_try,data_info.begin(), data_info.end(),infos_split_it, 1, 1, rng);
	BOOST_REQUIRE_CLOSE(loss, 23.33333333, 1e-4);
    
    num_t total_weight = 0;
    for (auto it = infos_split_it[0]; it!=infos_split_it[1]; ++it)
        total_weight += (*it).weight;
    for (auto it = infos_split_it[1]; it!=infos_split_it[2]; ++it)
        total_weight += (*it).weight;
    BOOST_REQUIRE_CLOSE(total_weight, data.num_data_points(), 1e-4);
    
    

	
	num_t split_val = split3.get_num_split_value();

	BOOST_REQUIRE(split_val >=59);
	BOOST_REQUIRE(split_val < 60);
}



// test serialization
BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_serialization){

    auto data = load_toy_data();
    
	std::vector<info_t > data_info(data.num_data_points());
	
	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].weight = 1;

	}

	std::array<std::vector<info_t>::iterator, 3> infos_split_it;
	std::vector<index_t> features_to_try({0,1});

	rng_type rng;


	split_type split4;
	num_t loss = split4.find_best_split(data, features_to_try,data_info.begin(), data_info.end(),infos_split_it, 1, 1, rng);

    num_t total_weight = 0;
    for (auto it = infos_split_it[0]; it!=infos_split_it[1]; ++it)
        total_weight += (*it).weight;
    for (auto it = infos_split_it[1]; it!=infos_split_it[2]; ++it)
        total_weight += (*it).weight;
    BOOST_REQUIRE_CLOSE(total_weight, data.num_data_points(), 1e-4);
    
	index_t index4 = split4.get_feature_index();
	auto split_val = split4.get_num_split_value();
	auto split_bits= split4.get_cat_split_set();
	std::ostringstream oss;
	{
		cereal::XMLOutputArchive oarchive(oss);
		oarchive(split4);
	}
	
		
	split_type split5;
	{
		std::istringstream iss(oss.str());
		cereal::XMLInputArchive iarchive(iss);
		iarchive(split5);
	}
	
	BOOST_REQUIRE(index4     == split5.get_feature_index());
	BOOST_REQUIRE(split_val  == split5.get_num_split_value());
	BOOST_REQUIRE(split_bits == split5.get_cat_split_set());
	
}


// test binary serialization
BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_binary_serialization){
	
	auto data = load_toy_data();

	std::vector<info_t > data_info(data.num_data_points());
	
	for (auto i=0u; i<data.num_data_points(); ++i){
		data_info[i].index=i;
		data_info[i].response = data.response(i);
		data_info[i].weight = 1;

	}

	std::array<std::vector<info_t>::iterator, 3> infos_split_it;
	std::vector<index_t> features_to_try({0,1});

	rng_type rng;


	split_type split4;
	num_t loss = split4.find_best_split(data, features_to_try,data_info.begin(), data_info.end(),infos_split_it, 1, 1, rng);

    num_t total_weight = 0;
    for (auto it = infos_split_it[0]; it!=infos_split_it[1]; ++it)
        total_weight += (*it).weight;
    for (auto it = infos_split_it[1]; it!=infos_split_it[2]; ++it)
        total_weight += (*it).weight;
    BOOST_REQUIRE_CLOSE(total_weight, data.num_data_points(), 1e-4);
    
	
	index_t index4 = split4.get_feature_index();
	auto split_val = split4.get_num_split_value();
	auto split_bits= split4.get_cat_split_set();

	std::ostringstream oss;
	{
		cereal::PortableBinaryOutputArchive oarchive(oss);
		oarchive(split4);
	}
	
		
	split_type split5;
	{
		std::istringstream iss(oss.str());
		cereal::PortableBinaryInputArchive iarchive(iss);
		iarchive(split5);
	}
	
	BOOST_REQUIRE_EQUAL(index4,    split5.get_feature_index());
	BOOST_REQUIRE_EQUAL(split_val, split5.get_num_split_value());
	BOOST_REQUIRE_EQUAL(split_bits,split5.get_cat_split_set());
	
}

