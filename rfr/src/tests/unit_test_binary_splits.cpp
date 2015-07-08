// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test


#include <cstring>
#include <numeric>
#include <vector>

#include <boost/test/unit_test.hpp>

#include "data_containers/mostly_continous_data_container.hpp"
#include "splits/binary_split_one_feature_rss_loss.hpp"


typedef float num_type;
typedef unsigned int index_type;
typedef rfr::mostly_contiuous_data<num_type, index_type> data_container_type;

BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_test){
	
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

	std::vector<index_type> indices(data.num_data_points());
	std::iota(indices.begin(), indices.end(), 0);

	std::vector<index_type>::iterator indices_split_it;
	std::vector<index_type> features_to_try(1,0);

	rfr::binary_split_one_feature_rss_loss<data_container_type, num_type, index_type> split1;
	num_type loss = split1.find_best_split(data, features_to_try, indices, indices_split_it);

	// actual best split and loss independently computed in python
	BOOST_REQUIRE_CLOSE(loss, 23.33333333, 1e-4);
	
	
	std::vector<num_type> split_criterion = split1.get_split_criterion();
	BOOST_REQUIRE(split_criterion[0] == 0);
	BOOST_REQUIRE(split_criterion[1] >=59);
	BOOST_REQUIRE(split_criterion[1] < 60);

	features_to_try.assign(1,1);
	rfr::binary_split_one_feature_rss_loss<data_container_type, num_type, index_type>split2;
	loss = split2.find_best_split(data, features_to_try, indices, indices_split_it);

	// actual best split and loss independently computed in python
	BOOST_REQUIRE_CLOSE(loss, 88.57142857, 1e-4);
	split_criterion = split2.get_split_criterion();
	
	BOOST_REQUIRE(std::binary_search(++split_criterion.begin(), split_criterion.end(), 1));
	BOOST_REQUIRE(std::binary_search(++split_criterion.begin(), split_criterion.end(), 2));
	BOOST_REQUIRE(!std::binary_search(++split_criterion.begin(), split_criterion.end(), 3));

	
	// check if it finds the best split out of the two above
	features_to_try.assign({1,0});
	rfr::binary_split_one_feature_rss_loss<data_container_type, num_type, index_type>split3;
	loss = split3.find_best_split(data, features_to_try, indices, indices_split_it);
	BOOST_REQUIRE_CLOSE(loss, 23.33333333, 1e-4);
	split_criterion = split3.get_split_criterion();
	BOOST_REQUIRE(split_criterion[0] == 0);
	BOOST_REQUIRE(split_criterion[1] >=59);
	BOOST_REQUIRE(split_criterion[1] < 60);
	


}
