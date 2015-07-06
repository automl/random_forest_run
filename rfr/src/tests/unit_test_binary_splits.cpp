// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test
#include <boost/test/unit_test.hpp>


#include <numeric>
#include "../data_containers/mostly_continous_data.hpp"
#include "../splits/binary_split_rss_error.hpp"

typedef double num_type;
typedef unsigned int index_type;


BOOST_AUTO_TEST_CASE(split_class_test){
	rfr::mostly_contiuous_data<num_type> data;
	data.read_feature_file("../../test_data_sets/diabetes_features.csv");
	data.read_response_file("../../test_data_sets/diabetes_responses.csv");



	std::vector<int> indices(data.num_data_points());
	std::iota(indices.begin(), indices.end(), 0);

	std::vector<int>::iterator indices_split_it;
	num_type score;

	//rfr::split<num_type> s (*(data.feature(0)), *(data.responses()), 0, data.type_of_feature(0), indices, indices_split_it, score);

}
