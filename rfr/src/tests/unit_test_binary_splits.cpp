// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test


#include <cstring>
#include <numeric>

#include <boost/test/unit_test.hpp>

#include "data_containers/mostly_continous_data_container.hpp"
#include "splits/binary_split_one_feature_rss_loss.hpp"


typedef double num_type;
typedef unsigned int index_type;


BOOST_AUTO_TEST_CASE(binary_split_one_feature_rss_loss_test){
	
	char filename[1000];
	
	// read the test dataset
	rfr::mostly_contiuous_data<num_type> data;
	strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);
	
	data.set_type_of_feature(1,3);
	

	std::vector<index_type> indices(data.num_data_points());
	std::iota(indices.begin(), indices.end(), 0);

	std::vector<index_type>::const_iterator indices_split_it;
	std::vector<index_type> features_to_try(1,0);

	rfr::binary_split_one_feature_rss_loss<rfr::mostly_contiuous_data<num_type>, num_type, index_type>( data, features_to_try, indices, indices_split_it);


	std::cout<<"num_data_points="<<data.num_data_points()<<"\nnum_features="<<data.num_features()<<std::endl;
	features_to_try.assign(1,1);

	rfr::binary_split_one_feature_rss_loss<rfr::mostly_contiuous_data<num_type, index_type>, num_type, index_type> (data, features_to_try, indices, indices_split_it);


}
