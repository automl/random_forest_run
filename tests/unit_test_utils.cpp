// compile with the following two options:
// -lboost_unit_test_framework -DBOOST_TEST_DYN_LINK
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE rfr_test


#include <cmath>

#include <boost/test/unit_test.hpp>


#include "rfr/util.hpp"

BOOST_AUTO_TEST_CASE(merge_feature_vectors_test){
	
	double v1[4] = {1,2,3,nan("")};
	double v2[4] = {nan(""), nan(""), 4, 5};
	double v3[4] = {nan(""), nan(""), nan(""), nan("")};
	
	rfr::merge_two_vectors(v1,v2,v3,4);
	
	BOOST_REQUIRE(v3[0] == 1);
	BOOST_REQUIRE(v3[1] == 2);
	BOOST_REQUIRE(v3[2] == 4);
	BOOST_REQUIRE(v3[3] == 5);
	
	BOOST_REQUIRE_THROW(rfr::merge_two_vectors(v1,v1,v3,4), std::runtime_error);
}



BOOST_AUTO_TEST_CASE(test_running_statistics){
	
	// 256 random ints in [0,256)
	double values[] = {61, 99, 125, 222, 34, 208, 247, 156, 59, 2, 226, 116, 203, 213, 123, 250, 209, 124, 218, 20, 43, 66, 154, 142, 223, 117, 252, 249, 105, 42, 2, 248, 69, 180, 142, 196, 237, 124, 25, 53, 76, 5, 9, 219, 114, 251, 21, 247, 183, 83, 147, 202, 16, 101, 192, 209, 140, 207, 225, 34, 160, 171, 173, 188, 161, 76, 242, 97, 104, 10, 163, 32, 243, 140, 204, 211, 106, 212, 199, 14, 115, 116, 196, 120, 87, 50, 204, 28, 158, 191, 127, 110, 210, 224, 162, 105, 68, 236, 25, 142, 80, 196, 235, 219, 140, 251, 113, 240, 81, 9, 133, 219, 186, 153, 55, 35, 166, 9, 238, 125, 233, 69, 181, 109, 63, 34, 193, 240, 174, 194, 213, 165, 26, 210, 167, 21, 168, 167, 55, 135, 170, 206, 13, 91, 225, 159, 253, 127, 196, 140, 144, 222, 190, 15, 158, 22, 185, 79, 11, 106, 57, 94, 78, 116, 183, 128, 161, 212, 38, 242, 157, 41, 253, 192, 184, 6, 163, 17, 66, 128, 245, 80, 194, 208, 73, 181, 91, 93, 38, 123, 213, 197, 109, 231, 36, 168, 199, 172, 211, 180, 246, 111, 45, 249, 73, 187, 42, 255, 83, 103, 45, 76, 145, 10, 59, 84, 179, 168, 251, 77, 218, 109, 221, 237, 135, 154, 94, 69, 49, 79, 102, 254, 77, 40, 107, 13, 226, 84, 78, 128, 35, 177, 4, 123, 172, 55, 174, 46, 176, 43, 77, 110, 43, 37, 148, 237};

	rfr::running_statistics stat1, stat2;

	for (auto &v: values){
		stat1(v);
		// second values are all shifted by a large number, which makes 'naive' algorithms fail
		stat2(v+1e9);
	}

	// reference computed with numpy, so should be fine
	BOOST_REQUIRE(stat1.number_of_points() == 256);
	BOOST_REQUIRE_CLOSE(stat1.mean(),134.5078125,1e-6);
	BOOST_REQUIRE_CLOSE(stat1.variance(),double(stat1.number_of_points())*5481.3124389648438/double(stat1.number_of_points()-1),1e-6);

	BOOST_REQUIRE(stat2.number_of_points() == 256);
	BOOST_REQUIRE_CLOSE(stat2.mean(),134.5078125+1e9,1e-6);
	BOOST_REQUIRE_CLOSE(stat2.variance(),double(stat2.number_of_points())*5481.3124389648438/double(stat2.number_of_points()-1),1e-6);
}
