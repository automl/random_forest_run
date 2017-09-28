#include <boost/test/unit_test.hpp>

#include <random>

#include <memory>

#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/trees/k_ary_mondrian_tree.hpp"
#include "rfr/forests/regression_forest.hpp"
#include "rfr/forests/mondrian_forest.hpp"
#include "rfr/forests/quantile_regression_forest.hpp"

#include <sstream>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
typedef cereal::PortableBinaryInputArchive iarch_type;
typedef cereal::PortableBinaryOutputArchive oarch_type;



typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;

typedef rfr::data_containers::default_container<num_t, response_t, index_t> data_container_type;
typedef rfr::nodes::k_ary_mondrian_node_full<2, num_t, response_t, index_t, rng_t> node_type;
typedef rfr::nodes::temporary_node<num_t, index_t> tmp_node_type;/**/
typedef rfr::trees::k_ary_mondrian_tree<2, node_type, num_t, response_t, index_t, rng_t> tree_type;
typedef rfr::forests::mondrian_forest< tree_type, num_t, response_t, index_t, rng_t> forest_type;

data_container_type load_diabetes_data(){
	//data_container_type data;
	data_container_type data(10);
	
    std::string feature_file, response_file;
    
    feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "diabetes_features.csv";
    response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "diabetes_responses.csv";

    data.import_csv_files(feature_file, response_file);
    return(data);
}


BOOST_AUTO_TEST_CASE( mondrian_forest_compile_tests ){
    
    
    auto data = load_diabetes_data();

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 4;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.hierarchical_smoothing = false;
	tree_opts.max_features = data.num_data_points()*3/4;
	tree_opts.life_time = 5;

	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);


	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 8;
	forest_opts.do_bootstrapping = true;
	forest_opts.compute_oob_error= true;
	
	forest_type the_forest(forest_opts);
	
	rng_t rng;

	BOOST_REQUIRE(std::isnan(the_forest.out_of_bag_error()));

	the_forest.fit(data, rng);
	BOOST_REQUIRE(!std::isnan(the_forest.out_of_bag_error()));

	response_t s_d, pred_mean;
    auto tmp = the_forest.predict(data.retrieve_data_point(5));

	std::ostringstream oss;
	{
		oarch_type oarchive(oss);
		oarchive(the_forest);
	}

	
	the_forest.save_to_binary_file("mondrian_forest_test.bin");
	
	forest_type the_forest2;

	auto s = the_forest.ascii_string_representation();

	the_forest2.load_from_ascii_string(s);

	forest_type the_forest3;
	the_forest3.load_from_binary_file("mondrian_forest_test.bin");

}

BOOST_AUTO_TEST_CASE( mondrian_forest_partial_fit ){
        
    auto data = load_diabetes_data();

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 4;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.hierarchical_smoothing = false;
	tree_opts.max_features = data.num_data_points()*3/4;
	tree_opts.life_time = 5;//1000

	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);


	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 8;
	forest_opts.do_bootstrapping = true;
	forest_opts.compute_oob_error= true;
	
	forest_type the_forest(forest_opts);
	
	rng_t rng;
	response_t s_d, pred_mean;
	BOOST_REQUIRE(std::isnan(the_forest.out_of_bag_error()));

	// TODO check prediction again with new API
	//BOOST_REQUIRE(std::isnan(the_forest.predict(data.retrieve_data_point(5), s_d, pred_mean, rng)));

	//response_t pre;
	//pre = the_forest.predict(data.retrieve_data_point(5), s_d, pred_mean, rng);
	
	//std::cout << "UNIT_TEST:: " << pre << std::endl;

	for(int i =0; i<data.num_data_points(); i++){
		the_forest.partial_fit(data, rng, i);	
		//BOOST_REQUIRE(!std::isnan(the_forest.predict(data.retrieve_data_point(5), s_d, pred_mean, rng)));
	}

    auto tmp = the_forest.predict(data.retrieve_data_point(5));

	std::ostringstream oss;
	{
		oarch_type oarchive(oss);
		oarchive(the_forest);
	}

	the_forest.save_to_binary_file("mondrian_forest_partial_fit_test.bin");
	
	forest_type the_forest2;

	auto s = the_forest.ascii_string_representation();

	the_forest2.load_from_ascii_string(s);

	forest_type the_forest3;
	the_forest3.load_from_binary_file("mondrian_forest_partial_fit_test.bin");

}

BOOST_AUTO_TEST_CASE( mondrian_forest_predict_median_test ){
    
    
    auto data = load_diabetes_data();

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 4;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.hierarchical_smoothing = false;
	tree_opts.max_features = data.num_data_points()*3/4;
	tree_opts.life_time = 5;//1000

	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);


	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 8;
	forest_opts.do_bootstrapping = true;
	forest_opts.compute_oob_error= true;
	
	forest_type the_forest(forest_opts);
	
	rng_t rng;

	BOOST_REQUIRE(std::isnan(the_forest.out_of_bag_error()));

	the_forest.fit(data, rng);
	BOOST_REQUIRE(!std::isnan(the_forest.out_of_bag_error()));

	response_t s_d, pred_mean;
    auto tmp = the_forest.predict_median(data.retrieve_data_point(5));

	std::ostringstream oss;
	{
		oarch_type oarchive(oss);
		oarchive(the_forest);
	}

	
	the_forest.save_to_binary_file("mondrian_forest_test.bin");
	
	forest_type the_forest2;

	auto s = the_forest.ascii_string_representation();

	the_forest2.load_from_ascii_string(s);

	forest_type the_forest3;
	the_forest3.load_from_binary_file("mondrian_forest_test.bin");

}





BOOST_AUTO_TEST_CASE( mondrian_forest_exceptions_tests ){
    
    auto data = load_diabetes_data();

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = data.num_data_points()*3/4;
	tree_opts.life_time = 5;

	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);

	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 8;
	forest_opts.do_bootstrapping = false;
	forest_opts.compute_oob_error= true;
	
	forest_type the_forest(forest_opts);
	rng_t rng;

	// no trees in the forest
	the_forest.options.num_trees = 0;
	BOOST_REQUIRE_THROW(the_forest.fit(data, rng), std::runtime_error);

	// no datapoints in any trees
	the_forest.options.num_trees = 8;
	the_forest.options.num_data_points_per_tree = 0;
	BOOST_REQUIRE_THROW(the_forest.fit(data, rng), std::runtime_error);

	// not enough datapoints and no bootstrapping
	the_forest.options.num_data_points_per_tree = 32768;
	BOOST_REQUIRE_THROW(the_forest.fit(data, rng), std::runtime_error);

	// a successful training without bootstrapping for covarage
	the_forest.options.num_data_points_per_tree = 128;
	the_forest.fit(data, rng);

	// no features to split
	the_forest.options.num_data_points_per_tree = 128;
	the_forest.options.tree_opts.max_features = 0;
	BOOST_REQUIRE_THROW(the_forest.fit(data, rng), std::runtime_error);

}
