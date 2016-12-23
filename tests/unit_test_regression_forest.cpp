#include <boost/test/unit_test.hpp>

#include <random>

#include <memory>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/forests/regression_forest.hpp"

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

typedef rfr::data_containers::mostly_continuous_data<num_t, response_t, index_t> data_container_type;

typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t> split_type;
typedef rfr::nodes::k_ary_node_full<2, split_type, num_t, response_t, index_t, rng_t> node_type;

typedef rfr::nodes::temporary_node<num_t, index_t> tmp_node_type;

typedef rfr::trees::k_ary_random_tree<2, node_type, num_t, response_t, index_t, rng_t> tree_type;


typedef rfr::forests::regression_forest< tree_type, num_t, response_t, index_t, rng_t> forest_type;

data_container_type load_diabetes_data(){
	data_container_type data;
	
    std::string feature_file, response_file;
    
    feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "diabetes_features.csv";
    response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "diabetes_responses.csv";

    data.import_csv_files(feature_file, response_file);
    return(data);
}



BOOST_AUTO_TEST_CASE( regression_forest_compile_tests ){
    
    
    auto data = load_diabetes_data();

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = data.num_data_points()*3/4;

	
	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);

	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 8;
	forest_opts.do_bootstrapping = true;
	forest_opts.compute_oob_error= true;
	
	forest_type the_forest(forest_opts);
	
	rng_t rng;

	the_forest.fit(data, rng);
	std::cout<<"OOB Error: "<<the_forest.out_of_bag_error()<<std::endl;

    auto tmp = the_forest.predict(data.retrieve_data_point(5));

	std::ostringstream oss;
	
	{
		oarch_type oarchive(oss);
		oarchive(the_forest);
	}

	
	the_forest.save_to_binary_file("regression_forest_test.bin");
	
	forest_type the_forest2;

	auto s = the_forest.ascii_string_representation();

	the_forest2.load_from_ascii_string(s);

	
	forest_type the_forest3;
	the_forest3.load_from_binary_file("regression_forest_test.bin");
	
}

BOOST_AUTO_TEST_CASE( regression_forest_exceptions_tests ){
    
    
    auto data = load_diabetes_data();

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = data.num_data_points()*3/4;


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





BOOST_AUTO_TEST_CASE( regression_forest_update_downdate_tests ){
	
	double unique_value = 42.424242;
	
	auto data = load_diabetes_data();

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = 10;

	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);

	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 10;
	forest_opts.do_bootstrapping = true;


	forest_type the_forest(forest_opts);
	
	rng_t rng;

	//fit forest
	the_forest.fit(data, rng);

	// get reference leaf values for one configuration
	std::vector<std::vector< num_t> > before = the_forest.all_leaf_values(data.retrieve_data_point(0));

	// update forest with that configuration and a unique response value
	the_forest.pseudo_update(data.retrieve_data_point(0), unique_value, 1);
	
	// get new leaf values
	std::vector<std::vector< num_t> > after_update = the_forest.all_leaf_values(data.retrieve_data_point(0));
	
	// compare them to ensure the data point has been added correctly
	
	for (auto i=0u; i < before.size(); ++i){
		BOOST_REQUIRE(before[i].size() == after_update[i].size() -1);
		BOOST_CHECK_EQUAL_COLLECTIONS( before[i].begin(), before[i].end(), after_update[i].begin(), --after_update[i].end());
		BOOST_REQUIRE( after_update[i].back() == unique_value);
	}
	
	// downdate the tree
	the_forest.pseudo_downdate(data.retrieve_data_point(0), unique_value, 1);
	
	// get new leaf values
	std::vector<std::vector< num_t> > after_downdate = the_forest.all_leaf_values(data.retrieve_data_point(0));
	
	// compare them to ensure the last data point has been removed
	for (auto i=0u; i < before.size(); ++i)
		BOOST_CHECK_EQUAL_COLLECTIONS( before[i].begin(), before[i].end(), after_downdate[i].begin(), after_downdate[i].end());
	
	auto m = the_forest.predict(data.retrieve_data_point(0));

	//num_t v2 = the_forest.covariance(data.retrieve_data_point(0).data(), data.retrieve_data_point(0));

	//BOOST_REQUIRE_CLOSE(v1, v2, 1e-4);
}
