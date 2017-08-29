#include <boost/test/unit_test.hpp>

#include <random>

#include <memory>

#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/forests/regression_forest.hpp"
#include "rfr/forests/quantile_regression_forest.hpp"
#include "rfr/forests/fanova_forest.hpp"


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

typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t> split_type;
typedef rfr::nodes::k_ary_node_full<2, split_type, num_t, response_t, index_t, rng_t> node_type;

typedef rfr::nodes::temporary_node<num_t, index_t> tmp_node_type;

typedef rfr::trees::k_ary_random_tree<2, node_type, num_t, response_t, index_t, rng_t> tree_type;


typedef rfr::forests::regression_forest< tree_type, num_t, response_t, index_t, rng_t> forest_type;

typedef rfr::forests::quantile_regression_forest< tree_type, num_t, response_t, index_t, rng_t> qrf_type;


typedef rfr::forests::fANOVA_forest<split_type, num_t, response_t, index_t, rng_t> fANOVAf_type;



data_container_type load_diabetes_data(){
	data_container_type data(10);
	
    std::string feature_file, response_file;
    
    feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "diabetes_features.csv";
    response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "diabetes_responses.csv";

    data.import_csv_files(feature_file, response_file);
    return(data);
}

BOOST_AUTO_TEST_CASE( regression_forest_serialize_test ){
    
    
    auto data = load_diabetes_data();

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = data.num_features()*3/4;
	tree_opts.max_depth=5;
	tree_opts.max_num_nodes=2048;

	
	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);

	forest_opts.num_data_points_per_tree = data.num_data_points()/2;
	forest_opts.num_trees = 16;
	forest_opts.do_bootstrapping = true;
	forest_opts.compute_oob_error= true;
	
	forest_type the_forest(forest_opts);
	
	rng_t rng;

	BOOST_REQUIRE(std::isnan(the_forest.out_of_bag_error()));

	the_forest.fit(data, rng);

	BOOST_REQUIRE(!std::isnan(the_forest.out_of_bag_error()));

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


	for (auto i=0u; i < data.num_data_points(); ++i){
		auto v1 = the_forest.predict(data.retrieve_data_point(i));
		auto v2 = the_forest2.predict(data.retrieve_data_point(i));
		auto v3 = the_forest3.predict(data.retrieve_data_point(i));

		BOOST_REQUIRE_EQUAL(v1,v2);
		BOOST_REQUIRE_EQUAL(v1,v3);
	}


	

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




BOOST_AUTO_TEST_CASE( quantile_regression_forest_test ){
	
	auto data = load_diabetes_data();

	rng_t rng;

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = 10;

	// don't split anything

	tree_opts.max_num_nodes = 1;
	tree_opts.max_depth = 0;

	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);

	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 1;
	forest_opts.do_bootstrapping = false;


	// just to test the default constructor
	qrf_type sudowoodo;
	BOOST_REQUIRE_THROW(sudowoodo.fit(data, rng), std::runtime_error);

	qrf_type the_forest(forest_opts);

	//fit forest
	the_forest.fit(data, rng);

	auto mew = data.retrieve_data_point(0);
	std::vector<num_t> mew2 = {0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1};

	auto qv1 = the_forest.predict_quantiles(mew , mew2);

	BOOST_REQUIRE_EQUAL(qv1.size(), mew2.size());

	// check that shuffling doesn't affect the outcome
	std::shuffle(mew2.begin(), mew2.end(), rng);
	auto qv2 = the_forest.predict_quantiles(mew , mew2);
	BOOST_CHECK_EQUAL_COLLECTIONS ( qv1.begin(), qv1.end(), qv2.begin(), qv2.end());

	//
	std::vector<num_t> qv_numpy_ref = {  25.,   60.,   77.,   94.,  115.,  141.,  168.,  197.,  233., 268.,  346.};

	BOOST_CHECK_EQUAL_COLLECTIONS ( qv1.begin(), qv1.end(), qv_numpy_ref.begin(), qv_numpy_ref.end());


	BOOST_REQUIRE_THROW(the_forest.predict_quantiles(mew, {0,-0.5,1}) ,std::runtime_error);
	BOOST_REQUIRE_THROW(the_forest.predict_quantiles(mew, {1.1,0.5}) ,std::runtime_error);
}

/* not interesting right now!
BOOST_AUTO_TEST_CASE( fANOVA_forest_test ){
	
	auto data = load_diabetes_data();

	rng_t rng;

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = 10;


	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);

	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 23;
	forest_opts.do_bootstrapping = false;


	fANOVAf_type the_forest(forest_opts);

	//fit forest
	the_forest.fit(data, rng);

	BOOST_REQUIRE_EQUAL(the_forest.num_trees(), 23);
	BOOST_REQUIRE_EQUAL(the_forest.options.num_trees, 23);
	BOOST_REQUIRE_EQUAL(the_forest.options.do_bootstrapping, false);


	the_forest.options.num_trees = 24;
	the_forest.options.do_bootstrapping = true;
	//refit forest
	the_forest.fit(data, rng);
	BOOST_REQUIRE_EQUAL(the_forest.num_trees(), 24);
	BOOST_REQUIRE_EQUAL(the_forest.options.do_bootstrapping, true);


	
	// 10th and 90th percentile of the response values
	the_forest.set_cutoffs(60,265);

}
*/

