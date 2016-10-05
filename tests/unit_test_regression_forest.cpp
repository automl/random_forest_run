#include <boost/test/unit_test.hpp>

#include <random>

#include <memory>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss_v2.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
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
typedef rfr::nodes::k_ary_node<2, split_type, num_t, response_t, index_t, rng_t> node_type;

typedef rfr::nodes::temporary_node<num_t, index_t> tmp_node_type;

typedef rfr::trees::k_ary_random_tree<2, split_type, num_t, response_t, index_t, rng_t> tree_type;


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
		cereal::XMLOutputArchive oarchive(oss);
		oarchive(the_forest);
	}

	
	the_forest.save_to_binary_file("regression_forest_test.bin");
	
	forest_type the_forest2;

	{
		std::istringstream iss(oss.str());
		cereal::XMLInputArchive iarchive(iss);
		iarchive(the_forest2);
	}	
	
	forest_type the_forest3;
	the_forest3.load_from_binary_file("regression_forest_test.bin");
	
}


/*
BOOST_AUTO_TEST_CASE( regression_forest_update_downdate_tests ){
	
	double unique_value = 42.424242;
	
    data_container_type data;

    char *filename = (char*) malloc(1024*sizeof(char));

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    //strcat(filename, "toy_data_set_features.csv");
    strcat(filename, "diabetes_features.csv");
    std::cout<<filename<<"\n";
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
	//strcat(filename, "toy_data_set_responses.csv");
	strcat(filename, "diabetes_responses.csv");
	std::cout<<filename<<"\n";
    data.read_response_file(filename);

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.max_features = 10;

	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);

	forest_opts.num_data_points_per_tree = data.num_data_points();
	forest_opts.num_trees = 10;
	forest_opts.do_bootstrapping = true;


	rfr::forests::regression_forest< tree_type, rng_t, num_t, response_t, index_t> the_forest(forest_opts);
	
	rng_t rng;

	//fit forest
	the_forest.fit(data, rng);

	// get reference leaf values for one configuration
	std::vector<std::vector< num_t> > before = the_forest.all_leaf_values(data.retrieve_data_point(0).data());

	// update forest with that configuration and a unique response value
	data_container_type pseudo_data(data.num_features());
	pseudo_data.add_data_point(data.retrieve_data_point(0).data(), data.num_features(), unique_value);
	
	the_forest.pseudo_update(pseudo_data);
	
	// get new leaf values
	std::vector<std::vector< num_t> > after_update = the_forest.all_leaf_values(data.retrieve_data_point(0).data());
	
	// compare them to ensure the data point has been added correctly
	
	for (auto i=0u; i < before.size(); ++i){
		BOOST_REQUIRE(before[i].size() == after_update[i].size() -1);
		BOOST_CHECK_EQUAL_COLLECTIONS( before[i].begin(), before[i].end(), after_update[i].begin(), --after_update[i].end());
		BOOST_REQUIRE( after_update[i].back() == unique_value);
	}
	
	// downdate the tree
	BOOST_REQUIRE(the_forest.pseudo_downdate() == true);
	
	// get new leaf values
	std::vector<std::vector< num_t> > after_downdate = the_forest.all_leaf_values(data.retrieve_data_point(0).data());
	
	// compare them to ensure the last data point has been removed
	for (auto i=0u; i < before.size(); ++i)
		BOOST_CHECK_EQUAL_COLLECTIONS( before[i].begin(), before[i].end(), after_downdate[i].begin(), after_downdate[i].end());
	
	BOOST_REQUIRE(the_forest.pseudo_downdate() == false);
	
	
	std::cout<<the_forest.covariance(data.retrieve_data_point(0).data(), data.retrieve_data_point(1).data())<<std::endl;

	num_t m1 , v1;
	std::tie(m1, v1) = the_forest.predict_mean_var(data.retrieve_data_point(0).data());

	num_t v2 = the_forest.covariance(data.retrieve_data_point(0).data(), data.retrieve_data_point(0).data());

	BOOST_REQUIRE_CLOSE(v1, v2, 1e-4);
	
    free(filename);
}


*/
