#include <boost/test/included/unit_test.hpp>

#include <random>
#include <numeric>
#include <cstring>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <fstream>
#include <sstream>

#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"

#include "rfr/trees/binary_fanova_tree.hpp"

typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;

typedef rfr::data_containers::default_container<num_t, response_t, index_t> 			data_container_t;
typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t> 	split_t;
typedef rfr::nodes::k_ary_node_full<2, split_t, num_t, response_t, index_t, rng_t> 		node_t;
typedef rfr::nodes::temporary_node<num_t, index_t> 											tmp_node_t;
typedef rfr::trees::k_ary_random_tree<2, node_t, num_t, response_t, index_t, rng_t>		tree_t;
typedef rfr::trees::binary_fANOVA_tree<split_t, num_t, response_t, index_t, rng_t>		fANOVA_tree_t;

data_container_t load_toy_data(){
	data_container_t data(2);
	
    std::string feature_file, response_file;
    
    feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_features.csv";
    response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_responses.csv";

    data.import_csv_files(feature_file, response_file);
	
	data.set_type_of_feature(1,10);
    
    BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);
    
    return(data);
}


data_container_t load_diabetes_data(){
	data_container_t data(10);
	
    std::string feature_file, response_file;
    
    feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "diabetes_features.csv";
    response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "diabetes_responses.csv";

    data.import_csv_files(feature_file, response_file);
    return(data);
}


// Test does not actually check the correctness of the split or anything.
// It makes sure everything compiles and runs

BOOST_AUTO_TEST_CASE( binary_tree_test ){

	auto data = load_toy_data();


    data.set_type_of_feature(1, 4);
    
    rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	
	
    tree_opts.max_features = 1;
    tree_opts.max_depth = 3;
	
    rng_t rng_engine;

    for (auto i = 0; i <1; i++){
	tree_t the_tree;
	the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng_engine);
	
	char filename[100];
		sprintf(filename, "/tmp/tree_%i.tex", i);
		the_tree.save_latex_representation(filename);
    }
    
	tree_t the_tree1;
	the_tree1.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng_engine);
	
	the_tree1.save_latex_representation("/tmp/rfr_test_tree1.tex");
	
	BOOST_REQUIRE(the_tree1.check_split_fractions(1e-6));
	
	std::ostringstream oss;
    
	{
		cereal::XMLOutputArchive oarchive(oss);
		oarchive(the_tree1);
	}
    
        		
	tree_t the_tree2;
	{
		std::istringstream iss(oss.str());
		cereal::XMLInputArchive iarchive(iss);
		iarchive(the_tree2);
	}

    the_tree2.save_latex_representation("/tmp/rfr_test_tree2.tex");
    
	for (auto i=0u; i < data.num_data_points(); ++i){
		auto v1 = the_tree1.predict(data.retrieve_data_point(i));
		auto v2 = the_tree2.predict(data.retrieve_data_point(i));
		BOOST_REQUIRE_EQUAL(v1,v2);
	}


	
}

// Test does not actually check the correctness of the split or anything.
// It makes sure everything compiles and runs
// TODO: add tests for the partitions and marginalized predictions
BOOST_AUTO_TEST_CASE( binary_fANOVA_tree_test ){

	auto data = load_toy_data();


    data.set_type_of_feature(1, 4);
    
    rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	
	
    tree_opts.max_features = 2;
    tree_opts.max_depth = 10;
	
    rng_t rng_engine;

	fANOVA_tree_t the_tree;
	the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng_engine);
	
	the_tree.predict(data.retrieve_data_point(1));
	
}

BOOST_AUTO_TEST_CASE( binary_tree_constraints_test ){

	auto data = load_diabetes_data();
    rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	
	
    tree_opts.max_features = 2;
    tree_opts.max_depth = 3;
	
    rng_t rng_engine;

	tree_t the_tree;
	the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng_engine);
	the_tree.save_latex_representation("/tmp/rfr.tex");
	BOOST_REQUIRE_EQUAL(the_tree.depth(), 3);

	tree_opts.max_depth = 1024;
	tree_opts.max_num_nodes = 15;
	the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng_engine);

	BOOST_REQUIRE_EQUAL(the_tree.depth(), 3);
	BOOST_REQUIRE_EQUAL(the_tree.number_of_nodes(), 15);
	BOOST_REQUIRE_EQUAL(the_tree.number_of_leafs(), 8);


	tree_opts.max_depth = 1024;
	tree_opts.max_num_leaves = 16;
	tree_opts.max_num_nodes = 1024;
	the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng_engine);


	BOOST_REQUIRE(the_tree.number_of_leafs() <= 16);
	BOOST_REQUIRE(the_tree.number_of_nodes() <= 31);


	tree_opts.max_num_leaves = 1024;
	tree_opts.max_num_nodes = 2048;
	tree_opts.min_weight_in_leaf = data.num_data_points()+1;

	the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng_engine);

	BOOST_REQUIRE_EQUAL(1, the_tree.number_of_nodes());

}

BOOST_AUTO_TEST_CASE(  second_binary_tree_constraints_test ){
	std::vector<num_t> y = {0,0,0, 0.001, 1, 1};

	std::vector<std::vector<num_t> > X = 
    {{0},
     {1},
     {2},
     {3},
     {4},
     {5}};

	rng_t rng(12345);
	tree_t the_tree;
    rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;


	tree_opts.max_features = 1;
	tree_opts.max_depth = 1;
	tree_opts.epsilon_purity = 1e-8;

	for (int iter=0; iter<2; iter++){

		data_container_t data(1);

		if (iter==1) data.set_type_of_feature(0,6);

		for (auto i=0u; i<y.size(); ++i)
			data.add_data_point(X[i], y[i]);


		tree_opts.min_weight_to_split = 1;
		tree_opts.min_samples_to_split = 1;
		tree_opts.min_weight_in_leaf = 1;
		tree_opts.min_samples_in_leaf = 1;

		the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng);

		auto stat1 = the_tree.leaf_statistic(data.retrieve_data_point(0));
		auto stat2 = the_tree.leaf_statistic(data.retrieve_data_point(5));

		BOOST_REQUIRE_EQUAL(3, the_tree.number_of_nodes());
		BOOST_REQUIRE_EQUAL(4, stat1.sum_of_weights());
		BOOST_REQUIRE_EQUAL(2, stat2.sum_of_weights());



		tree_opts.min_weight_to_split = 6;
		tree_opts.min_samples_to_split = 1;
		tree_opts.min_weight_in_leaf = 3;
		tree_opts.min_samples_in_leaf = 1;

		the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng);
		stat1 = the_tree.leaf_statistic(data.retrieve_data_point(0));
		stat2 = the_tree.leaf_statistic(data.retrieve_data_point(5));


		BOOST_REQUIRE_EQUAL(3, the_tree.number_of_nodes());
		BOOST_REQUIRE_EQUAL(3, stat1.sum_of_weights());
		BOOST_REQUIRE_EQUAL(3, stat2.sum_of_weights());
		

		// now the same with the number of samples in leaf

		tree_opts.min_weight_to_split = 1;
		tree_opts.min_samples_to_split = 6;
		tree_opts.min_weight_in_leaf = 1;
		tree_opts.min_samples_in_leaf = 3;

		the_tree.fit(data, tree_opts, std::vector<num_t>(data.num_data_points(), 1), rng);

		stat1 = the_tree.leaf_statistic(data.retrieve_data_point(0));
		stat2 = the_tree.leaf_statistic(data.retrieve_data_point(5));

		BOOST_REQUIRE_EQUAL(3, the_tree.number_of_nodes());
		BOOST_REQUIRE_EQUAL(3, stat1.sum_of_weights());
		BOOST_REQUIRE_EQUAL(3, stat2.sum_of_weights());

	}
}


