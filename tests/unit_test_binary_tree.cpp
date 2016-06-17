#include <boost/test/unit_test.hpp>

#include <random>
#include <numeric>
#include <cstring>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <fstream>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"


typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

typedef rfr::data_containers::mostly_continuous_data<num_type, response_type, index_type> data_container_type;

typedef rfr::splits::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type;
typedef rfr::nodes::k_ary_node<2, split_type, rng_type, num_type, response_type, index_type> node_type;

typedef rfr::nodes::temporary_node<num_type, index_type> tmp_node_type;

typedef rfr::trees::k_ary_random_tree<2, split_type, rng_type, num_type, response_type, index_type> tree_type;

// Test does not actually check the correctness of the split or anything.
// It makes sure everything compiles and runs
BOOST_AUTO_TEST_CASE( binary_tree_test ){

    data_container_type data;
    char filename [1024];

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_features.csv");
    data.read_feature_file(filename);

    strcpy(filename, boost::unit_test::framework::master_test_suite().argv[1]);
    strcat(filename, "toy_data_set_responses.csv");
    data.read_response_file(filename);


    data.set_type_of_feature(1, 4);
    
    rfr::trees::tree_options<num_type, response_type, index_type> tree_opts;
	
	
    tree_opts.max_features = 1;
    tree_opts.max_depth = 3;
	
    rng_type rng_engine;

    for (auto i = 0; i <4; i++){
	tree_type the_tree;
	the_tree.fit(data, tree_opts, rng_engine);
	
	char filename[100];
	sprintf(filename, "/tmp/tree_%i.tex", i);
	the_tree.save_latex_representation(filename);
    }
    
    
    
    tree_type the_tree;
	the_tree.fit(data, tree_opts, rng_engine);
    
	{
		std::ofstream ofs("test_binary_tree.xml");
		cereal::XMLOutputArchive oarchive(ofs);
		oarchive(the_tree);
	}
    
        		
	tree_type the_tree2;
	{
		std::ifstream ifs("test_binary_tree.xml");
		cereal::XMLInputArchive iarchive(ifs);
		iarchive(the_tree2);
	}
    
    
}
