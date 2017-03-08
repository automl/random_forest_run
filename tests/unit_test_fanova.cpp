#include <boost/test/unit_test.hpp>

#include <random>
#include <numeric>
#include <cstring>
#include <vector>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <fstream>
#include <sstream>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"

#include "rfr/trees/binary_fanova_tree.hpp"

typedef double num_type;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;

typedef rfr::data_containers::mostly_continuous_data<num_type, response_t, index_t> 			data_container_type;
typedef rfr::splits::binary_split_one_feature_rss_loss<num_type, response_t, index_t, rng_t>	split_type;
typedef rfr::nodes::k_ary_node_full<2, split_type, num_type, response_t, index_t, rng_t> 		node_type;
typedef rfr::trees::k_ary_random_tree<2, node_type, num_type, response_t, index_t, rng_t>		tree_type;
typedef rfr::trees::binary_fANOVA_tree<split_type, num_type, response_t, index_t, rng_t>		fANOVA_tree_type;

data_container_type load_toy_data(){
  data_container_type data;

  std::string feature_file, response_file;
  feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_features.csv";
  response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_responses.csv";

  data.import_csv_files(feature_file, response_file);

  BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);

  return(data);
}


BOOST_AUTO_TEST_CASE (fanova_test) {
	auto data = load_toy_data();
	data.set_type_of_feature(1, 3);

	rfr::trees::tree_options<num_type, response_t, index_t> tree_opts;
	tree_opts.max_features = 2;
	tree_opts.max_depth = 3;
	rng_t rng_engine;

	for (auto i = 0; i <1; i++){
		fANOVA_tree_type the_tree;
		std::vector<std::vector<num_type>> pcs = {{0, 100}, {0, 1, 2}};
		std::vector<index_t> types = {0, 3};
		num_type inf = std::numeric_limits<num_type>::infinity();


		// an unfit tree should throw an exception if you try to precompute the marginals!
		BOOST_REQUIRE_THROW(the_tree.precompute_marginals(-inf, inf, pcs, types),std::runtime_error);


		the_tree.fit(data, tree_opts, std::vector<num_type>(data.num_data_points(), 1), rng_engine);

		the_tree.save_latex_representation("/tmp/rfr_test.tex");
		the_tree.precompute_marginals(-inf, inf, pcs, types);


		std::vector<num_type> feature_3({10., NAN});
		std::vector<num_type> feature_4({50, NAN});
		std::vector<num_type> feature_56({90., NAN});

		std::vector<num_type> feature_345({NAN, 0});
		std::vector<num_type> feature_346({NAN, 1});

		std::vector<num_type> feature_6({90., 1});


		// check subspace sizes without cutoffs
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(0), 300, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(1), 178.61191331472548, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(2), 121.38808668527452, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(3),  59.86622661388443, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(4), 118.74568670084104, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(5),  40.46269556175817, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(6),  80.92539112351633, 1e-6);

		// the correpsonding marginal predictions
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(0), 2.4748241706496, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(1), 1.6648251199885, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(2), 3.6666666666666, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(3),               1, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(4),               2, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(5),               3, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(6),               4, 1e-6);

		// and the active variables
		BOOST_REQUIRE( the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE( the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE( the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE( the_tree.get_active_variables(2)[1]);

		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_3),                1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_4),                2,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_56), 3.6666666666666,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_345),1.9114295757429,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_346),2.3929475797473,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_6),                4,1e-6);


		// now let's exclude exactly one leaf
		the_tree.precompute_marginals(-inf, 3.5, pcs, types);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(0), 219.07460887648367, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(1), 178.61191331472548, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(2),  40.46269556175817, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(3),  59.86622661388443, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(4), 118.74568670084104, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(5),  40.46269556175817, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(6),  80.92539112351633, 1e-6);

		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(0), 1.9114295757430011, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(1), 1.6648251199885176, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(2),                  3, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(3),                  1, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(4),                  2, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(5),                  3 , 1e-6);
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(6)));

		BOOST_REQUIRE( the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE( the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE( the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE( the_tree.get_active_variables(2)[1]);

		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_3),                1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_4),                2,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_56),               3,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_345),1.9114295757429,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_346),1.6648251199885,1e-6);
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_6)));


		// now let's exclude exactly two leaves that belong to the same internal node
		the_tree.precompute_marginals(-inf, 2.5, pcs, types);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(0), 178.61191331472548, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(1), 178.61191331472548, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(2),                  0, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(3),  59.86622661388443, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(4), 118.74568670084104, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(5),  40.46269556175817, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(6),  80.92539112351633, 1e-6);

		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(0), 1.6648251199885176, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(1), 1.6648251199885176, 1e-6);
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(2)));
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(3),  1, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(4),  2, 1e-6);
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(5)));
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(6)));

		BOOST_REQUIRE( the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE( the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[1]);


		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_3),                1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_4),                2,1e-6);
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_56)));
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_345),1.6648251199885,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_mean_prediction(feature_346),1.6648251199885,1e-6);
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_6)));



		// now let's exclude all leaves, just to see what happens
		the_tree.precompute_marginals(2.25, 2.75, pcs, types);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(0),                  0, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(1),                  0, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(2),                  0, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(3),  59.86622661388443, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(4), 118.74568670084104, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(5),  40.46269556175817, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(6),  80.92539112351633, 1e-6);

		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(0)));
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(1)));
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(2)));
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(3)));
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(4)));
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(5)));
		BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(6)));

		BOOST_REQUIRE(!the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[1]);

		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_3)));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_4)));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_56)));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_345)));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_346)));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_mean_prediction(feature_6)));
		

		

/*
      for (index_t node_index = 0; node_index < nodes.size(); ++node_index) {
        index_t parent_index = nodes[node_index].parent();
        bool any_variable = false;
        for (index_t var_index = 0; var_index < the_tree.get_vars(node_index).size(); ++var_index) {
          any_variable |= the_tree.get_vars(node_index)[var_index];
          BOOST_ASSERT(!the_tree.get_vars(node_index)[var_index] || the_tree.get_vars(parent_index)[var_index]);
          // not active in node or active in parent <==> active in node implies active in parent
        }
        BOOST_ASSERT(any_variable); // TODO: Remove later, just ensuring the test is not trivial
        num_type subspace = 1;
        for (index_t child_index : nodes[node_index].get_children()) {
          subspace *= the_tree.get_subspace_size(child_index);
        }
        BOOST_REQUIRE_EQUAL(subspace, the_tree.get_subspace_size(node_index));
        BOOST_ASSERT(nodes[node_index].get_children().empty() || subspace > 1.001); // TODO: Remove later, just ensuring the test is not trivial
      }

      // 

      char filename[100];
      sprintf(filename, "/tmp/tree_%i.tex", i);
      // the_tree.save_latex_representation(filename); // FIXME(mostafa): error here
    */
    }
}
