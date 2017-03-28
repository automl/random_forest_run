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

#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"

#include "rfr/trees/binary_fanova_tree.hpp"

typedef double num_type;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;

typedef rfr::data_containers::default_container<num_type, response_t, index_t> 			data_container_type;
typedef rfr::splits::binary_split_one_feature_rss_loss<num_type, response_t, index_t, rng_t>	split_type;
typedef rfr::nodes::k_ary_node_full<2, split_type, num_type, response_t, index_t, rng_t> 		node_type;
typedef rfr::trees::k_ary_random_tree<2, node_type, num_type, response_t, index_t, rng_t>		tree_type;
typedef rfr::trees::binary_fANOVA_tree<split_type, num_type, response_t, index_t, rng_t>		fANOVA_tree_type;

data_container_type load_toy_data(){
  data_container_type data(2);

  std::string feature_file, response_file;
  feature_file  = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_features.csv";
  response_file = std::string(boost::unit_test::framework::master_test_suite().argv[1]) + "toy_data_set_responses.csv";

  data.import_csv_files(feature_file, response_file);

  BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);

  return(data);
}



/* TODO: add test for the mean and total_variance of the tree*/
/*
BOOST_AUTO_TEST_CASE (hooker_fanova_test) {
	auto data = load_toy_data();
	data.set_type_of_feature(1, 3);

	rfr::trees::tree_options<num_type, response_t, index_t> tree_opts;
	tree_opts.max_features = 2;
	tree_opts.max_depth = 3;
	rng_t rng_engine(0);

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

		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_3  , pcs, types).mean(),              1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_4  , pcs, types).mean(),              2,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_56 , pcs, types).mean(), 3.666666666666,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean(), 2.205072866904,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean(), 2.609699822522,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_6  , pcs, types).mean(),              4,1e-6);

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
		//BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(6)));

		BOOST_REQUIRE( the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE( the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE( the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE( the_tree.get_active_variables(2)[1]);

		
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_3  , pcs, types).mean(),             1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_4  , pcs, types).mean(),             2,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_56 , pcs, types).mean(),             3,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean(),2.205072866904,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean(),1.664825119989,1e-6);
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_6, pcs, types).mean()));
		

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
		//BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(5)));
		//BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(6)));

		BOOST_REQUIRE( the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE( the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[1]);

		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_3  , pcs, types).mean(),                1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_4  , pcs, types).mean(),                2,1e-6);
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_56, pcs, types).mean()));
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean(),  1.6648251199885,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean(),  1.6648251199885,1e-6);
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_6, pcs, types).mean()));

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
		//BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(3)));
		//BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(4)));
		//BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(5)));
		//BOOST_REQUIRE(std::isnan(the_tree.get_marginal_prediction(6)));

		BOOST_REQUIRE(!the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[1]);

		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_3  , pcs, types).mean()));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_4  , pcs, types).mean()));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_56 , pcs, types).mean()));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean()));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean()));
		BOOST_REQUIRE( std::isnan(the_tree.marginalized_prediction_stat(feature_6  , pcs, types).mean()));
    }
}
*/


BOOST_AUTO_TEST_CASE (legacy_fanova_test) {
	auto data = load_toy_data();
	data.set_type_of_feature(1, 3);

	rfr::trees::tree_options<num_type, response_t, index_t> tree_opts;
	tree_opts.max_features = 2;
	tree_opts.max_depth = 3;
	rng_t rng_engine(0);

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


		num_type s0 = 300;
		num_type s1 = 178.61191331472548;
		num_type s2 = 121.38808668527452;
		num_type s3 =  59.86622661388443;
		num_type s4 = 118.74568670084104;
		num_type s5 =  40.46269556175817;
		num_type s6 =  80.92539112351633;
		


		// check subspace sizes without cutoffs
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(0), s0, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(1), s1, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(2), s2, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(3), s3, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(4), s4, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(5), s5, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(6), s6, 1e-6);

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

		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_3  , pcs, types).mean(),              1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_4  , pcs, types).mean(),              2,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_56 , pcs, types).mean(), 3.666666666666,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean(), 2.205072866904,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean(), 2.609699822522,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_6  , pcs, types).mean(),              4,1e-6);
		
		{
			double m = (1*s3 + 2*s4+ 3*s5 + 4*s6)/(s3+s4+s5+s6);
			double v = ((1.-m)*(1.-m)*s3 + (2.-m)*(2.-m)*s4+ (3.-m)*(3.-m)*s5 + (4.-m)*(4.-m)*s6)/(s3+s4+s5+s6);
			BOOST_REQUIRE_CLOSE( the_tree.get_total_variance(), v, 1e-6);
		}
		

		// now let's exclude exactly one leaf
		the_tree.precompute_marginals(-inf, 3.5, pcs, types);
		// subspace don't change with cutoffs
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(0), s0, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(1), s1, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(2), s2, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(3), s3, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(4), s4, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(5), s5, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_subspace_size(6), s6, 1e-6);

		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(0), (s3*1 + s4*2 + s5*3+s6*3.5)/(s3+s4+s5+s6), 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(1), (s3*1 + s4*2)/(s3+s4), 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(2),  (s5*3+s6*3.5)/(s5+s6), 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(3),                  1, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(4),                  2, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(5),                  3 , 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(6),                  4 , 1e-6);

		BOOST_REQUIRE( the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE( the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE( the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE( the_tree.get_active_variables(2)[1]);
		
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_3  , pcs, types).mean(),             1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_4  , pcs, types).mean(),             2,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_56 , pcs, types).mean(),(s5*3+s6*3.5)/(s5+s6),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean(),(s3*1./3.+s4*2./3.+s5*3)/(s3/3.+s4/3.+s5),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean(),(s3*1./3.+s4*2./3.+s6*3.5/2.)/(s3/3.+s4/3.+s6/2.),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_6  , pcs, types).mean(),           3.5,1e-6);
		

		{
			double m = (1*s3 + 2*s4+ 3*s5 + 3.5*s6)/(s3+s4+s5+s6);
			double v = ((1.-m)*(1.-m)*s3 + (2.-m)*(2.-m)*s4+ (3.-m)*(3.-m)*s5 + (3.5-m)*(3.5-m)*s6)/(s3+s4+s5+s6);
			BOOST_REQUIRE_CLOSE( the_tree.get_total_variance(), v, 1e-6);
		}



		// now let's exclude exactly two leaves that belong to the same internal node
		the_tree.precompute_marginals(-inf, 2.5, pcs, types);

		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(0), (s3*1 + s4*2 + s5*2.5+s6*2.5)/(s3+s4+s5+s6), 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(1), (s3*1 + s4*2)/(s3+s4), 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(2),  (s5*2.5+s6*2.5)/(s5+s6), 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(3),                  1, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(4),                  2, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(5),                  3 , 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(6),                  4 , 1e-6);


		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_3  , pcs, types).mean(),             1,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_4  , pcs, types).mean(),             2,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_56 , pcs, types).mean(),(s5*2.5+s6*2.5)/(s5+s6),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean(),(s3*1./3.+s4*2./3.+s5*2.5)/(s3/3.+s4/3.+s5),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean(),(s3*1./3.+s4*2./3.+s6*2.5/2.)/(s3/3.+s4/3.+s6/2.),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_6  , pcs, types).mean(),           2.5,1e-6);


		BOOST_REQUIRE( the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE( the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[1]);

		{
			double m = (1*s3 + 2*s4+ 2.5*s5 + 2.5*s6)/(s3+s4+s5+s6);
			double v = ((1.-m)*(1.-m)*s3 + (2.-m)*(2.-m)*s4+ (2.5-m)*(2.5-m)*s5 + (2.5-m)*(2.5-m)*s6)/(s3+s4+s5+s6);
			BOOST_REQUIRE_CLOSE( the_tree.get_total_variance(), v, 1e-6);
		}



		// now let's exclude all leaves, just to see what happens
		the_tree.precompute_marginals(2.25, 2.75, pcs, types);

		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(0), ((s3+s4)*2.25 + (s5+s6)*2.75)/(s3+s4+s5+s6), 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(0), (s1*2.25 + s2*2.75)/(s1+s2), 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(1),               2.25, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(2),               2.75, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(3),                  1, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(4),                  2, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(5),                  3, 1e-6);
		BOOST_REQUIRE_CLOSE(the_tree.get_marginal_prediction(6),                  4, 1e-6);

		BOOST_REQUIRE(!the_tree.get_active_variables(0)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(0)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(1)[1]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[0]);
		BOOST_REQUIRE(!the_tree.get_active_variables(2)[1]);

		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_3  , pcs, types).mean(),               2.25,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_4  , pcs, types).mean(),               2.25,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_56 , pcs, types).mean(),               2.75,1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean(),(s3*2.25/3.+s4*2.25/3.+s5*2.75)/(s3/3.+s4/3.+s5),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_345, pcs, types).mean(),(s1*2.25/3.+s5*2.75)/(s1/3.+s5),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean(),(s3*2.25/3.+s4*2.25/3.+s6*2.75/2.)/(s3/3.+s4/3.+s6/2.),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_346, pcs, types).mean(),(s1*2.25/3.+s6*2.75/2.)/(s1/3.+s6/2.),1e-6);
		BOOST_REQUIRE_CLOSE( the_tree.marginalized_prediction_stat(feature_6  , pcs, types).mean(),               2.75,1e-6);

		{
			double m = (2.25*s3 + 2.25*s4+ 2.75*s5 + 2.75*s6)/(s3+s4+s5+s6);
			double v = ((2.25-m)*(2.25-m)*s3 + (2.25-m)*(2.25-m)*s4+ (2.75-m)*(2.75-m)*s5 + (2.75-m)*(2.75-m)*s6)/(s3+s4+s5+s6);
			BOOST_REQUIRE_CLOSE( the_tree.get_total_variance(), v, 1e-6);
		}

    }
}
