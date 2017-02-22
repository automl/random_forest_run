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
  data.set_type_of_feature(1,10);

  BOOST_REQUIRE_EQUAL(data.num_data_points(), 100);

  return(data);
}


BOOST_AUTO_TEST_CASE (fanova_test) {
    auto data = load_toy_data();
    data.set_type_of_feature(1, 4);

    rfr::trees::tree_options<num_type, response_t, index_t> tree_opts;
    tree_opts.max_features = 2;
    tree_opts.max_depth = 3;
    rng_t rng_engine;

    for (auto i = 0; i <1; i++){
      fANOVA_tree_type the_tree;
      std::vector<std::vector<num_type>> pcs = {{1, 2}, {0, 1, 2, 3, 4}, {4, 20}};
      std::vector<index_t> types = {0, 5, 0};
      num_type inf = std::numeric_limits<num_type>::infinity();

      auto nodes = the_tree.get_nodes();
      the_tree.precompute_marginals(-inf, inf, pcs, types);
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

      // the_tree.fit(data, tree_opts, std::vector<num_type>(data.num_data_points(), 1), rng_engine);

      char filename[100];
      sprintf(filename, "/tmp/tree_%i.tex", i);
      // the_tree.save_latex_representation(filename); // FIXME(mostafa): error here
    }
}
