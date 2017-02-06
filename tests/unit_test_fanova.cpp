#include <boost/test/unit_test.hpp>

#include <random>
#include <numeric>
#include <cstring>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/xml.hpp>
#include <fstream>
#include <sstream>

#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/trees/binary_fanova_tree.hpp"


typedef double num_type;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;

typedef rfr::splits::binary_split_one_feature_rss_loss<num_type, response_t, index_t, rng_t> split_type;
typedef rfr::nodes::k_ary_node_full<2, split_type, num_type, response_t, index_t, rng_t> node_type;
typedef rfr::trees::binary_fANOVA_tree<node_type, num_type, response_t, index_t, rng_t> tree_type;


BOOST_AUTO_TEST_CASE (fanova_test) {

}