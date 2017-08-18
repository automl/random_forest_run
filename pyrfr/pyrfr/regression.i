%module regression

%pythonnondynamic;

%{
#include <random>
#include "rfr/data_containers/data_container.hpp"
#include "rfr/data_containers/default_data_container.hpp"
#include "rfr/data_containers/default_data_container_with_instances.hpp"
#include "rfr/splits/split_base.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/forests/regression_forest.hpp"
#include "rfr/forests/quantile_regression_forest.hpp"
#include "rfr/forests/fanova_forest.hpp"


// put typedefs here for later use when specifying templates
typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;
typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128> binary_rss_split_t;
typedef rfr::nodes::k_ary_node_minimal<2, rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128>, num_t, response_t, index_t, rng_t> binary_minimal_node_rss_t;
typedef rfr::nodes::k_ary_node_full<2, rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128>, num_t, response_t, index_t, rng_t> binary_full_node_rss_t;

typedef rfr::trees::k_ary_random_tree<2, binary_full_node_rss_t, num_t, response_t, index_t, rng_t> binary_full_tree_rss_t;
typedef rfr::trees::binary_fANOVA_tree< binary_rss_split_t,num_t,response_t,index_t,rng_t > binary_fanova_tree_t;
%}


%include "docstrings.i"


// vanilla exeption handling for everything
%include "exception.i" 
%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  } catch (const std::string& e) {
    SWIG_exception(SWIG_RuntimeError, e.c_str());
  }
} 


class std::default_random_engine{
	public:
		default_random_engine ();
		default_random_engine (unsigned int seed);
		void seed (unsigned int);
};

typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;


// generate std templates here
%include "std_string.i"
%include "std_vector.i"
%include "std_pair.i"

%template(num_vector) std::vector<num_t>;
%template(idx_vector) std::vector<index_t>;
%template(num_vector_vector) std::vector< std::vector<num_t> >;
%template(num_vector_vector_vector) std::vector<std::vector< std::vector<num_t> > >;
%template(num_num_pair) std::pair<num_t, num_t>;


// put everything here that should be ignored globally
%ignore rfr::*::serialize;



// DATA CONTAINERS
%include "rfr/data_containers/data_container.hpp";
%include "rfr/data_containers/default_data_container.hpp";
%include "rfr/data_containers/default_data_container_with_instances.hpp";

%template(data_base) rfr::data_containers::base<num_t, response_t, index_t>;
%template(default_data_container) rfr::data_containers::default_container<num_t, response_t, index_t>;
%template(default_data_container_with_instances) rfr::data_containers::default_container_with_instances<num_t, response_t, index_t>;


// SPLITS
// Turns out, nothing needs to be instantiated here in order to use it later!
// But I keep the code for later reference :)
%include "rfr/splits/split_base.hpp"
//%ignore rfr::splits::k_ary_split_base<2, double, double, unsigned int, std::default_random_engine>::find_best_split;
//%template(split_base)rfr::splits::k_ary_split_base< 2,double,double,unsigned int,std::default_random_engine>;
//%template(data_info) rfr::splits::data_info_t<num_t, response_t, index_t>;

// don't wrap these internal functions, as the nested templates cause trouble
//%ignore rfr::splits::binary_split_one_feature_rss_loss::find_best_split;
//%ignore rfr::splits::binary_split_one_feature_rss_loss::best_split_continuous;
//%ignore rfr::splits::binary_split_one_feature_rss_loss::best_split_categorical;
//%include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
//%template(binary_rss_split) rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128>;
//typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128> binary_rss_split_t;


// NODES
%include "rfr/nodes/k_ary_node.hpp"
typedef rfr::nodes::k_ary_node_full<2, rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128>, num_t, response_t, index_t, rng_t> binary_full_node_rss_t;

// TREES
%include "rfr/trees/tree_options.hpp"

%template(tree_opts) rfr::trees::tree_options<num_t, response_t, index_t>;

%include "rfr/trees/tree_base.hpp"
%template(base_tree)       rfr::trees::tree_base<num_t, response_t, index_t, rng_t>;

%include "rfr/trees/k_ary_tree.hpp"
%template(binary_full_tree_rss) rfr::trees::k_ary_random_tree<2, binary_full_node_rss_t, num_t, response_t, index_t, rng_t>;
typedef rfr::trees::k_ary_random_tree<2,rfr::nodes::k_ary_node_full<2, binary_rss_split_t, num_t, response_t, index_t, rng_t>, num_t, response_t, index_t, rng_t> binary_full_tree_rss_t;

%include "rfr/trees/binary_fanova_tree.hpp"
typedef rfr::trees::binary_fANOVA_tree< binary_rss_split_t,num_t,response_t,index_t,rng_t > binary_fanova_tree_t;

// FOREST(S)
%include "rfr/forests/forest_options.hpp"
%template(forest_opts) rfr::forests::forest_options<num_t, response_t, index_t>;
%include "rfr/forests/regression_forest.hpp"
%template(binary_rss_forest) rfr::forests::regression_forest< binary_full_tree_rss_t, num_t, response_t, index_t, rng_t>;


%include "rfr/forests/quantile_regression_forest.hpp"
%template(qr_forest) rfr::forests::quantile_regression_forest< binary_full_tree_rss_t, num_t, response_t, index_t, rng_t>;

%include "rfr/forests/fanova_forest.hpp"
%template(fanova_forest_prototype) rfr::forests::regression_forest< binary_fanova_tree_t,num_t, response_t, index_t, rng_t >; 
%template(fanova_forest) rfr::forests::fANOVA_forest<binary_rss_split_t, num_t, response_t, index_t, rng_t>;


// adds required members to make the forests 'pickable'
// note: the forest is stored in an ASCII string as the default translation
// from std::string (raw bytes) to a Python string changes the encoding, and
// I couldn't find an easy way around that.
// Not sure if the pickle module compresses the data, if not that could be added here using the zlib module.

%extend rfr::forests::regression_forest< binary_full_tree_rss_t, num_t, response_t, index_t, rng_t>{
	 %pythoncode %{
		def __getstate__(self):
			d = {}
			d['str_representation'] = self.ascii_string_representation()
			return (d)
		
		def __setstate__(self, sState):
			self.__init__()
			self.load_from_ascii_string(sState['str_representation'])
      %}
};
