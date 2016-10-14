%module regression


%{
#include <random>
#include "rfr/data_containers/data_container.hpp"
#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/split_base.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss_v2.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/forests/regression_forest.hpp"

typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;
typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128> binary_rss_split_t;
typedef rfr::trees::k_ary_random_tree<2, binary_rss_split_t, num_t, response_t, index_t, rng_t> binary_rss_tree_t;
%}


%include "docstrings.i"
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



%include "std_string.i"
%include "std_vector.i"
%template(num_vector) std::vector<double>;
%template(idx_vector) std::vector<unsigned int>;


%ignore rfr::*::serialize;



// DATA CONTAINERS
%include "rfr/data_containers/data_container.hpp";
%include "rfr/data_containers/mostly_continuous_data_container.hpp";

%template(data_base) rfr::data_containers::base<num_t, response_t, index_t>;
%template(data_container) rfr::data_containers::mostly_continuous_data<num_t, response_t, index_t>;


// SPLITS
%include "rfr/splits/split_base.hpp"
%ignore rfr::splits::k_ary_split_base<2, double, double, unsigned int, std::default_random_engine>::find_best_split;
%template(split_base)rfr::splits::k_ary_split_base< 2,double,double,unsigned int,std::default_random_engine>;
%template(data_info) rfr::splits::data_info_t<num_t, response_t, index_t>;

// don't wrap these internal functions, as the nested templates cause trouble
%ignore rfr::splits::binary_split_one_feature_rss_loss::find_best_split;
%ignore rfr::splits::binary_split_one_feature_rss_loss::best_split_continuous;
%ignore rfr::splits::binary_split_one_feature_rss_loss::best_split_categorical;
%include "rfr/splits/binary_split_one_feature_rss_loss_v2.hpp"
%template(binary_rss_split) rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128>;
typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128> binary_rss_split_t;


// NODES
// not necessary at this point

// TREES
%include "rfr/trees/tree_options.hpp"
%template(tree_opts) rfr::trees::tree_options<num_t, response_t, index_t>;

%include "rfr/trees/tree_base.hpp"
%include "rfr/trees/k_ary_tree.hpp"
%template(base_tree)       rfr::trees::tree_base<num_t, response_t, index_t, rng_t>;
%template(binary_rss_tree) rfr::trees::k_ary_random_tree<2, binary_rss_split_t, num_t, response_t, index_t, rng_t>;

typedef rfr::trees::k_ary_random_tree<2, rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128>, num_t, response_t, index_t, rng_t> binary_rss_tree_t;

// FOREST(S)
%include "rfr/forests/forest_options.hpp"
%template(forest_opts) rfr::forests::forest_options<num_t, response_t, index_t>;
%include "rfr/forests/regression_forest.hpp"
%template(binary_rss_forest) rfr::forests::regression_forest< binary_rss_tree_t, num_t, response_t, index_t, rng_t>;
