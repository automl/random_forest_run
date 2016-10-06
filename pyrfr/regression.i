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
%}

%feature("autodoc",1);

class std::default_random_engine{
	std::default_random_engine ();
	std::default_random_engine (unsigned int seed);
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

// NODES
// not necessary at this point

// TREES
%include "rfr/trees/tree_options.hpp"
%template(tree_opts) rfr::trees::tree_options<double, double, unsigned int>;
%include "rfr/trees/k_ary_tree.hpp"
%template(binary_rss_tree) rfr::trees::k_ary_random_tree<2, rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t, 128>, num_t, response_t, index_t, rng_t>;


// FOREST(S)
%include "rfr/forests/forest_options.hpp"
%template(forest_options) rfr::trees::forest_options<double, double, unsigned int>;

