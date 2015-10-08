#ifndef PYRFR_REGRESSION_HPP
#define PYRFR_REGRESSION_HPP


#include "binary_rss.cpp"
#include "../data_container/container.hpp"


void export_regression(){
    namespace bp = boost::python;

    // map the regression namespace to a sub-module
    // make "from mypackage.Util import <whatever>" work
    bp::object regressionModule(bp::handle<>(bp::borrowed(PyImport_AddModule("rfr.regression"))));

    // make "from mypackage import Util" work
    bp::scope().attr("regression") = regressionModule;

    // set the current scope to the new sub-module
    bp::scope regression_scope = regressionModule;



	void (pyrfr::regression::binary_rss::binary_rss::*f1) ( pyrfr::data_container::mostly_continuous_data<pyrfr_num_type, pyrfr_response_type_regression, pyrfr_index_type> &data) = &pyrfr::regression::binary_rss::binary_rss::fit;
	void (pyrfr::regression::binary_rss::binary_rss::*f2) ( pyrfr::data_container::numpy_data_container<pyrfr_num_type, pyrfr_response_type_regression, pyrfr_index_type> &data) = &pyrfr::regression::binary_rss::binary_rss::fit;
	void (pyrfr::regression::binary_rss::binary_rss::*f3) ( pyrfr::data_container::numpy_transposed_data_container<pyrfr_num_type, pyrfr_response_type_regression, pyrfr_index_type> &data) = &pyrfr::regression::binary_rss::binary_rss::fit;



    boost::python::class_<pyrfr::regression::binary_rss::binary_rss>("binary_rss")
	// add methods
	.def("fit",	f1)
	.def("fit",	f2)
	.def("fit",	f3)
	.def("predict",
		&pyrfr::regression::binary_rss::binary_rss::predict)

	.def("save_latex_representation",
		&pyrfr::regression::binary_rss::binary_rss::save_latex_representation)
	// add attribute variables
	.def_readwrite("num_trees",
			&pyrfr::regression::binary_rss::binary_rss::num_trees)
	.def_readwrite("seed",
			&pyrfr::regression::binary_rss::binary_rss::seed)
	.def_readwrite("do_bootstrapping",
			&pyrfr::regression::binary_rss::binary_rss::do_bootstrapping)
	.def_readwrite("num_data_points_per_tree",
			&pyrfr::regression::binary_rss::binary_rss::num_data_points_per_tree )
	.def_readwrite("max_num_nodes",
			&pyrfr::regression::binary_rss::binary_rss::max_num_nodes )
	.def_readwrite("max_depth",
			&pyrfr::regression::binary_rss::binary_rss::max_depth)
	.def_readwrite("max_features_per_split",
			&pyrfr::regression::binary_rss::binary_rss::max_features_per_split)
	.def_readwrite("min_samples_to_split",
			&pyrfr::regression::binary_rss::binary_rss::min_samples_to_split)
	.def_readwrite("min_samples_in_leaf",
			&pyrfr::regression::binary_rss::binary_rss::min_samples_in_leaf)
	.def_readwrite("epsilon_purity",
			&pyrfr::regression::binary_rss::binary_rss::epsilon_purity);

	/*


    boost::python::class_<pyrfr::regression::binary_rss::binary_rss_v2>("binary_rss_v2")
	// add methods
	.def("fit",
		&pyrfr::regression::binary_rss::binary_rss_v2::fit)

	.def("predict",
		&pyrfr::regression::binary_rss::binary_rss_v2::predict)

	.def("save_latex_representation",
		&pyrfr::regression::binary_rss::binary_rss_v2::save_latex_representation)

	// add attribute variables
	.def_readwrite("num_trees",
			&pyrfr::regression::binary_rss::binary_rss_v2::num_trees)
	.def_readwrite("seed",
			&pyrfr::regression::binary_rss::binary_rss_v2::seed)
	.def_readwrite("do_bootstrapping",
			&pyrfr::regression::binary_rss::binary_rss_v2::do_bootstrapping)
	.def_readwrite("num_data_points_per_tree",
			&pyrfr::regression::binary_rss::binary_rss_v2::num_data_points_per_tree )
	.def_readwrite("max_num_nodes",
			&pyrfr::regression::binary_rss::binary_rss_v2::max_num_nodes )
	.def_readwrite("max_depth",
			&pyrfr::regression::binary_rss::binary_rss_v2::max_depth)
	.def_readwrite("max_features_per_split",
			&pyrfr::regression::binary_rss::binary_rss_v2::max_features_per_split)
	.def_readwrite("min_samples_to_split",
			&pyrfr::regression::binary_rss::binary_rss_v2::min_samples_to_split)
	.def_readwrite("min_samples_in_leaf",
			&pyrfr::regression::binary_rss::binary_rss_v2::min_samples_in_leaf)
	.def_readwrite("epsilon_purity",
			&pyrfr::regression::binary_rss::binary_rss_v2::epsilon_purity);

	*/

}

#endif
