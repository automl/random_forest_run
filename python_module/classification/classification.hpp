#include "class_split.cpp"



void export_classification(){
    namespace bp = boost::python;

    // map the classification namespace to a sub-module
    // make "from mypackage.Util import <whatever>" work
    bp::object classificationModule(bp::handle<>(bp::borrowed(PyImport_AddModule("rfr.classification"))));

    // make "from mypackage import Util" work
    bp::scope().attr("classification") = classificationModule;

    // set the current scope to the new sub-module
    bp::scope classification_scope = classificationModule;

    boost::python::class_<pyrfr::classification::class_split::class_split>("class_split")
	// add methods
	.def("fit",
		&pyrfr::classification::class_split::class_split::fit)

	.def("predict",
		&pyrfr::classification::class_split::class_split::predict)

	.def("save_latex_representation",
		&pyrfr::classification::class_split::class_split::save_latex_representation)

	// add attribute variables
	.def_readwrite("num_trees",
			&pyrfr::classification::class_split::class_split::num_trees)
	.def_readwrite("seed",
			&pyrfr::classification::class_split::class_split::seed)
	.def_readwrite("do_bootstrapping",
			&pyrfr::classification::class_split::class_split::do_bootstrapping)
	.def_readwrite("num_data_points_per_tree",
			&pyrfr::classification::class_split::class_split::num_data_points_per_tree )
	.def_readwrite("max_num_nodes",
			&pyrfr::classification::class_split::class_split::max_num_nodes )
	.def_readwrite("max_depth",
			&pyrfr::classification::class_split::class_split::max_depth)
	.def_readwrite("max_features_per_split",
			&pyrfr::classification::class_split::class_split::max_features_per_split)
	.def_readwrite("min_samples_to_split",
			&pyrfr::classification::class_split::class_split::min_samples_to_split)
	.def_readwrite("min_samples_in_leaf",
			&pyrfr::classification::class_split::class_split::min_samples_in_leaf)
	.def_readwrite("epsilon_purity",
			&pyrfr::classification::class_split::class_split::epsilon_purity);
}

