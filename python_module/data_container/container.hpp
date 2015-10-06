#include "mostly_continuous.cpp"



typedef unsigned int index_type;


void export_data_container(){
    namespace bp = boost::python;

    // map the regression namespace to a sub-module
    // make "from mypackage.Util import <whatever>" work
    bp::object dataModule(bp::handle<>(bp::borrowed(PyImport_AddModule("rfr.data_container"))));

    // make "from mypackage import Util" work
    bp::scope().attr("data_container") = dataModule;

    // set the current scope to the new sub-module
    bp::scope data_scope = dataModule;

    boost::python::class_<pyrfr::data_container::mostly_continuous_data>("mostly_continuous_data", boost::python::init<unsigned int>())
	// add methods
	.def("add_data_point",
		&pyrfr::data_container::mostly_continuous_data::add_data_point_numpy)
	.def("set_type_of_feature",
		&pyrfr::data_container::mostly_continuous_data::set_type_of_feature);
}
