#include <boost/python.hpp>

// header to include also the 
#include "my_boost_numpy.hpp"






namespace pyrfr{


// example of accessing/changing numpy arrays inside a C++ function
char const* print_array( boost::numpy::ndarray const & array){

    
    if (array.get_dtype() != boost::numpy::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        boost::python::throw_error_already_set();
    }
    if (array.get_nd() != 2) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        boost::python::throw_error_already_set();
    }
    if (!(array.get_flags() & boost::numpy::ndarray::C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous");
        boost::python::throw_error_already_set();
    }

    // access the array and write the content into a string
    std::stringstream str;
    double * iter = reinterpret_cast<double*>(array.get_data());
    int rows = array.shape(0);
    int cols = array.shape(1);

    str<<"shape: ["<<rows<<","<<cols<<"]\n";

    for (auto i = 0; i<rows;i++){
	for (auto j = 0; j<cols;j++)
	    str<<*(iter+(i*cols)+j)<<" ";
	str<<"\n";
    }

    // write into the array directly from here
    for (auto i = 0; i<rows;i++){
	for (auto j = 0; j<cols;j++)
	    *(iter+(i*cols)+j) = i*cols+j;
    }

    return(str.str().c_str());
}



template<typename val_type>
void check_array ( boost::numpy::ndarray const & array, int num_dims){
    if (array.get_dtype() != boost::numpy::dtype::get_builtin<val_type>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        boost::python::throw_error_already_set();
    }
    if (array.get_nd() != num_dims) {
        PyErr_SetString(PyExc_TypeError, "Incorrect number of dimensions");
        boost::python::throw_error_already_set();
    }
    if (!(array.get_flags() & boost::numpy::ndarray::C_CONTIGUOUS)) {
        PyErr_SetString(PyExc_TypeError, "Array must be row-major contiguous");
        boost::python::throw_error_already_set();
    }
}

} // end of namespace pyrfr

#include "regression/regression.hpp"





BOOST_PYTHON_MODULE(rfr){
	namespace bp = boost::python;
	namespace bn = boost::numpy;
    
	bn::initialize();
	bp::def("print_array", pyrfr::print_array);


	// specify that this module is actually a package
	bp::object package = bp::scope();
	package.attr("__path__") = "rfr";


	// add submodules
	export_regression();
}




