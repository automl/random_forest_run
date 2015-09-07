//#include "rfr/forest/regression_forest.hpp"

#include <boost/numpy.hpp>


#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/forests/regression_forest.hpp"
#include "rfr/forests/forest_options.hpp"



typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;
typedef std::default_random_engine rng_type;

typedef rfr::mostly_contiuous_data<num_type, response_type, index_type> data_container_type;

typedef rfr::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type;
typedef rfr::k_ary_node<2, split_type, rng_type, num_type, response_type, index_type> node_type;
typedef rfr::temporary_node<num_type, index_type> tmp_node_type;

typedef rfr::k_ary_random_tree<2, split_type, rng_type, num_type, response_type, index_type> tree_type;



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






class binary_regression_forest_rss_split{
  private:
    tree_type the_tree;

  public:

    // constructor
    binary_regression_forest_rss_split(){

	rfr::tree_options tree_opts();
	rfr::forest_options forest_opts(tree_opts);

	the_tree = tree_type (tree_opts);

    }



};






#include <boost/python.hpp>

BOOST_PYTHON_MODULE(librfr){
	boost::numpy::initialize();
	boost::python::def("greet", greet);
	boost::python::def("print_array", print_array);
}
