#include <iostream>
#include <random>

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

typedef rfr::binary_split_one_feature_rss_loss<rng_type, num_type, response_type, index_type> split_type;
typedef rfr::k_ary_node<2, split_type, rng_type, num_type, response_type, index_type> node_type;
typedef rfr::temporary_node<num_type, index_type> tmp_node_type;

typedef rfr::k_ary_random_tree<2, split_type, rng_type, num_type, response_type, index_type> tree_type;
typedef rfr::regression_forest< tree_type, rng_type, num_type, response_type, index_type> forest_type;




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




void check_array ( boost::numpy::ndarray const & array, int num_dims){
    if (array.get_dtype() != boost::numpy::dtype::get_builtin<double>()) {
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





class binary_regression_forest_rss_split{
  private:
    
    rfr::forest_options<num_type, response_type, index_type> forest_opts;
    std::default_random_engine rng;
    forest_type * the_forest;

  public:

    binary_regression_forest_rss_split(	unsigned int num_trees,
					unsigned int seed = 0,
   					bool do_bootstrapping = false,
					unsigned int num_data_points_per_tree = 0,
					unsigned int max_num_nodes = 0,
					unsigned int max_depth = 0,
					unsigned int max_features_per_split = 0,
					unsigned int min_samples_to_split = 0,
					unsigned int min_samples_in_leaf = 0,
					response_type epsilon_purity = -1
					){

	if (seed > 0) rng.seed(seed);

	// store all the tree related options
	if (max_depth > 0) forest_opts.tree_opts.max_depth = max_depth;
	if (max_features_per_split > 0) forest_opts.tree_opts.max_features = max_features_per_split;
	if (min_samples_to_split > 0) forest_opts.tree_opts.min_samples_to_split = min_samples_to_split;
	if (min_samples_in_leaf > 0) forest_opts.tree_opts.min_samples_in_leaf = min_samples_in_leaf;
	if (max_num_nodes >  0) forest_opts.tree_opts.max_num_nodes = max_num_nodes;
	if (epsilon_purity >= 0) forest_opts.tree_opts.epsilon_purity = epsilon_purity;


	// now the forest related options
	forest_opts.num_trees = num_trees;
	forest_opts.do_bootstrapping = do_bootstrapping;
	if (num_data_points_per_tree > 0) forest_opts.num_data_points_per_tree = num_data_points_per_tree;

	the_forest = new forest_type (forest_opts);
	
    }

	~binary_regression_forest_rss_split(){ delete the_forest;}
					

    void fit (	boost::numpy::ndarray const & features,
		boost::numpy::ndarray const & responses,
		boost::numpy::ndarray const & types
		){

	// simple sanity checks
	check_array(features, 2);
	check_array(responses, 1);
	check_array(types, 1);

	// create the data container


	
	

    }



};



#include <boost/python.hpp>

BOOST_PYTHON_MODULE(rfr){
	boost::numpy::initialize();
	boost::python::def("print_array", print_array);
}
