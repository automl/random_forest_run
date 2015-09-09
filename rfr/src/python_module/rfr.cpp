#include <iostream>
#include <random>
#include <memory>


#include <boost/python.hpp>
#include <boost/numpy.hpp>


#include "rfr/data_containers/numpy_simple_data_container.hpp"
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



class regression_forest_binary_rss_split{
  private:
    

    std::default_random_engine rng;
    forest_type* forest_ptr;


  public:

    unsigned int num_trees=10;
    unsigned int seed = 0;
    bool do_bootstrapping = true;
    unsigned int num_data_points_per_tree = 0;
    unsigned int max_num_nodes = 0;
    unsigned int max_depth = 0;
    unsigned int max_features_per_split = 0;
    unsigned int min_samples_to_split = 0;
    unsigned int min_samples_in_leaf = 0;
    response_type epsilon_purity = -1;



    ~regression_forest_binary_rss_split(){ delete forest_ptr;}
    
    void fit (	boost::numpy::ndarray const & features,
		boost::numpy::ndarray const & responses,
		boost::numpy::ndarray const & types
		){


	if (seed > 0) {rng.seed(seed); seed = 0;}


	// simple sanity checks
	check_array<num_type>(features, 2);
	check_array<response_type>(responses, 1);
	check_array<num_type>(types, 1);

	// create the data container
	rfr::numpy_simple_data_container<num_type, response_type, index_type> data(features, responses, types);


	// construct the forest_option object
	rfr::forest_options<num_type, response_type, index_type> forest_opts;

	// store all the tree related options
	forest_opts.num_data_points_per_tree = (num_data_points_per_tree > 0)? num_data_points_per_tree : data.num_data_points();
	
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


	delete forest_ptr;
	forest_ptr = new forest_type(forest_opts);

	forest_ptr->fit(data, rng);
    }


    boost::numpy::ndarray predict(boost::numpy::ndarray const & data_point){
	check_array<num_type>(data_point, 1);
	std::tuple< num_type, num_type > res = forest_ptr->predict_mean_std(reinterpret_cast<num_type*>(data_point.get_data()));

	boost::python::object tmp = boost::python::make_tuple(std::get<0>(res), std::get<1>(res));

	return(boost::numpy::array(tmp));
	
    }


    void save_latex_representation(const char * filename_template){
	forest_ptr->save_latex_representation(filename_template);
    }

    
};





BOOST_PYTHON_MODULE(rfr){
	boost::numpy::initialize();
	boost::python::def("print_array", print_array);

	boost::python::class_<regression_forest_binary_rss_split>("regression_forest")
	    .def("fit", &regression_forest_binary_rss_split::fit)
	    .def("predict", &regression_forest_binary_rss_split::predict)
	    .def("save_latex_representation", &regression_forest_binary_rss_split::save_latex_representation)
	    .def_readwrite("num_trees",			&regression_forest_binary_rss_split::num_trees)
	    .def_readwrite("seed", 			&regression_forest_binary_rss_split::seed)
	    .def_readwrite("do_bootstrapping", 		&regression_forest_binary_rss_split::do_bootstrapping)
	    .def_readwrite("num_data_points_per_tree", 	&regression_forest_binary_rss_split::num_data_points_per_tree )
	    .def_readwrite("max_num_nodes",		&regression_forest_binary_rss_split::max_num_nodes )
	    .def_readwrite("max_depth",			&regression_forest_binary_rss_split::max_depth)
	    .def_readwrite("max_features_per_split",	&regression_forest_binary_rss_split::max_features_per_split)
	    .def_readwrite("min_samples_to_split",	&regression_forest_binary_rss_split::min_samples_to_split)
	    .def_readwrite("min_samples_in_leaf",	&regression_forest_binary_rss_split::min_samples_in_leaf)
	    .def_readwrite("epsilon_purity",		&regression_forest_binary_rss_split::epsilon_purity);
	
}
