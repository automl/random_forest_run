#include <iostream>
#include <random>
#include <memory>


#include <boost/python.hpp>
#include <boost/numpy.hpp>


#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/trees/k_ary_tree.hpp"
//#include "rfr/forests/forest_options.hpp"

#include "regression_forest_base.hpp"



namespace pyrfr{ namespace regression { namespace binary_rss{ 

typedef rfr::binary_split_one_feature_rss_loss< pyrfr_rng_type,  pyrfr_num_type,  pyrfr_response_type_regression,  pyrfr_index_type> split_type;
typedef rfr::k_ary_node<2, split_type,  pyrfr_rng_type,  pyrfr_num_type,  pyrfr_response_type_regression,  pyrfr_index_type> node_type;
typedef rfr::temporary_node< pyrfr_num_type,  pyrfr_index_type> tmp_node_type;

typedef rfr::k_ary_random_tree<2, split_type,  pyrfr_rng_type,  pyrfr_num_type,  pyrfr_response_type_regression,  pyrfr_index_type> tree_type;
typedef rfr::regression_forest< tree_type,  pyrfr_rng_type,  pyrfr_num_type,  pyrfr_response_type_regression,  pyrfr_index_type> forest_type;


class binary_rss : public  pyrfr::regression::regression_forest_base<forest_type, pyrfr_rng_type, pyrfr_num_type, pyrfr_response_type_regression, pyrfr_index_type>{
  public:
    boost::numpy::ndarray predict(boost::numpy::ndarray const & data_point){
	check_array<pyrfr_num_type>(data_point, 1);
	std::tuple< pyrfr_num_type, pyrfr_num_type > res = forest_ptr->predict_mean_std(reinterpret_cast<pyrfr_num_type*>(data_point.get_data()));

	boost::python::object tmp = boost::python::make_tuple(std::get<0>(res), std::get<1>(res));

	return(boost::numpy::array(tmp));
    }
};


}}}// of namespace pyrfr::regression::binary_rss
