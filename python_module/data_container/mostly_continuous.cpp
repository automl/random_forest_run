#ifndef PYRFR_DATA_CONTAINER_HPP
#define PYRFR_DATA_CONTAINER_HPP


#include<vector>


#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"


namespace pyrfr{ namespace data_container{


typedef double num_type;
typedef double response_type;
typedef unsigned int index_type;


/** \brief A data container for mostly continuous data.
 *
 *  It might happen that only a small fraction of all features is categorical.
 *  In that case it would be wasteful to store the type of every feature separately.
 *  Instead, this data_container only stores the non-continuous ones in a hash-map.
 */
class mostly_continuous_data : public rfr::mostly_continuous_data<num_type, response_type, index_type>{
  public:

	mostly_continuous_data (index_type number_of_features): rfr::mostly_continuous_data<num_type, response_type, index_type>(number_of_features) {}
  
	virtual bool add_data_point_numpy (boost::numpy::ndarray const & feature_vector, num_type response){
		pyrfr::check_array<num_type>(feature_vector, 1);
		num_type* feature_array = reinterpret_cast<num_type*>(feature_vector.get_data());
		return(add_data_point(feature_array, feature_vector.shape(0), response));
	}

};


}}//namespace rfr
#endif
