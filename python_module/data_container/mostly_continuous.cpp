#ifndef PYRFR_MC_DATA_CONTAINER_HPP
#define PYRFR_MC_DATA_CONTAINER_HPP


#include<vector>


#include <boost/python.hpp>
#include <boost/numpy.hpp>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"


namespace pyrfr{ namespace data_container{

/** \brief A data container for mostly continuous data.
 *
 *  It might happen that only a small fraction of all features is categorical.
 *  In that case it would be wasteful to store the type of every feature separately.
 *  Instead, this data_container only stores the non-continuous ones in a hash-map.
 */

template <typename num_type,typename response_type,typename index_type>
class mostly_continuous_data : public rfr::mostly_continuous_data<num_type, response_type, index_type>{
  public:

	// constructor setting the number of features calling the base class' constructor
	mostly_continuous_data (index_type number_of_features): rfr::mostly_continuous_data<num_type, response_type, index_type>(number_of_features) {}
  
	virtual bool add_data_point_numpy (boost::numpy::ndarray const & feature_vector, num_type response){
		pyrfr::check_array<num_type>(feature_vector, 1);
		num_type* feature_array = reinterpret_cast<num_type*>(feature_vector.get_data());
		return(this->add_data_point(feature_array, feature_vector.shape(0), response));
	}

};


}}//namespace rfr
#endif
