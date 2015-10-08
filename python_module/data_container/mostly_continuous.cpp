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

	void import_numpy_arrays(boost::numpy::ndarray const & features, boost::numpy::ndarray const &responses){
		pyrfr::check_array<num_type>(features, 2);
		pyrfr::check_array<num_type>(responses, 1);

		if (features.shape(1) != this->num_features()){
			PyErr_SetString(PyExc_NotImplementedError, "The features matix has not the correct number of columns.");
			boost::python::throw_error_already_set();
		}

		if (features.shape(0) != responses.shape(0)){
			PyErr_SetString(PyExc_NotImplementedError, "The features matix and the responses have not the same number of data points.");
			boost::python::throw_error_already_set();
		}

		num_type * feats =  reinterpret_cast<num_type*>(features.get_data());
		num_type * resp =  reinterpret_cast<num_type*>(responses.get_data());

	    
		// make sure the data vectors have only to be reallocated once
		for (auto &v: this->feature_values)
			v.reserve(v.size()+ features.shape(0));
		this->response_values.reserve(this->num_data_points()+features.shape(0));

		for (auto i=0u; i < features.shape(0); i++){
			this->add_data_point(feats+i*this->num_features(), this->num_features(), resp[i]);
		}
	}



	boost::numpy::ndarray export_features (){
		boost::python::tuple shape = boost::python::make_tuple(this->num_data_points(), this->num_features());
		boost::numpy::dtype dtype = boost::numpy::dtype::get_builtin<num_type>();
		boost::numpy::ndarray ret = boost::numpy::empty(shape, dtype);
		for (auto i=0u; i < this->num_features(); i++)
			for (auto j=0u; j<this->num_data_points(); j++)
				ret[j][i] = this->feature_values[i][j];
		return(ret);
	}

	boost::numpy::ndarray export_responses (){
		boost::python::tuple shape = boost::python::make_tuple(this->num_data_points());
		boost::numpy::dtype dtype = boost::numpy::dtype::get_builtin<response_type>();

		boost::numpy::ndarray ret = boost::numpy::from_data( this->response_values.data(), dtype, shape,  boost::python::make_tuple(sizeof(response_type)), boost::python::object());

		return(ret);
	}
	
};


}}//namespace rfr
#endif
