#ifndef RFR_NUMPY_TRANSPOSED_CONTAINER_HPP
#define RFR_NUMPY_TRANSPOSED_CONTAINER_HPP


#include <algorithm>
#include <vector>
#include <cmath>

#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include <rfr/data_containers/data_container_base.hpp>


namespace pyrfr{ namespace data_container{

template <typename num_type, typename response_type, typename index_type>
class numpy_transposed_data_container : public rfr::data_container_base<num_type, response_type, index_type>{

  private:
	index_type n_data_points;
	index_type n_features;
	num_type * feature_array;
	response_type * response_array;
	index_type * type_array;

  public:

	numpy_transposed_data_container(boost::numpy::ndarray const & features,
									boost::numpy::ndarray const & responses,
									boost::numpy::ndarray const & types){

		n_data_points = features.shape(1);
		n_features = features.shape(0);

		if (n_data_points != responses.shape(0) ) {
			PyErr_SetString(PyExc_ValueError, "Number of responses does not match number of datapoints provided.");
			boost::python::throw_error_already_set();
		}


		if (n_features != types.shape(0)) {
			PyErr_SetString(PyExc_ValueError, "Number of features and number of types defined do not match.");
			boost::python::throw_error_already_set();
		}

		feature_array = reinterpret_cast<num_type *>(features.get_data());
		response_array= reinterpret_cast<response_type *>(responses.get_data());
		type_array    = reinterpret_cast<index_type *>(types.get_data());

		// make sure the types are properly rounded
		std::transform(type_array, type_array+n_features, type_array, round);

		for (auto i=0u; i<n_features; i++){
			if (types[i] > 0.5){
				for (auto j=0u; j < n_data_points; j++)
					feature_array[i*n_data_points + j] = std::round(feature_array[i*n_data_points + j]);
			}
		}
	}
  
	virtual num_type feature (index_type feature_index, index_type sample_index) const {
		return (feature_array[sample_index + n_data_points*feature_index]);
	}


	virtual std::vector<num_type> features (index_type feature_index, std::vector<index_type> &sample_indices) const {
		std::vector<num_type> rv;
		rv.reserve(sample_indices.size());

		for (auto i: sample_indices)
			rv.push_back(feature_array[i+n_data_points*feature_index]);

		return(rv);
	}

	
	virtual response_type response (index_type sample_index) const{
		return (response_array[sample_index]);
	}
	
	virtual bool add_data_point (num_type* features, index_type num_elements, response_type response){
			PyErr_SetString(PyExc_NotImplementedError, "This data container does not support this functionality.");
			boost::python::throw_error_already_set();
			return(false);	
	}
	
	
	virtual std::vector<num_type> retrieve_data_point (index_type index){
		std::vector<num_type> rv(n_features);
		for (auto i = 0u; i < rv.size(); i++)
			rv[i] = feature_array[index*n_features + i];
		return(rv);
	}
	
	
	virtual index_type get_type_of_feature (index_type feature_index) const{
		return(type_array[feature_index]);		
	}

	virtual bool set_type_of_feature (index_type feature_index, index_type feature_type){
		PyErr_SetString(PyExc_NotImplementedError, "This data container does not support this functionality.");
		boost::python::throw_error_already_set();
		return(true);
	}

	virtual index_type num_features() const {return(n_features);}
	virtual index_type num_data_points()  const {return(n_data_points);}
};

}} // namespace pyrfr
#endif // RFR_NUMPY_TRANSPOSED_CONTAINER_HPP
