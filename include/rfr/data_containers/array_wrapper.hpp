#ifndef RFR_ARRAY_CONTAINER_HPP
#define RFR_ARRAY_CONTAINER_HPP


#include <cmath>
#include <stdexcept>
#include <rfr/data_containers/data_container_base.hpp>


namespace rfr{ namespace data_containers{

template <typename num_type, typename response_type, typename index_type>
class array_data_container : public rfr::data_containers::data_container_base<num_type, response_type, index_type>{

  private:
	index_type n_data_points;
	index_type n_features;
	num_type * feature_array;
	response_type * response_array;
	index_type * type_array;

  public:

	array_data_container( 	num_type      * features,
							response_type * responses,
							index_type    *  types,
							index_type      n_data_points,
							index_type      n_features):

								n_data_points(n_data_points),
								n_features(n_features),
								feature_array(features),
								response_array(responses),
								type_array(types){

		//properly round the feature values for categorical features
		for (auto i=0u; i<n_features; i++){
			if (types[i] > 0){
				for (auto j=0u; j < n_data_points; j++)
					feature_array[j*n_features + i] = std::round(feature_array[j*n_features + i]);
			}
		}
	}

	virtual ~array_data_container() {};
  
	virtual num_type feature (index_type feature_index, index_type sample_index) const {
		return (feature_array[sample_index*n_features + feature_index]);
	}


	virtual std::vector<num_type> features (index_type feature_index, std::vector<index_type> &sample_indices) const {
		std::vector<num_type> rv;
		rv.reserve(sample_indices.size());

		for (auto i: sample_indices)
			rv.push_back(feature_array[i*n_features+feature_index]);

		return(rv);
	}

	
	virtual response_type response (index_type sample_index) const{
		return (response_array[sample_index]);
	}
	
	virtual bool add_data_point (num_type* features, index_type num_elements, response_type response){
		throw std::runtime_error("Array data containers do not support adding new data points.");
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
		throw std::runtime_error("Array data containers do not support changing a feature type.");
		return(false);
	}

	virtual index_type num_features() const {return(n_features);}
	virtual index_type num_data_points()  const {return(n_data_points);}
};

}} // namespace rfr
#endif // RFR_ARRAY_CONTAINER_HPP
