#ifndef RFR_ARRAY_CONTAINER_HPP
#define RFR_ARRAY_CONTAINER_HPP


#include <cmath>
#include <stdexcept>
//#include <rfr/data_containers/data_container_base.hpp>
#include <rfr/data_containers/data_container.hpp>


namespace rfr{ namespace data_containers{

template <typename num_t, typename response_t, typename index_t>
class array_data_container : public rfr::data_containers::data_containers::base<num_t, response_t, index_t>{

  private:
	index_t n_data_points;
	index_t n_features;
	num_t * feature_array;
	response_t * response_array;
	index_t * type_array;
	index_t response_t;

  public:

	array_data_container( 	num_t      * features,
							response_t * responses,
							index_t    * types,
							index_t      n_data_points,
							index_t      n_features):

								n_data_points(n_data_points),
								n_features(n_features),
								feature_array(features),
								response_array(responses),
								type_array(types),
								response_t(0){

		//properly round the feature values for categorical features
		for (auto i=0u; i<n_features; i++){
			if (types[i] > 0){
				for (auto j=0u; j < n_data_points; j++)
					feature_array[j*n_features + i] = std::round(feature_array[j*n_features + i]);
			}
		}
	}

	virtual ~array_data_container() {};
  
	virtual num_t feature (index_t feature_index, index_t sample_index) const {
		return (feature_array[sample_index*n_features + feature_index]);
	}


	virtual std::vector<num_t> features (index_t feature_index, std::vector<index_t> &sample_indices) const {
		std::vector<num_t> rv;
		rv.reserve(sample_indices.size());

		for (auto i: sample_indices)
			rv.push_back(feature_array[i*n_features+feature_index]);

		return(rv);
	}

	
	virtual response_t response (index_t sample_index) const{
		return (response_array[sample_index]);
	}
	
	virtual void add_data_point (num_t* features, index_t num_elements, response_t response){
		throw std::runtime_error("Array data containers do not support adding new data points.");
	}
	
	virtual std::vector<num_t> retrieve_data_point (index_t index) const{
		std::vector<num_t> rv(n_features);
		for (auto i = 0u; i < rv.size(); i++)
			rv[i] = feature_array[index*n_features + i];
		return(rv);
	}
	
	virtual index_t get_type_of_feature (index_t feature_index) const{
		return(type_array[feature_index]);		
	}

	virtual void set_type_of_feature (index_t feature_index, index_t feature_type){
		throw std::runtime_error("Array data containers do not support changing a feature type.");
	}


	virtual index_t get_type_of_response () const{
		return(response_t);		
	}

	virtual void set_type_of_response (index_t resp_t){
		if (resp_t > 0){
			for (auto i=0u; i < n_data_points; i++){
				if (!(response_array[i] < resp_t))
					throw std::runtime_error("Response value not consistent with provided type. Data contains a value larger than allowed.");
				if (response_array[i] < 0)
					throw std::runtime_error("Response values contain a negative value, can't make that a categorical value.");
			}
		response_t = resp_t;
		}
	}


	virtual index_t num_features() const {return(n_features);}
	virtual index_t num_data_points()  const {return(n_data_points);}

	virtual void normalize_data(){
		std::vector<num_t> features;
		num_t min = std::numeric_limits<num_t>::max();
		num_t max = std::numeric_limits<num_t>::lowest();
		for(int i = 0; i<n_features; i++){
			if (types[i] > 0){
				for(int j = 0; j<n_data_points; j++){
					min = std::min(min, feature_array[j*n_features + i]);
					max = std::max(max, feature_array[j*n_features + i]);
				}
			}
		}

		for(int i = 0; i<n_features; i++){
			if (types[i] > 0){
				for(int j = 0; j<n_data_points; j++){
					feature_array[j*n_features + i] = (feature_array[j*n_features + i]-min)/(max-min);
				}
			}
		}
	}
};

}} // namespace rfr
#endif // RFR_ARRAY_CONTAINER_HPP
