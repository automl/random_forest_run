#ifndef RFR_MOSTLY_CONTINUOUS_DATA_CONTAINER_HPP
#define RFR_MOSTLY_CONTINUOUS_DATA_CONTAINER_HPP


#include<vector>
#include<map>


#include "rfr/data_containers/data_container_base.hpp"
#include "rfr/data_containers/data_container_utils.hpp"


namespace rfr{ namespace data_containers{

/** \brief A data container for mostly continuous data with instances.
 *
 * Similar to the mostly_continuous_data container, but with the capability
 * to handle instance features.
 */
template<typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class mostly_continuous_data_with_instances : public rfr::data_containers::data_container_base<num_type, response_type, index_type>{
  protected:
  
	std::vector< std::vector<num_type> > configurations;//!< 2d vector to store the feature values of all configurations
	std::vector< std::vector<num_type> > instances; 	//!< 2d vector to store the feature values of all instances
	
	std::vector<std::pair<index_type, index_type> > config_instance_pairs;
	std::vector<num_type> response_values;
	std::map<index_type, index_type> categorical_ranges;//!< a map storing the few categorical indices and their range
	index_type response_t;
  public:

	// empty constructor. Use this only if you read the data from a file!
	// the private vectors are not properly initialized! Adding data
	// points via 'add_data_point' may or may not fail!
	mostly_continuous_data_with_instances() {
		throw std::runtime_error("The empty constructor is not supported by this container.");
		}

	// if you plan on filling the container with single data points one at a time
	// use this constructor to specify the number of features for configurations and instances
	mostly_continuous_data (index_type num_config_f, index_type num_instance_f):
		configurations(num_config_f, std::vector<num_type>(0)),
		instances(num_instance_f, std::vector<num_type>(0)),
		response_t(0){}
  
	virtual num_type feature  (index_type feature_index, index_type sample_index) const {
		// find out if this is a config or instance feature
		
		throw std::runtime_error("Not implemented yet.");
		return(feature_values[feature_index][sample_index]);
	}


	virtual std::vector<num_type> features (index_type feature_index, std::vector<index_type> &sample_indices) const {
		throw std::runtime_error("Not implemented yet.");
		std::vector<num_type> rv;
		rv.reserve(sample_indices.size());
		for (auto i : sample_indices)
			rv.push_back(feature_values[feature_index][i]);
		return(rv);
	}


	virtual response_type response (index_type sample_index) const{
		return(response_values[sample_index]);
	}

	virtual void add_data_point (num_type* config_feats, index_type num_config_elements, num_type* instance_feats, index_type num_instance_elements, response_type response){

		throw std::runtime_error("Not implemented yet.");

		if (num_features() != num_elements)
					throw std::runtime_error("Number of elements does not match.");

		for (size_t i=0; i<num_elements; i++){
			if (get_type_of_feature(i) > 0){
				if (feats[i] >= get_type_of_feature(i))
					throw std::runtime_error("Feature values not consistent with provided type. Data contains a value larger than allowed.");
				if (feats[i]< 0)
					throw std::runtime_error("Feature values contain a negative value, can't make that a categorical feature.");
			}
		}

		for (size_t i=0; i<num_elements; i++)
				feature_values[i].push_back(feats[i]);

		response_values.push_back(response);
	}

	virtual std::vector<num_type> retrieve_data_point (index_type index){
		throw std::runtime_error("Not implemented yet.");
		std::vector<num_type> vec(feature_values.size());
		for (index_type i = 0; i < num_features(); i++)
			vec[i] = feature_values[i].at(index);
		return(vec);
	}

	/** \brief method to query the type of a feature
	 *
	 * As most features are assumed to be numerical, it is actually
	 * beneficial to store only the categorical exceptions in a hash-map.
	 * Type = 0 means continuous, and Type = n >= 1 means categorical with
	 * options \in {1, n}. For consistency, we exclude zero from the categorical
	 * values if anyone wants to add sparse data later on.
	 *
	 * \param feature_index the index of the feature
	 * \return int type of the feature: 0 - numerical value (float or int); n>0 - categorical value with n different values {1,2,...,n}
	 *
	 */
	virtual index_type get_type_of_feature (index_type feature_index) const{
		auto it = categorical_ranges.find(feature_index);
		if ( it == categorical_ranges.end())
			return(0);
		return(it->second);
	}


	virtual void set_type_of_feature(index_type index, index_type type){
		if (index >= num_features())
			throw std::runtime_error("Unknown index specified.");
		if (type < 0)
			throw std::runtime_error("Type value should be >= 0");

		if (type > 0){
			//check if the data so far is consistent with the choice
			for (auto &fv: feature_values[index]){
				if (!(fv<type))
					throw std::runtime_error("Feature values not consistent with provided type. Data contains a value larger than allowed.");
				if (fv < 0)
					throw std::runtime_error("Feature values contain a negative value, can't make that a categorical feature.");
				}


			categorical_ranges[index] = type;
		}
		else{
			
		}
	}

	virtual index_type num_features() const {return(feature_values.size());}
	virtual index_type num_data_points() const {return(feature_values[0].size());}


	// some helper functions
	int read_feature_file (const char* filename){
		feature_values =  rfr::read_csv_file<num_type>(filename);
		return(feature_values.size());
	}

	int read_response_file (const char* filename){
		response_values = (read_csv_file<num_type>(filename))[0];
		return(response_values.size());
	}

	bool check_consistency(){

		/* number of stored values has to be the same for:
		*  1. the response values
		*  2. every (row) vector in feature_values
		*/

		index_type num_data_points = response_values.size();
		for (auto it = feature_values.begin(); it != feature_values.end(); it ++){
			if (num_data_points != it->size())
				return(false);
		}
		return(true);
	}

	virtual index_type get_type_of_response () const{
		return(response_t);		
	}

	virtual void set_type_of_response (index_type resp_t){
		if (resp_t > 0){
			for (auto &rv: response_values){
				if (!(rv < resp_t))
					throw std::runtime_error("Response value not consistent with provided type. Data contains a value larger than allowed.");
				if (rv < 0)
					throw std::runtime_error("Response values contain a negative value, can't make that a categorical value.");
			}
			response_t = resp_t;
		}
	}


	void print_data(){
		for (auto i = 0u; i < feature_values.size(); i++){
			for (auto v: feature_values[i]) std::cout<< v <<" ";
			std::cout<<"-> "<<response_values[i]<<std::endl;
		}
	}
};


}}//namespace rfr
#endif
