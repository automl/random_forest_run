#ifndef RFR_MOSTLY_CONTINUOUS_DATA_CONTAINER_HPP
#define RFR_MOSTLY_CONTINUOUS_DATA_CONTAINER_HPP


#include<vector>
#include<string>
#include<map>


#include "rfr/data_containers/data_container.hpp"
#include "rfr/data_containers/data_container_utils.hpp"


namespace rfr{ namespace data_containers{

/** \brief A data container for mostly continuous data.
 *
 *  It might happen that only a small fraction of all features is categorical.
 *  In that case it would be wasteful to store the type of every feature separately.
 *  Instead, this data_container only stores the non-continuous ones in a hash-map.
 */
template<typename num_t = float, typename response_t = float, typename index_t = unsigned int>
class mostly_continuous_data : public rfr::data_containers::base<num_t, response_t, index_t>{
  protected:
	std::vector< std::vector<num_t> > feature_values;//!< 2d vector to store the feature values
	std::vector<response_t> response_values;              //!< the associated responses
	std::vector<num_t> weights;            				 //!< the associated weights
	std::map<index_t, index_t> categorical_ranges;//!< a map storing the few categorical indices and their range
	response_t response_type;
  public:

	// empty constructor. Use this only if you read the data from a file!
	// the private vectors are not properly initialized! Adding data
	// points via 'add_data_point' may or may not fail!
	mostly_continuous_data() {}

	// if you plan on filling the container with single data points one at a time
	// use this constructor to specify the number of features 
	mostly_continuous_data (index_t num_f): feature_values(num_f, std::vector<num_t>(0)), response_type(0){}
  
	virtual num_t feature  (index_t feature_index, index_t sample_index) const {
		//return(feature_values.at(feature_index).at(sample_index));
		return(feature_values[feature_index][sample_index]);
	}


	virtual std::vector<num_t> features (index_t feature_index, std::vector<index_t> &sample_indices) const {
		std::vector<num_t> rv;
		rv.reserve(sample_indices.size());
		for (auto i : sample_indices)
			rv.push_back(feature_values[feature_index][i]);
		return(rv);
	}


	virtual response_t response (index_t sample_index) const{
		return(response_values[sample_index]);
	}

	virtual void add_data_point (std::vector<num_t> features, response_t response, num_t weight = 1){

		if (weight <= 0)
			throw std::runtime_error("Weight of a datapoint has to be positive.");

		if (num_features() != features.size())
			throw std::runtime_error("Number of elements does not match.");

		for (size_t i=0; i<features.size(); i++){
			if (get_type_of_feature(i) > 0){
				if (features[i] >= get_type_of_feature(i))
					throw std::runtime_error("Feature values not consistent with provided type. Data contains a value larger than allowed.");
				if (features[i]< 0)
					throw std::runtime_error("Feature values contain a negative value, can't make that a categorical feature.");
			}
		}

		for (size_t i=0; i<features.size(); i++)
				feature_values[i].push_back(features[i]);

		response_values.push_back(response);
		weights.push_back(weight);
	}

	virtual std::vector<num_t> retrieve_data_point (index_t index) const {
		std::vector<num_t> vec(feature_values.size());
		for (index_t i = 0; i < num_features(); i++)
			vec[i] = feature_values[i].at(index);
		return(vec);
	}
	virtual num_t weight(index_t sample_index) const{ return(weights[sample_index]);}

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
	virtual index_t get_type_of_feature (index_t feature_index) const{
		auto it = categorical_ranges.find(feature_index);
		if ( it == categorical_ranges.end())
			return(0);
		return(it->second);
	}


	virtual void set_type_of_feature(index_t index, index_t type){
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

	virtual index_t num_features() const {return(feature_values.size());}
	virtual index_t num_data_points() const {return(response_values.size());}


	// some helper functions
	int import_csv_files (const std::string &feature_file, const std::string &response_file, std::string weight_file=""){
		feature_values =  rfr::read_csv_file<num_t>(feature_file);
        response_values = (read_csv_file<response_t>(response_file))[0];
        if (weight_file.size()>0)
            weights = (read_csv_file<num_t>(weight_file))[0];
        else
            weights = std::vector<num_t> (response_values.size(), 1);
		return(feature_values.size());
	}

	bool check_consistency(){

		/* number of stored values has to be the same for:
		*  1. the response values
		*  2. every (row) vector in feature_values
		*/

		for (auto it = feature_values.begin(); it != feature_values.end(); it ++){
			if (num_data_points() != it->size())
				return(false);
		}
		
		if (weights.size() != num_data_points()) return(false);
		
		return(true);
	}

	virtual index_t get_type_of_response () const{return(response_type);}

	virtual void set_type_of_response (index_t resp_t){
		if (resp_t > 0){
			for (auto &rv: response_values){
				if (!(rv < resp_t))
					throw std::runtime_error("Response value not consistent with provided type. Data contains a value larger than allowed.");
				if (rv < 0)
					throw std::runtime_error("Response values contain a negative value, can't make that a categorical value.");
			}
			response_type = resp_t;
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
