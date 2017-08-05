#ifndef RFR_DEFAULT_CONTAINER_WITH_INSTANCES_HPP
#define RFR_DEFAULT_CONTAINER_WITH_INSTANCES_HPP


#include <vector>
#include <map>
#include <cmath>

#include "rfr/data_containers/data_container.hpp"
#include "rfr/data_containers/data_container_utils.hpp"


namespace rfr{ namespace data_containers{

/** \brief A data container for mostly continuous data with instances.
 *
 * Similar to the mostly_continuous_data container, but with the capability
 * to handle instance features.
 */
template<typename num_t = float, typename response_t = float, typename index_t = unsigned int>
class default_container_with_instances : public rfr::data_containers::base<num_t, response_t, index_t>{
  protected:

	std::vector< std::vector<num_t> > configurations;//!< 2d vector to store the feature values of all configurations
	std::vector< std::vector<num_t> > instances; 	//!< 2d vector to store the feature values of all instances
	
	std::vector<std::pair<index_t, index_t> > config_instance_pairs;
	std::vector<num_t> response_values;
	std::vector<num_t> weights;
	std::map<index_t, index_t> categorical_ranges;//!< a map storing the few categorical indices and their range
	index_t response_type;
  public:

	// empty constructor. Use this only if you read the data from a file!
	// the private vectors are not properly initialized! Adding data
	// points via 'add_data_point' may or may not fail!
	default_container_with_instances() { throw std::runtime_error("The empty constructor is not supported by this container.");}

	// if you plan on filling the container with single data points one at a time
	// use this constructor to specify the number of features for configurations and instances
	default_container_with_instances (index_t num_config_f, index_t num_instance_f):
		configurations(num_config_f, std::vector<num_t>(0)),
		instances(num_instance_f, std::vector<num_t>(0)),
		response_values(), weights(), response_type(0){}

	virtual num_t feature  (index_t feature_index, index_t sample_index) const {
		// find out if this is a config feature
		if (feature_index < configurations.size()){
			index_t i = config_instance_pairs[sample_index].first;
			return(configurations[feature_index][i]);
		}

		// otherwise it should be a instance feature
		index_t i = config_instance_pairs[sample_index].second;
		feature_index -= configurations.size();
		return(instances[feature_index][i]);
	}

	virtual std::vector<num_t> features (index_t feature_index, const std::vector<index_t> &sample_indices) const {
		std::vector<num_t> rv;
		rv.reserve(sample_indices.size());

		if (feature_index < configurations.size()){
			for (auto i : sample_indices)
				rv.push_back( configurations[feature_index][config_instance_pairs[i].first]);
		}
		else{
			feature_index -= configurations.size();
			for (auto i : sample_indices)
				rv.push_back(instances[feature_index][config_instance_pairs[i].second]);
		}
		return(rv);
	}

	virtual response_t response (index_t sample_index) const{ return(response_values[sample_index]); }

	virtual void add_data_point (std::vector<num_t>, response_t, num_t){
		throw std::runtime_error("This container does not support adding a data point with this function");
	}

	void add_data_point( index_t config_index, index_t instance_index, response_t r, num_t weight = 1){
		if (config_index >= num_configurations() )
			throw std::runtime_error("Configuration index too large.");
		if (instance_index >= num_instances() )
			throw std::runtime_error("Instance index too large.");
		config_instance_pairs.emplace_back(std::pair<index_t, index_t> (config_index, instance_index));
		response_values.emplace_back(r);
		weights.emplace_back(weight);
	}

	virtual num_t weight(index_t sample_index) const{ return(weights[sample_index]);}

	index_t num_configurations(){ 	return(configurations[0].size());	}

	index_t num_instances(){		return(instances[0].size());}

	index_t add_configuration(const std::vector<num_t> &config_features){
		if (config_features.size() != configurations.size())
			throw std::runtime_error("Number of configuration features is not what it should be!");

		for (auto i = 0u; i< config_features.size(); i++)
			configurations[i].push_back(config_features[i]);
		return(num_configurations()-1);
	}

	index_t add_instance(const std::vector<num_t> instance_features){
		if (instance_features.size() != instances.size())
			throw std::runtime_error("Number of instance features is not what it should be!");
		for (auto i = 0u; i< instance_features.size(); i++)
			instances[i].push_back(instance_features[i]);
		return(num_instances()-1);
	}

	virtual std::vector<num_t> retrieve_data_point (index_t index) const {
		std::vector<num_t> vec;
		vec.reserve(num_features());

		for (auto i = 0u; i< num_features(); i++)
			vec.emplace_back(feature(i, index));

		return(vec);
	}

	virtual index_t get_type_of_feature (index_t feature_index) const{
		auto it = categorical_ranges.find(feature_index);
		if ( it == categorical_ranges.end())
			return(0);
		return(it->second);
	}

	void set_type_of_configuration_feature(index_t index, index_t type){
		if (type > 0){
			// check consistency for categorical features
			for (auto &fv: configurations[index]){
				if (!(fv<type))
					throw std::runtime_error("Feature values not consistent with provided type. Data contains a value larger than allowed.");
				if (fv < 0)
					throw std::runtime_error("Feature values contain a negative value, can't make that a categorical feature.");
				// round it properly
				fv = std::round(fv);
			}
			categorical_ranges[index] = type;
		}
		else{
			categorical_ranges.erase(index);
		}
	}

	void set_type_of_instance_feature(index_t index, index_t type){
		if (type > 0){
			// check consistency for categorical features
			for (auto &fv: instances[index]){
				if (!(fv<type))
					throw std::runtime_error("Feature values not consistent with provided type. Data contains a value larger than allowed.");
				if (fv < 0)
					throw std::runtime_error("Feature values contain a negative value, can't make that a categorical feature.");
				// round it properly
				fv = std::round(fv);
			}
			categorical_ranges[index + configurations.size()] = type;
		}
		else{
			categorical_ranges.erase(index + configurations.size());
		}
	}

	virtual void set_type_of_feature(index_t index, index_t type){
		throw std::runtime_error("This container does not support setting the feature type with this function. Use set_type_of_configuration_feature or set_type_of_instance_feature instead.");
	}

	virtual index_t num_features() const {return(configurations.size() + instances.size());}

	virtual index_t num_data_points() const {return(config_instance_pairs.size());}

	void check_consistency(){

		/* number of stored values has to be the same for:
		*  1. the response values
		*  2. every (row) vector in feature_values
		*/

		if (config_instance_pairs.size() != num_data_points())
			throw std::runtime_error("config_instance_pairs has the wrong size!");

		if (response_values.size() != num_data_points())
			throw std::runtime_error("response_values has the wrong size!");


		for (auto p: config_instance_pairs){
			if ( (p.first <0 ) || (p.first >= num_configurations()))
				throw std::runtime_error("Invalid configuration index");
			if ( (p.second <0 ) || (p.second >= num_instances()))
				throw std::runtime_error("Invalid instance index");
		}

		// check all values for all features
		for (auto f = 0u; f<num_features(); f++){
			if (get_type_of_feature(f) == 0){
				for (auto n = 0u; n < num_data_points(); n++){
					if (std::isnan(feature(f,n)))
						throw std::runtime_error("Features contain a NaN!");
				}
			}
			else{
				index_t t = get_type_of_feature(f);
				for (auto n = 0u; n < num_data_points(); n++){
					if (std::isnan(feature(f,n)))
						throw std::runtime_error("Features contain a NaN!");

					if ((feature(f,n) <0) || (feature(f,n) >= t))
						throw std::runtime_error("A categorical feature has an invalid value!");
				}
			}
		}
		
		index_t t = get_type_of_response();
		for (auto r: response_values){
			if (std::isnan(r))
				throw std::runtime_error("Responses contain a NaN!");
		}
		
	}

	virtual index_t get_type_of_response () const{ return(response_type);}

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

	virtual void set_bounds_of_feature(index_t feature_index, num_t min, num_t max){
		throw std::runtime_error("Function not supported by this data container! Use set_bounds_of_configuration_feature or set_bounds_of_instance_feature instead!");
	}

	 virtual std::pair<num_t,num_t> get_bounds_of_feature(index_t feature_index) const{
		std::pair<num_t,num_t> r(NAN, NAN);
		return(r);
	 }

	
	
	/** \brief method to get instance as set_feature for predict_mean_var_of_mean_response_on_set method in regression forest
	 */
	virtual std::vector<num_t> get_instance_set(){
		std::vector<num_t> set_feature;
		set_feature.reserve( num_instances() * num_features());
		for (auto instance_idx = 0u; instance_idx < num_instances(); ++instance_idx){
				for (auto i = 0u; i <  configurations.size(); ++i){
						set_feature.emplace_back(NAN);
				}
				for (auto i = 0u; i < instances.size(); ++i){
						set_feature.emplace_back(instances[i][instance_idx]);
				}       
		}
		return set_feature;
	}
	virtual std::vector<num_t> get_configuration_set(num_t configuration_index){
		std::vector<num_t> features;
		features.reserve(num_features());
		for (auto i = 0u; i < configurations.size(); ++i){
				features.emplace_back(configurations[i][configuration_index]);
		}
		for (auto i = 0u; i < instances.size(); ++i){
				features.emplace_back(NAN);
		}   
		return features;
	}
	virtual std::vector<num_t> get_features_by_configuration_and_instance(num_t configuration_index, num_t instance_index){
		std::vector<num_t> features;
		features.reserve(num_features());
		for (auto i = 0u; i < configurations.size(); ++i){
				features.emplace_back(configurations[i][configuration_index]);
		}
		for (auto i = 0u; i < instances.size(); ++i){
				features.emplace_back(instances[i][instance_index]);
		}   
		return features;
	}
};


}}//namespace rfr
#endif
