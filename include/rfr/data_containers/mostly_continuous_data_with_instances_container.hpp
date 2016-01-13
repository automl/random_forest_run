#ifndef RFR_MOSTLY_CONTINUOUS_DATA_CONTAINER_WITH_INSTANCES_HPP
#define RFR_MOSTLY_CONTINUOUS_DATA_CONTAINER_WITH_INSTANCES_HPP


#include <vector>
#include <map>
#include <cmath>

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
	mostly_continuous_data_with_instances() { throw std::runtime_error("The empty constructor is not supported by this container.");}

	// if you plan on filling the container with single data points one at a time
	// use this constructor to specify the number of features for configurations and instances
	mostly_continuous_data_with_instances (index_type num_config_f, index_type num_instance_f):
		configurations(num_config_f, std::vector<num_type>(0)),
		instances(num_instance_f, std::vector<num_type>(0)),
		response_t(0){}

	virtual num_type feature  (index_type feature_index, index_type sample_index) const {
		// find out if this is a config feature
		if (feature_index < configurations.size()){
			index_type i = config_instance_pairs[sample_index].first;
			return(configurations[feature_index][i]);
		}

		// otherwise it should be a instance feature
		index_type i = config_instance_pairs[sample_index].second;
		feature_index -= configurations.size();
		return(instances[feature_index][i]);
	}

	virtual std::vector<num_type> features (index_type feature_index, std::vector<index_type> &sample_indices) const {
		std::vector<num_type> rv;
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

	virtual response_type response (index_type sample_index) const{
		return(response_values[sample_index]);
	}

	virtual void add_data_point (num_type* features, index_type num_elements, response_type response){
		throw std::runtime_error("This container does not support adding a data point with this function");
	}

	void add_data_point( index_type config_index, index_type instance_index, response_type r){
		if (config_index >= num_configurations() )
			throw std::runtime_error("Configuration index too large.");
		if (instance_index >= num_instances() )
			throw std::runtime_error("Instance index too large.");
		config_instance_pairs.emplace_back(std::pair<index_type, index_type> (config_index, instance_index));
		response_values.emplace_back(r);
	}

	index_type num_configurations(){
		if (configurations.size() > 0)
			return(configurations[0].size());
		return(0);
	}

	index_type num_instances(){
		if (instances.size() > 0)
			return(instances[0].size());
		return(0);
	}

	index_type add_configuration(num_type* config_features, index_type num_elements){
		if (num_elements != configurations.size())
			throw std::runtime_error("Number of configuration features is not what it should be!");

		for (auto i = 0u; i< num_elements; i++)
			configurations[i].push_back(config_features[i]);
		return(num_configurations()-1);
	}

	index_type add_instance(num_type* instance_features, index_type num_elements){
		if (num_elements != instances.size())
			throw std::runtime_error("Number of instance features is not what it should be!");
		for (auto i = 0u; i< num_elements; i++)
			instances[i].push_back(instance_features[i]);
		return(num_instances()-1);
	}

	virtual std::vector<num_type> retrieve_data_point (index_type index){
		std::vector<num_type> vec;
		vec.reserve(num_features());

		for (auto i = 0u; i< num_features(); i++)
			vec.emplace_back(feature(i, index));

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

	void set_type_of_configuration_feature(index_type index, index_type type){
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

	void set_type_of_instance_feature(index_type index, index_type type){
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

	virtual void set_type_of_feature(index_type index, index_type type){
		if (index >= num_features())
			throw std::runtime_error("Unknown index specified.");
		if (! (type >= 0))
			throw std::runtime_error("Type value should be >= 0");


		if (index < configurations.size())
			set_type_of_configuration_feature(index, type);

		else
			set_type_of_instance_feature(index-configurations.size(), type);

	}

	virtual index_type num_features() const {return(configurations.size() + instances.size());}

	virtual index_type num_data_points() const {return(config_instance_pairs.size());}

	void check_consistency(){

		/* number of stored values has to be the same for:
		*  1. the response values
		*  2. every (row) vector in feature_values
		*/

		if (config_instance_pairs.size() != num_data_points)
			throw std::runtime_error("config_instance_pairs has the wrong size!");

		if (response_values.size() != num_data_points)
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
					if (isnan(feature(f,n)))
						throw std::runtime_error("Features contain a NaN!");
				}
			}
			else{
				index_type t = get_type_of_feature(f);
				for (auto n = 0u; n < num_data_points(); n++){
					if (isnan(feature(f,n)))
						throw std::runtime_error("Features contain a NaN!");

					if ((feature(f,n) <0) || (feature(f,n) >= t))
						throw std::runtime_error("A categorical feature has an invalid value!");
				}
			}
		}
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
};


}}//namespace rfr
#endif
