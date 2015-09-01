#ifndef RFR_MOSTLY_CONTINUOUS_DATA_CONTAINER_HPP
#define RFR_MOSTLY_CONTINUOUS_DATA_CONTAINER_HPP


#include<vector>
#include<map>


#include "data_container_base.hpp"
#include "data_container_utils.hpp"


namespace rfr{

/** \brief A data container for mostly continuous data.
 *
 *  It might happen that only a small fraction of all features is categorical.
 *  In that case it would be wasteful to store the type of every feature separately.
 *  Instead, this data_container only stores the non-continuous ones in a hash-map.
 */
template<typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class mostly_contiuous_data : public rfr::data_container_base<num_type, response_type, index_type>{
  private:
	std::vector< std::vector<num_type> > feature_values;//!< 2d vector to store the feature values
	std::vector<num_type> response_values;              //!< the associated responses
	std::map<index_type, index_type> categorical_ranges;//!< a map storing the few categorical indices and their range
  public:
	virtual num_type feature  (index_type feature_index, index_type sample_index) const {
		return(feature_values.at(feature_index).at(sample_index));
	}

	virtual response_type response (index_type sample_index) const{
		return(response_values[sample_index]);
	}

	virtual bool add_data_point (num_type* feats, index_type num_elements, response_type &response){
		if (num_features() != num_elements) return(false);

		for (size_t i=0; i<num_elements; i++)
			feature_values[i].push_back(feats[i]);
		response_values.push_back(response);
		return(true);
	}

	virtual std::vector<num_type> retrieve_data_point (index_type index){
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


	virtual bool set_type_of_feature(index_type index, index_type type){
		if (index >= num_features()) return(false);
		if (type < 0)   return(false);
		// here, only store the categorical values
		if (type > 0)   categorical_ranges[index] = type;
			return(true);
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

};


}//namespace rfr
#endif
