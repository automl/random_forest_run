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

template<typename num_type = float, typename index_type = unsigned int>
class mostly_contiuous_data : rfr::data_container_base<num_type, index_type>{
  private:
	std::vector< std::vector<num_type> > feature_values;//!< 2d vector to store the feature values
	std::vector<num_type> response_values;              //!< the associated responses
	std::map<index_type, index_type> categorical_ranges;//!< a map storing the few categorical indices and their range

  public:


    virtual num_type feature (int feature_index, int sample_index){
        return(feature_values[feature_index][sample_index]);
    }

    virtual num_type response (int sample_index){
        return(response_values[sample_index]);
    }

    virtual bool add_data_point (num_type* feats, index_type num_elements, num_type &response){
        if (num_features() != num_elements) return(false);

        for (size_t i=0; i<num_elements; i++)
            feature_values[i].push_back(feats[i]);
        response_values.push_back(response);
        return(true);
    }

    /** \brief method to query the type of a feature
     *
     * As most features are assumed to be categorical, it is actually
	 * beneficial to store only the categorical exceptions in a hash-map.
	 * Type = 0 means continuous, and Type = n >= 1 means categorical with
	 * options \in {1, n}. For consistency, we exclude zero from the categorical
	 * values if anyone wants to add sparse data later on.
	 *
     * \param feature_index the index of the feature
     * \return int type of the feature: 0 - numerical value (float or int); n>0 - categorical value with n different values {1,2,...,n}
     *
     */
    virtual index_type get_type_of_feature (index_type feature_index){
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


    virtual index_type num_features(){return(feature_values.size());}
	virtual index_type num_data_points(){return(feature_values[0].size());}


    // some helper functions (might be worth including some of them into the base class, as performance does not really matter here)
	int read_feature_file (const char* filename){
		feature_values = rfr::read_csv_file<num_type>(filename);
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


    /* Old stuff, need to lock through it before removing it
	// method get the responses of a subset of the samples represented by indices
	std::vector<num_type> responses_of(std::vector<int> indices){
		std::vector<num_type> return_vector(indices.size());
		for(auto i = 0; i < indices.size(); i++)
			return_vector[i] = response_values[indices[i]];
		return(return_vector);
	}
	// same as above only with an iterator range of unknown size
	std::vector<num_type> responses_of(const std::vector<int>::iterator start, const std::vector<int>::iterator finish){
		std::vector<num_type> return_vector;
		for(std::vector<int>::iterator it = start; it != finish; it++){
			return_vector.push_back(response_values[(*it)]);
		}
		return(return_vector);
	}

	// method to extract one feature value for all samples in indices
	std::vector<num_type> feature_of(int index, std::vector<int> indices){
		std::vector<num_type> return_vector(indices.size());
		for(auto i = 0; i < indices.size(); i++)
			return_vector[i] = feature_values[index][indices[i]];
		return(return_vector);
	}

	// Same as above, but with iterator range of unknown size
	std::vector<num_type> feature_of(int index, std::vector<int>::iterator start, std::vector<int>::iterator finish){
		std::vector<num_type> return_vector;
		for(auto it = start; it != finish; it++)
			return_vector.push_back( feature_values[index][*it]);
		return(return_vector);
	}
    */
};


}//namespace rfr
#endif
