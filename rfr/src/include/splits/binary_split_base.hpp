#ifndef RFR_BINARY_SPLIT_BASE_HPP
#define RFR_BINARY_SPLIT_BASE_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "boost/variant.hpp"

#include "data_containers/data_container_base.hpp"



namespace rfr{

template <typename num_type = float, typename index_type = unsigned int>
class binary_split{
  public:
	/** \brief the constructor for a binary split 
	 *
	 * \param data the container holding the training data
	 * \param features_to_try a vector with the indices of all the features that can be considered for this split
	 * \param indices a vector containing the subset of data point indices to be considered (output!)
	 * \param an iterator into this vector that says where to split the data for the two children
	 * 
	 */
	virtual split(	const data_container_base<num_type, index_type> &data,
					const std::vector<index_type> &features_to_try,
					std::vector<index_type> & indices,
					std::vector<index_type>::iterator &split_indices_it) = 0;
	/** \brief this operator tells into which child the given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return bool whether the feature_vector falls into the left (true) or right (false) child
	 */
	virtual bool operator(num_type *feature_vector) = 0;
};



}//namespace rfr
#endif
