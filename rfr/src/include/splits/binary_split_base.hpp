#ifndef RFR_BINARY_SPLIT_BASE_HPP
#define RFR_BINARY_SPLIT_BASE_HPP

#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

#include "boost/variant.hpp"

#include "data_containers/data_container_base.hpp"



namespace rfr{

template <typename data_container_type, typename num_type = float, typename index_type = unsigned int>
class binary_split_base{
  public:
	/** \brief member function to find the optimal binary split for a subset of the data and features
	 *
	 * Defining the interface that every binary_split has to implement. Unfortunately, virtual constructors are
	 * not allowed in C++, so this function is called instead. Code in the nodes and the tree will only use the 
	 * default constructor and this method during construction.
	 * 
	 * \param data the container holding the training data
	 * \param features_to_try a vector with the indices of all the features that can be considered for this split
	 * \param indices a vector containing the subset of data point indices to be considered (output!)
	 * \param split_indices_it an iterator into indices specifying where to split the data for the two children
	 * 
	 * \return float the loss of the found split
	 */
	virtual num_type find_best_split(	const data_container_type &data,
									const std::vector<index_type> &features_to_try,
									std::vector<index_type> & indices,
									typename std::vector<index_type>::iterator &split_indices_it) = 0;

	/** \brief operator telling into which child the given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return bool whether the feature_vector falls into the left (true) or right (false) child
	 */
	virtual bool operator() (num_type *feature_vector) = 0;
};



}//namespace rfr
#endif
