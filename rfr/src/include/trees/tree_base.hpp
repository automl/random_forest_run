#ifndef RFR_TREE_BASE_HPP
#define RFR_TREE_BASE_HPP

#include "data_containers/data_container_base.hpp"



namespace rfr{

template <typename num_type = float, typename index_type = unsigned int>
class tree_base{
  public:
	/** \brief member function to fit the tree to the data
	 *
	 * The interface is very general, and allows for deterministic and randomized decision tree at this point.
	 * For a random forest, some randomness has to be introduced, and the number of features
	 * considered for every step has to be set to be less than actual the number of features
	 * 
	 * \param data the container holding the training data
	 * \param tree_opts a tree_options opject that controls certain aspects of "growing" the tree
	 */
	virtual void fit(const rfr::data_container_base<num_type, index_type> &data,
			 rfr::tree_options<num_type, index_type> tree_opts) = 0;

	/** \brief member function to predict the response value for a single feature vector
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return num_type the prediction of the response value (usually the mean of all responses in the corresponding leaf)
	 */
	virtual num_type predict (num_type *feature_vector) = 0;
	
	
	/** \brief member function to predict the response values for a batch of  feature vectors stored in a data container
	 * 
	 * \param data a filled data container. For the prediction the (possibly empty) response values are ignored.
	 * 
	 * \return std::vector<num_type> the predictions for all points in a vector.
	 */	
	//virtual std::vector<num_type> predict (const rfr::data_container_base<num_type, index_type> &data) = 0;
	
	
	
	virtual index_type number_of_nodes() = 0;
	virtual index_type number_of_leafs() = 0;
	virtual index_type depth() = 0;
	
};



}//namespace rfr
#endif
