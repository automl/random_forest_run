#ifndef RFR_TREE_BASE_HPP
#define RFR_TREE_BASE_HPP

#include <vector>

#include "rfr/trees/tree_options.hpp"
#include "rfr/data_containers/data_container_base.hpp"



namespace rfr{ namespace trees{

template <typename rng_type, typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class tree_base{
  public:
	/** \brief member function to fit the tree to the whole data
	 *
	 * The interface is very general, and allows for deterministic and randomized decision tree at this point.
	 * For a random forest, some randomness has to be introduced, and the number of features
	 * considered for every step has to be set to be less than actual the number. In its
	 * standard implementation this function just calls the second fit method with an indicex vector = [0, ..., num_data_points-1].
	 * 
	 * 
	 * \param data the container holding the training data
	 * \param tree_opts a tree_options opject that controls certain aspects of "growing" the tree
	 * \param rng a (pseudo) random number generator
	 */
	virtual void fit(const rfr::data_containers::data_container_base<num_type, response_type, index_type> &data,
			 rfr::trees::tree_options<num_type, response_type, index_type> tree_opts,
			 rng_type &rng){

		std::vector<index_type> data_indices(data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);
		fit(data, tree_opts, data_indices,rng);
	}

	/** \brief fits a (possibly randomized) decision tree to a subset of the data
	 * 
	 * At each node, if it is 'splitworthy', a random subset of all features is considered for the
	 * split. Depending on the split_type provided, greedy or randomized choices can be
	 * made. Just make sure the max_features in tree_opts to a number smaller than the number of features!
	 * 
	 * \param data the container holding the training data
	 * \param tree_opts a tree_options opject that controls certain aspects of "growing" the tree
	 * \param data_indices vector containing the indices of all allowed datapoints to be used (to implement subsampling, no checks are done here!)
	 */
	virtual void fit(const rfr::data_containers::data_container_base<num_type, response_type, index_type> &data,
			 rfr::trees::tree_options<num_type, response_type, index_type> tree_opts,
			 std::vector<index_type> &data_indices,
			 rng_type &rng) = 0;



	/** \brief predicts the response value for a single feature vector
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return num_type the prediction of the response value (usually the mean of all responses in the corresponding leaf)
	 */
	virtual response_type predict (num_type *feature_vector) = 0;
	
	
	
	/** \brief returns all response values in the leaf into which the given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return std::vector<response_type> all response values in that leaf
	 */
	virtual std::vector<response_type> const &leaf_entries (num_type *feature_vector) = 0;
	
	
	
	/** \brief member function to predict the response values for a batch of  feature vectors stored in a data container
	 * 
	 * \param data a filled data container. For the prediction the (possibly empty) response values are ignored.
	 * 
	 * \return std::vector<num_type> the predictions for all points in a vector.
	 */	
	//virtual std::vector<response_type> predict (const rfr::data_container_base<num_type, index_type> &data) = 0;
	
	
	
	virtual index_type number_of_nodes() = 0;
	virtual index_type number_of_leafs() = 0;
	virtual index_type depth() = 0;
	
	/** \brief creates a LaTeX document visualizing the tree*/
	virtual void save_latex_representation(const char* filename) = 0;
	
	
};



}}//namespace rfr::trees
#endif
