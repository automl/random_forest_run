#ifndef RFR_TREE_BASE_HPP
#define RFR_TREE_BASE_HPP

#include <vector>

#include "rfr/trees/tree_options.hpp"
#include "rfr/data_containers/data_container.hpp"



namespace rfr{ namespace trees{

template <typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_type=std::default_random_engine>
class tree_base{
  public:

	virtual ~tree_base() {}
  
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
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data,
			 rfr::trees::tree_options<num_t, response_t, index_t> tree_opts,
			 rng_type &rng){

		std::vector<num_t> sample_weights(data.num_data_points(), 1.);
		fit(data, tree_opts, sample_weights,rng);
	}

	/** \brief fits a (possibly randomized) decision tree to a subset of the data
	 * 
	 * At each node, if it is 'splitworthy', a random subset of all features is considered for the
	 * split. Depending on the split_type provided, greedy or randomized choices can be
	 * made. Just make sure the max_features in tree_opts to a number smaller than the number of features!
	 * 
	 * \param data the container holding the training data
	 * \param tree_opts a tree_options opject that controls certain aspects of "growing" the tree
	 * \param sample_weights vector containing the weights of all datapoints, can be used for subsampling (no checks are done here!)
	 * \param rng a (pseudo) random number generator
	 */
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data,
			 rfr::trees::tree_options<num_t, response_t, index_t> tree_opts,
			 const std::vector<num_t> &sample_weights,
			 rng_type &rng) = 0;



	/** \brief predicts the response value for a single feature vector
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return num_t the prediction of the response value (usually the mean of all responses in the corresponding leaf)
	 */
	virtual response_t predict (const std::vector<num_t> &feature_vector) const = 0;
	
	
	
	/** \brief returns all response values in the leaf into which the given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return std::vector<response_t> all response values in that leaf
	 */
	virtual std::vector<response_t> const &leaf_entries (const std::vector<num_t> &feature_vector) const = 0;
	
	
	
	
	virtual index_t number_of_nodes() const = 0;
	virtual index_t number_of_leafs() const = 0;
	virtual index_t depth() const = 0;
	
	/** \brief creates a LaTeX document visualizing the tree*/
	virtual void save_latex_representation(const char* filename) const = 0;
	
};



}}//namespace rfr::trees
#endif
