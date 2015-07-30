#ifndef RFR_K_ARY_TREE_HPP
#define RFR_K_ARY_TREE_HPP

#include<vector>


#include "data_containers/data_container_base.hpp"
#include "nodes/k_ary_nodes.hpp"
#include "trees/tree_options.hpp"



namespace rfr{

template <const int k,typename split_type, typename RNG_type, typename num_type = float, typename index_type = unsigned int>
class k_ary_random_tree : tree_base<num_type, index_type>{
  private:
	static RNG_type rng;
	
	
  public:
	/** \brief fits a randomized decision tree to the data
	 * 
	 * At each node, if it is 'splitworthy', a random subset of all features is considered for the
	 * split. Depending on the split_type provided, greedy or randomized choices can be
	 * made. Just make sure the max_features in tree_opts to a number smaller than the number of features!
	 * 
	 * \param data the container holding the training data
	 * \param tree_opts a tree_options opject that controls certain aspects of "growing" the tree
	 */
	virtual void fit(const rfr::data_container_base<num_type, index_type> &data,
					 const rfr::tree_options<num_type, index_type> tree_opts){
						 
		
		
	}

	/** \brief member function to predict the response value for a single feature vector
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return num_type the prediction of the response value (usually the mean of all responses in the corresponding leaf)
	 */
	virtual num_type predict (num_type *feature_vector){
		
	}
	
	
	/** \brief member function to predict the response values for a batch of  feature vectors stored in a data container
	 * 
	 * \param data a filled data container. For the prediction the (possibly empty) response values are ignored.
	 * 
	 * \return std::vector<num_type> the predictions for all points in a vector.
	 */	
	virtual std::vector<num_type> predict (const rfr::data_container_base<num_type, index_type> &data){
		
	}
};



}//namespace rfr
#endif
