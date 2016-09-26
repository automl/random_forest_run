#ifndef RFR_SPLIT_BASE_HPP
#define RFR_SPLIT_BASE_HPP

#include <vector>
#include <array>
#include <string>
#include "rfr/data_containers/data_container_base.hpp"



namespace rfr{ namespace splits{



template <typename index_t = unsigned int, typename num_t = float, typename response_t = float>
struct data_info{
	index_t index;
	response_t response;
	num_t feature;
	num_t weight;
};




template <const int k,  typename rng_t, typename num_t = float, typename response_t = float, typename index_t = unsigned int>
class k_ary_split_base{
  public:
	/** \brief member function to find the optimal split for a subset of the data and features
	 *
	 * Defining the interface that every split has to implement. Unfortunately, virtual constructors are
	 * not allowed in C++, so this function is called instead. Code in the nodes and the tree will only use the 
	 * default constructor and the methods below for training and prediction.
	 * 
	 * \param data the container holding the training data
	 * \param features_to_try a vector with the indices of all the features that can be considered for this split
	 * \param indices a vector containing the subset of data point indices to be considered (output!)
	 * \param split_indices_it iterators into indices specifying where to split the data for the children. Number of iterators is k+1, for easier iteration
	 * \param rng (pseudo) random number generator as a source for stochasticity
	 * 
	 * \return float the loss of the found split
	 */
	virtual num_t find_best_split(const rfr::data_containers::data_container_base<num_t, response_t, index_t> &data,
									const std::vector<index_t> &features_to_try,
									std::vector<index_t> & indices,
									std::array<typename std::vector<index_t>::iterator, k+1> &split_indices_it,
									rng_t &rng) = 0;

	/** \brief tells into which child a given feature vector falls
	 * 
	 * \param feature_vector an array containing a valid (in terms of size and values!) feature vector
	 * 
	 * \return index_t index of the child into which this feature falls
	 */
	virtual index_t operator() (num_t *feature_vector) = 0;
	
	/** \brief some debug output that prints a informative representation to std::cout*/
	virtual void print_info() = 0;
	
	/** \brief hopefully all trees can create a LaTeX document as a visualization, this contributes the text of the split.*/
	virtual std::string latex_representation () = 0;
};



}}//namespace rfr::splits
#endif
