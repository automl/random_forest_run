#ifndef RFR_K_ARY_TREE_HPP
#define RFR_K_ARY_TREE_HPP

#include<vector>
#include<deque>
#include<stack>
#include<utility>       // std::pair
#include<algorithm>     // std::shuffle
#include<numeric>       // std::iota
#include<cmath>         // std::abs
#include<iterator>      // std::advance
#include<fstream>
#include<random>


#include "cereal/cereal.hpp"
#include <cereal/types/bitset.hpp>
#include <cereal/types/vector.hpp>

#include "rfr/data_containers/data_container.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_base.hpp"
#include "rfr/trees/tree_options.hpp"

#include "rfr/trees/k_ary_tree.hpp"


namespace rfr{ namespace trees{

template <typename node_type, typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_type = std::default_random_engine>
class binary_fanova_tree : public rfr::trees::k_ary_random_tree<2, node_type, num_t, response_t, index_t, rng_type> {

  private:
	typedef rfr::trees::k_ary_random_tree<2, node_type, num_t, response_t, index_t, rng_type> super;
  protected:


	std::vector<num_t> subspace_sizes;					// size of the subspace in node's subtree
	std::vector<num_t> marginal_prediction;				// prediction of the subtree below a node
	std::vector<std::vector<bool>  > active_variables;	// note: vector<bool> uses bitwise operations, so it might be too slow

	
	std::vector<std::vector<num_t> > all_split_values;
	
  public:
  
	binary_fanova_tree(): super() {}

	virtual ~binary_fanova_tree() {}
	
    /* serialize function for saving forests */
  	template<class Archive>
  	void serialize(Archive & archive){
		super.serialize();
		
	}


	/* \brief fit the fANOVA forest
	 *
	 * Overloads the ancestor's method to reinitialize variables after fitting.
	 */
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data,
			 rfr::trees::tree_options<num_t, response_t, index_t> tree_opts,
			 const std::vector<num_t> &sample_weights,
			 rng_type &rng){
	}

	/* \brief function to recursively compute the marginalized predictions
	 * 
	 * To compute the fANOVA, the mean prediction over partial assingments is needed.
	 * To accomplish that, feed this function a numerical vector where each element that
	 * is NAN will be marginalized over.
     * 
	 * At any split, this function either follows one path or averages the
	 * prediction of all children weighted by the subspace size. If the subtree
	 * does not split on any of the 'active' features, a pre-computed values is used.
     * 
     * \param feature_vector the features vector with NAN for dimensions over which is marginalized
     * 
     * \returns the mean prediction marginalized over the desired inputs
	 * */
	num_t marginalized_mean_prediction(const std::vector<num_t> &feature_vector) const{

	}

	/* \brief precomputes the marginal prediction in each node based on the subspace sizes
	 *
	 * To compute the fANOVA faster, the tree can efficiently compute and cache the marginal
	 * prediction for the subtree of any node. Combined with storing which variables remain constant
	 * within it, this should reducet the computational overhead; at least for not too important variables. */
	void precompute_marginals(num_t lower_cutoff, num_t upper_cutoff,
		const std::vector<std::vector<num_t> >& pcs, const std::vector<index_t>& types){

		/* This function should work in two steps:
		 * 		1.	Compute the size of the subspace for each node. See partition_recursor
		 * 			for an example. This is can be done in a top-down fashion.
		 * 		2. 	Starting from the leaves, the marginalized prediction can be computed recursively
		 * 			by averaging the prediction from the children w.r.t their subspace size.
		 * 			Note, the cutoffs should be used to exclude leaves with a prediction outside the
		 * 			bounds.
		 * 			During this step the active variables should also be stored, such that it can
		 * 			be checked if the subtree's prediction depends on any of the 'active' variables
		 */

		// FIXME: types is not yet used.
		assert(pcs.size() == types.size());

		size_t variables_size = the_nodes[0].get_split().get_num_categories();	//  Get_variables_size

		subspace_sizes.resize(the_nodes.size());
		active_variables.resize(the_nodes.size());
		marginal_prediction.resize(the_nodes.size());

		for (index_t node_index = 0; node_index < the_nodes.size(); ++node_index) {
			std::array<std::vector< std::vector<num_t> >, 2> subspaces = the_nodes[node_index].compute_subspaces(pcs[node_index]);
			subspace_sizes[node_index] = 1; // #TODO Check type
			for (const std::vector< std::vector<num_t> >& subspace : subspaces) {
				for (const std::vector<num_t>& feature : subspace) {
					num_t feature_size = 0;
					if (!std::isnan(the_nodes[node_index].get_split().get_num_split_value())) {
						// if feature is numerical
						feature_size = feature[1] - feature[0];
					} else {
						// feature is discrete values
						feature_size = feature.size();
					}
					subspace_sizes[node_index] *= feature_size;
				}
			}
			active_variables[node_index].resize(variables_size);
		}
		for (index_t node_index = the_nodes.size() - 1; node_index >= 0; --node_index) {
			std::vector<bool> active_vars = get_split_values(the_nodes[node_index]);  // TODO: Get split variables
			index_t parent = the_nodes[node_index].parent();

			assert(variables_size == active_vars.size());						// The total number of variables is always the same.
			for (index_t var = 0; var < active_variables[par].size(); ++var) {
				active_variables[par][var] |= active_vars[var];					// Take the active variables made by this split
				active_variables[par][var] |= active_variables[par][var];		// Take the active variables of the children
			}

			if (the_nodes[node_index].is_a_leaf()) {
				marginal_prediction[node_index] = the_nodes[node_index].leaf_statistic().mean(); // Get leaf's mean prediction
			} else {
				// TODO: Update the prediction formula, currently it's weighted sum over children
				marginal_prediction[node_index] = 0.0;
				for (size_t child_index : get_child_index(node_type)) {
					marginal_prediction[node_index] += marginal_prediction[child_index] * subspace_sizes[child_index];
				}
			}
		}

	}




	////////////////////////////////////////////////////////////////////
	// LEGACY CODE below, should be refactored/removed soon!
	////////////////////////////////////////////////////////////////////

	/* \brief finds all the split points for each dimension of the input space
	 * 
	 * This function only makes sense for axis aligned splits!
	 * */

	 /*
	std::vector<std::vector<num_t> > all_split_values (const std::vector<index_t> &types) const {
		std::vector<std::vector<num_t> > split_values(types.size());
		
		for (auto &n: the_nodes){
			if (n.is_a_leaf()) continue;
			
			const auto &s = n.get_split();
			auto fi = s.get_feature_index();

			// if a split on a categorical occurs, just add all its possible values
			if((types[fi] > 0) && (split_values[fi].size() == 0)){
				split_values[fi].resize(types[fi]);
				std::iota(split_values[fi].begin(), split_values[fi].end(), 0);
			}
			else{
				split_values[fi].emplace_back(s.get_num_split_value());
			}
		}
		
		for (auto &v: split_values)
			std::sort(v.begin(), v.end());
		return(split_values);
	}
	*/
	
	/* \brief Function to recursively compute the partition induced by the tree
	 *
	 * Do not call this function from the outside! Needs become private at some point!
	 */
	 /*
	void partition_recursor (	std::vector<std::vector< std::vector<num_t> > > &the_partition,
							std::vector<std::vector<num_t> > &subspace, num_t node_index) const {

		// add subspace for a leaf
		if (the_nodes[node_index].is_a_leaf())
			the_partition.push_back(subspace);
		else{
			// compute subspaces of children
			auto subs = the_nodes[node_index].compute_subspaces(subspace);
			// recursively go trough the tree
			for (auto i=0u; i<k; i++){
				partition_recursor(the_partition, subs[i], the_nodes[node_index].get_child_index(i));
			}
		}
	}
	*/


	/* \brief computes the partitioning of the input space induced by the tree */
	/*
	std::vector<std::vector< std::vector<num_t> > > partition( std::vector<std::vector<num_t> > pcs) const {
	
		std::vector<std::vector< std::vector<num_t> > > the_partition;
		the_partition.reserve(num_leafs);
		
		partition_recursor(the_partition, pcs, 0);
	
	return(the_partition);
	}

	*/


};

}}//namespace rfr::trees
#endif
