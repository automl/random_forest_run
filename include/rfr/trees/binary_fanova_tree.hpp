#ifndef RFR_FANOVA_TREE_HPP
#define RFR_FANOVA_TREE_HPP

#include <vector>
#include <deque>
#include <stack>
#include <utility>       // std::pair
#include <algorithm>     // std::shuffle
#include <numeric>       // std::iota
#include <cmath>         // std::abs
#include <iterator>      // std::advance
#include <fstream>
#include <random>


#include "cereal/cereal.hpp"
#include <cereal/types/bitset.hpp>
#include <cereal/types/vector.hpp>

#include "rfr/data_containers/data_container.hpp"
#include "rfr/nodes/temporary_node.hpp"
#include "rfr/nodes/k_ary_node.hpp"
#include "rfr/trees/tree_base.hpp"
#include "rfr/trees/tree_options.hpp"
#include "rfr/util.hpp"

#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"


namespace rfr{ namespace trees{

template <typename split_t, typename num_t = float, typename response_t = float, typename index_t = unsigned int, typename rng_t = std::default_random_engine>
class binary_fANOVA_tree : public k_ary_random_tree<2,  rfr::nodes::k_ary_node_full<2, split_t, num_t, response_t, index_t, rng_t> , num_t, response_t, index_t, rng_t> {

  private:
	typedef rfr::trees::k_ary_random_tree<2, rfr::nodes::k_ary_node_full<2, split_t, num_t, response_t, index_t, rng_t>, num_t, response_t, index_t, rng_t> super;
  protected:


	std::vector<num_t> subspace_sizes;					// size of the subspace in node's subtree -> for leaves it's the actual size but for the internal nodes it's the 'active size considering the cutoffs
	std::vector<num_t> marginal_prediction;				// prediction of the subtree below a node
	std::vector<std::vector<bool>  > active_variables;	// note: vector<bool> uses bitwise operations, so it might be too slow

	std::vector<std::vector<num_t> > split_values;
	
	num_t mean;
	num_t total_variance;
	
	
  public:
  
	binary_fANOVA_tree(): super(), split_values(0) {}

	virtual ~binary_fANOVA_tree() {}
	
    /* serialize function for saving forests
     * TODO: actually implement and test
     * */
  	template<class Archive>
  	void serialize(Archive & archive){
		super::serialize();
		
	}


	/* \brief fit the fANOVA tree
	 *
	 * Overloads the ancestor's method to reinitialize variables after fitting.
	 */
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data,
			 rfr::trees::tree_options<num_t, response_t, index_t> tree_opts,
			 const std::vector<num_t> &sample_weights,
			 rng_t &rng){
				 
		super::fit(data, tree_opts, sample_weights, rng);

			subspace_sizes.clear();
			active_variables.clear();
			marginal_prediction.clear();
			
			mean = NAN;
			total_variance = NAN;
	}

	/* \brief function to precompute the marginalized predictions
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
     * \returns the mean prediction marginalized over the desired inputs, NAN if the cutoffs exclude all potential leaves the feature vector would fall in
	 * */

	num_t marginalized_mean_prediction(const std::vector<num_t> &feature_vector) const{

		auto active_features = rfr::util::get_non_NAN_indices(feature_vector);
		std::deque<index_t> active_nodes;
		active_nodes.push_back(0);

		// to average the predictions of the individual leafes/nodes
		rfr::util::weighted_running_statistics<num_t> stats;

		while (active_nodes.size() > 0){
			index_t node_index = active_nodes.back();
			active_nodes.pop_back();
			
			// four cases
			// 1. active node has no weight (predicts NAN) -> skip
			if (std::isnan(marginal_prediction[node_index]))	continue;

			// 2. node itself splits on an active variable -> add corresponding child to active nodes
			if (super::the_nodes[node_index].can_be_split(feature_vector)){
				active_nodes.push_back(super::the_nodes[node_index].falls_into_child(feature_vector));
				continue;
			}

			// 3. node's subtree split on any active varialble and 
			if (rfr::util::any_true(active_variables[node_index], active_features)){
				for (auto &c: super::the_nodes[node_index].get_children()){
					active_nodes.push_back(c);
				}
				continue;
			}
			
			// 4. node's subtree does not split on any of the active variables (this includes leaves) -> add to statistics
			stats.push( marginal_prediction[node_index], subspace_sizes[node_index]);
		}
		return stats.mean();
	}
	/* \brief precomputes the marginal prediction in each node based on the subspace sizes
	 *
	 * To compute the fANOVA faster, the tree can efficiently compute and cache the marginal
	 * prediction for the subtree of any node. Combined with storing which variables remain constant
	 * within it, this should reduce the computational overhead; at least for not too important variables. */
	void precompute_marginals(num_t lower_cutoff, num_t upper_cutoff,
		const std::vector<std::vector<num_t> >& pcs, const std::vector<index_t>& types){

		/*
		 * Starting from the leaves, the marginalized prediction can be computed recursively
		 * by averaging the prediction from the children w.r.t their subspace size.
		 * Note, the cutoffs should be used to exclude leaves with a prediction outside the
		 * bounds.
		 * During this step the active variables should also be stored, such that it can
		 * be checked if the subtree's prediction depends on any of the 'active' variables
		 */
		assert(pcs.size() == types.size());

		if (super::the_nodes.size() == 0){
			throw std::runtime_error("The tree has not been fitted, yet. Call fit first and then precompute_marginals!");
		}
		size_t num_features = types.size();

		if (subspace_sizes.size() == 0){
			/* Compute the size of the subspace for each node and the active variables
			 * in the subtree below it. This is can be done in a top-down fashion.
			 * These values don't change during the lifetime of the tree, so this step only
			 * needs to be performed once.
			 */

			subspace_sizes.resize(super::the_nodes.size());
			active_variables.resize(super::the_nodes.size());
			marginal_prediction.resize(super::the_nodes.size());


			// unfortunately, the subspaces have to be stored on the downward pass
			// this could be done dynamically by only storing the elements still needed, but for 
			// simplicity, let's store all of them for now
			std::vector< std::vector< std::vector <num_t> > > subspaces;
			subspaces.resize(super::the_nodes.size());
			subspaces[0] = pcs;


			// simple down pass
			for (index_t node_index = 0; node_index < super::the_nodes.size(); ++node_index) {
				auto & n = super::the_nodes[node_index];
				subspace_sizes[node_index] = rfr::util::subspace_cardinality(subspaces[node_index], types);
				
				auto subss = n.compute_subspaces(subspaces[node_index]);
				subspaces[n.get_child_index(0)] = subss[0];
				subspaces[n.get_child_index(1)] = subss[1];
				
				// delete no longer needed subspaces right away
				subspaces[node_index].clear();
			}
		}

		// reinitialize the active variables
		active_variables = std::vector<std::vector<bool> >(super::the_nodes.size(), std::vector<bool>(num_features, false));


		rfr::util::weighted_running_statistics<double> total_stat;
		// node_index needs to be an int so it can be smaller than 0
		for (int node_index = super::the_nodes.size() - 1; node_index >= 0; --node_index) {
			auto & the_node = super::the_nodes[node_index];
			
			if (the_node.is_a_leaf()) {
				marginal_prediction[node_index] = the_node.leaf_statistic().mean(); // Get leaf's mean prediction
				// apply cutoffs
				if ((marginal_prediction[node_index] < lower_cutoff) || (marginal_prediction[node_index] > upper_cutoff)){
					marginal_prediction[node_index] = NAN;
				}
				else{
					total_stat.push(marginal_prediction[node_index], subspace_sizes[node_index]);
				}
			}
			else{

				rfr::util::weighted_running_statistics<num_t> stat;

				for (index_t child_index : super::the_nodes[node_index].get_children()) {
					if (!std::isnan(marginal_prediction[child_index])){
						stat.push(marginal_prediction[child_index], subspace_sizes[child_index]);
						active_variables[node_index][the_node.get_split().get_feature_index()] = true;
					}
				}
				rfr::util::disjunction(active_variables[node_index], active_variables[the_node.parent()]);
				subspace_sizes[node_index] = stat.sum_of_weights();
				marginal_prediction[node_index] = stat.mean();
			}
		}
		mean = total_stat.mean();
		total_variance = total_stat.variance_unbiased_importance();
	}


	/* \brief finds all the split points for each dimension of the input space
	 * 
	 * This function only makes sense for axis aligned splits!
	 * */
	std::vector<std::vector<num_t> > all_split_values (const std::vector<index_t> &types) {
		
		if (split_values.size() == 0){
			
			split_values.resize(types.size());
			
			for (auto &n: super::the_nodes){
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
		}
		return(split_values);
	}


	// below are functions mainly for testing as they expose the internal variables
	num_t get_mean() const { return(mean);}
	
	num_t get_total_variance() const {return(total_variance);}
	
	
	num_t get_subspace_size(index_t node_index) const {
		return subspace_sizes[node_index];
	}

	num_t get_marginal_prediction(index_t node_index) const {
		return marginal_prediction[node_index];
	}
  
	const std::vector<bool>& get_active_variables(index_t node_index) const {
		return active_variables[node_index];
	}

	const std::vector<rfr::nodes::k_ary_node_full<2, split_t, num_t, response_t, index_t, rng_t>>& get_nodes() const {
		return super::the_nodes;
	}

};

}}//namespace rfr::trees
#endif
