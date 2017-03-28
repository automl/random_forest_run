#ifndef RFR_FANOVA_TREE_HPP
#define RFR_FANOVA_TREE_HPP

#include <vector>
#include <deque>
//#include <stack>
#include <numeric>
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

	std::vector<rfr::util::weighted_running_statistics<num_t> > marginal_prediction_stats;
	std::vector<std::vector<bool>  > active_variables;	// note: vector<bool> uses bitwise operations, so it might be too slow
	std::vector<std::vector<num_t> > split_values;
	
	num_t lower_cutoff;
	num_t upper_cutoff;
	
	
  public:
  
	binary_fANOVA_tree(): super(), split_values(0), lower_cutoff(NAN), upper_cutoff(NAN) {}

	virtual ~binary_fANOVA_tree() {}
	
    /* serialize function for saving forests
     * TODO: actually implement and test
     * */
  	template<class Archive>
  	void serialize(Archive & archive){
		super::serialize(archive);
		
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

		// reset internal variables	
		split_values.clear();
		active_variables.clear();
		marginal_prediction_stats.clear();
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

	rfr::util::weighted_running_statistics<num_t> marginalized_prediction_stat(const std::vector<num_t> &feature_vector,  std::vector<std::vector<num_t> > pcs, std::vector<index_t> types) const{

		auto active_features = rfr::util::get_non_NAN_indices(feature_vector);
		
		// change pcs for inactive variables to recycle subspace cardinality
		for (auto i=0u; i< pcs.size(); ++i){
			if (std::find(active_features.begin(), active_features.end(), i) != active_features.end())
				continue;
			if (types[i] == 0)
				pcs[i] = {0,1}; // interval of size 1
			else
				pcs[i] = {0};	// exactly one categorical value
		}

		rfr::util::weighted_running_statistics<num_t> stat;
		
		std::deque< std::vector<std::vector<num_t> > > pcss;
		pcss.push_back(pcs);

		std::deque<index_t> active_nodes;
		active_nodes.push_back(0);

		while (active_nodes.size() > 0){
			index_t node_index = active_nodes.back();
			active_nodes.pop_back();

			auto current_pcs = pcss.back();
			pcss.pop_back();
			
			// four cases
			// 1. active node has no weight (predicts NAN) -> skip
			if (std::isnan(marginal_prediction_stats[node_index].mean()))	continue;

			// 2. node itself splits on an active variable -> add corresponding child to active nodes
			if (super::the_nodes[node_index].can_be_split(feature_vector)){
				active_nodes.push_back(super::the_nodes[node_index].falls_into_child(feature_vector));
				auto &s = super::the_nodes[node_index].get_split();
				// let the split compute the new pcs
				pcss.push_back(s.compute_subspaces(current_pcs)[s(feature_vector)]);
				continue;
			}

			// 3. node's subtree split on any active varialble further down
			if (rfr::util::any_true(active_variables[node_index], active_features)){
				for (auto &c: super::the_nodes[node_index].get_children()){
					active_nodes.push_back(c);
					pcss.push_back(current_pcs);
				}
				continue;
			}
			
			// 4. node's subtree does not split on any of the active variables
			// this includes leaves  -> add to statistics if within the cutoffs
			
			auto size_correction = rfr::util::subspace_cardinality(current_pcs, types);
			
			if (marginal_prediction_stats[node_index].mean() < lower_cutoff){
				rfr::util::weighted_running_statistics<num_t> mew;
				mew.push( lower_cutoff, marginal_prediction_stats[node_index].sum_of_weights()/size_correction);
				stat += mew;
				continue;
			}

			if (marginal_prediction_stats[node_index].mean() > upper_cutoff){
				rfr::util::weighted_running_statistics<num_t> mew;
				mew.push( upper_cutoff, marginal_prediction_stats[node_index].sum_of_weights()/size_correction);
				stat += mew;
				continue;
			}			
			
			// @ this point, the nodes statistic can just be added to the final statistic
			stat += marginal_prediction_stats[node_index].multiply_weights_by(1./size_correction);
			
		}
		return stat;
	}
	/* \brief precomputes the marginal prediction in each node based on the subspace sizes
	 *
	 * To compute the fANOVA faster, the tree can efficiently compute and cache the marginal
	 * prediction for the subtree of any node. Combined with storing which variables remain constant
	 * within it, this should reduce the computational overhead; at least for not too important variables. */
	void precompute_marginals(num_t l_cutoff, num_t u_cutoff,
		const std::vector<std::vector<num_t> >& pcs, const std::vector<index_t>& types){

		/*
		 * Starting from the leaves, the marginalized prediction can be computed recursively
		 * by averaging the prediction from the children w.r.t their subspace size.
		 * Note, the cutoffs should be used to exclude leaves with a prediction outside the
		 * bounds.
		 * During this step the active variables should also be stored, such that it can
		 * be checked if the subtree's prediction depends on any of the 'active' variables
		 */

		lower_cutoff = l_cutoff;
		upper_cutoff = u_cutoff;
		
		assert(pcs.size() == types.size());

		if (super::the_nodes.size() == 0){
			throw std::runtime_error("The tree has not been fitted, yet. Call fit first and then precompute_marginals!");
		}
		size_t num_features = types.size();

		if (marginal_prediction_stats.size() == 0){
			/* Compute the size of the subspace for each leaf.
			 * This is can be done in a top-down fashion.
			 * These values don't change during the lifetime of the tree, so this step only
			 * needs to be performed once.
			 */

			marginal_prediction_stats.resize(super::the_nodes.size());
			active_variables.resize(super::the_nodes.size());

			// unfortunately, the subspaces have to be stored on the downward pass
			// this could be done dynamically by only storing the elements still needed, but for 
			// simplicity, let's store all of them for now
			std::vector< std::vector< std::vector <num_t> > > subspaces;
			subspaces.resize(super::the_nodes.size());
			subspaces[0] = pcs;


			// simple down pass to fill the leaves
			for (index_t node_index = 0; node_index < super::the_nodes.size(); ++node_index) {
				auto & n = super::the_nodes[node_index];
				
				if (n.is_a_leaf())
					marginal_prediction_stats[node_index].push(n.leaf_statistic().mean(), rfr::util::subspace_cardinality(subspaces[node_index], types));
				else{
					auto subss = n.compute_subspaces(subspaces[node_index]);
					subspaces[n.get_child_index(0)] = subss[0];
					subspaces[n.get_child_index(1)] = subss[1];
				}
				// delete no longer needed subspaces right away
				subspaces[node_index].clear();
			}
		}

		// reinitialize the active variables
		active_variables = std::vector<std::vector<bool> >(super::the_nodes.size(), std::vector<bool>(num_features, false));


		// node_index needs to be an int so it can be smaller than 0
		for (int node_index = super::the_nodes.size() - 1; node_index >= 0; --node_index) {
			auto & the_node = super::the_nodes[node_index];
			
			if (the_node.is_a_leaf())	continue;

			marginal_prediction_stats[node_index] = rfr::util::weighted_running_statistics<num_t> ();

			for (index_t child_index : super::the_nodes[node_index].get_children()) {
				// only consider children with a 'legal' subtree 
				if (!std::isnan(marginal_prediction_stats[child_index].mean())){
					
					if (marginal_prediction_stats[child_index].mean() <= lower_cutoff){
						rfr::util::weighted_running_statistics<num_t> stat;
						stat.push(lower_cutoff, marginal_prediction_stats[child_index].sum_of_weights());
						marginal_prediction_stats[node_index] += stat;
						continue;
					}
				
					if (marginal_prediction_stats[child_index].mean() >= upper_cutoff){
						rfr::util::weighted_running_statistics<num_t> stat;
						stat.push(upper_cutoff, marginal_prediction_stats[child_index].sum_of_weights());
						marginal_prediction_stats[node_index] += stat;
						continue;
					}

					marginal_prediction_stats[node_index] += marginal_prediction_stats[child_index];
					active_variables[node_index][the_node.get_split().get_feature_index()] = true;
				}
				
			}
			rfr::util::disjunction(active_variables[node_index], active_variables[the_node.parent()]);
		}
	}


	/* \brief finds all the split points for each dimension of the input space
	 * 
	 * This function only makes sense for axis aligned splits!
	 * 
	 * One could potentially make the split points cutoff dependent to remove unnecessary intervals
	 * 
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
				if (types[fi] == 0) {
					split_values[fi].emplace_back(s.get_num_split_value());
				}
			}
			
			for (auto &v: split_values)
				std::sort(v.begin(), v.end());
		}
		return(split_values);
	}


	// below are functions mainly for testing as they expose the internal variables
	num_t get_mean() const { return marginal_prediction_stats[0].mean();}
	
	num_t get_total_variance() const {return marginal_prediction_stats[0].variance_population();}
	
	
	num_t get_subspace_size(index_t node_index) const {
		return marginal_prediction_stats[node_index].sum_of_weights();
	}

	num_t get_marginal_prediction(index_t node_index) const {
		return marginal_prediction_stats[node_index].mean();
	}
 
 
	rfr::util::weighted_running_statistics<num_t> get_marginal_prediction_stat(index_t node_index) const {
		return marginal_prediction_stats[node_index];
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
