#ifndef RFR_FANOVA_FOREST_HPP
#define RFR_FANOVA_FOREST_HPP

#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <tuple>
#include <random>
#include <algorithm>
#include <functional>
#include <memory>


#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>

#include <iostream>
#include <sstream>



#include "rfr/trees/tree_options.hpp"
#include "rfr/forests/forest_options.hpp"

#include "rfr/trees/binary_fanova_tree.hpp"

#include "rfr/util.hpp"

namespace rfr{ namespace forests{

template <typename node_t, typename num_t = float, typename response_t = float, typename index_t = unsigned int,  typename rng_t=std::default_random_engine>
class fANOVA_forest: public	rfr::forests::regression_forest< rfr::trees:binary_fANOVA_tree<node_t, num_t, response_t, index_t, rng_t>, num_t, response_t, index_t, rng_t> {
  private:

	typedef rfr::forests::regression_forest<rfr::trees:binary_fANOVA_tree<node_t, num_t, response_t, index_t, rng_t> , num_t, response_t, index_t, rng_t> super;

  protected:
	// to compute 'improvement over default' and such...
	num_t lower_cutoff = -std::numeric_limits<num_t>::infinity();
	num_t upper_cutoff = std::numeric_limits<num_t>::infinity();

  public:

	fANOVA_forest() : super() {}
	fANOVA_forest (forest_options<num_t, response_t, index_t> forest_opts): super(forest_opts) {};

  	/** \brief serialize function for saving forests with cerial*/
  	template<class Archive>
	void serialize(Archive & archive)
	{
		super::archive( archive);
	}

	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_t &rng){
		// fit the forest normaly
		super::fit(data, rng);

		// compute all the other stuff specific to the fANOVA here
		
	}

	/* \brief sets the cutoff to perform fANOVA on subspaces with bounded predictions
	 *
	 * This function is used for the fANOVA with a uniform prior on the subspace
	 * lower_cutoff <= y <= upper_cutoff as outlined in
	 * "Generalized Functional ANOVA Diagnostics for High Dimensional Functions
	 * of Dependent Variables" by Hooker.
	 */
	void set_cutoffs (num_t lower, num_t upper){
		lower_cutoff = lower;
		upper_cutoff= upper;

		prepare_trees_for_marginals();
	}

	/* \brief to read out the used cutoffs */
	std::pair<num_t, num_t> get_cutoffs(){ return(std::pair<num_t, num_t> (lower_cutoff, upper_cutoff));}


	/* \brief just calls the precompute marginals function of every tree */
	void prepare_trees_for_marginal(){
		for (auto &t: super::the_trees)
			t.precompute_marginals(lower_cutoff, upper_cutoff);
	}


	/* \brief returns the marginal prediction when some variables are not specified (NANs)
	 *
	 * this function implements equation 1 of
	 * "An efficient Approach for Assessing Hyperparameter Importance"
	 * by Hutter et al.
	 */

	 /*
	num_t marginal_mean_prediction( std::<num_t> feature_vector){
		if (std::isnan(lower_cutoff){
			lower_cutoff = -std::numeric_limits<num_t>::infinity()
			upper_cutoff = std::numeric_limits<num_t>::infinity()
			prepare_trees_for_marginal();
		}
		return(0);
	}
	*/


	/* \brief aggregates all used split values for all features in each tree
	 *
	 * TODO: move to fANOVA forest
	 */
	std::vector<std::vector<std::vector<num_t> > > all_split_values(const std::vector<index_t> &types){
		std::vector<std::vector<std::vector<num_t> > > rv;
		rv.reserve(super::the_trees.size());
			
		for (auto &t: the_trees)
			rv.emplace_back(t.all_split_values(types));
		return(rv);
	}


};


}}//namespace rfr::forests
#endif
