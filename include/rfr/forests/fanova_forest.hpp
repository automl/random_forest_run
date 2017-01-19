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
#include "rfr/util.hpp"

namespace rfr{ namespace forests{

typedef cereal::PortableBinaryInputArchive binary_iarch_t;
typedef cereal::PortableBinaryOutputArchive binary_oarch_t;

typedef cereal::JSONInputArchive ascii_iarch_t;
typedef cereal::JSONOutputArchive ascii_oarch_t;




template <typename tree_t, typename num_t = float, typename response_t = float, typename index_t = unsigned int,  typename rng_t=std::default_random_engine>
class fANOVA_forest: public rfr::forests::regression_forest<tree_t, num_t, response_t, index_t, rng_t> {
  private:

	typedef rfr::forests::regression_forest<tree_t, num_t, response_t, index_t, rng_t> super;

	
	std::vector<std::vector<num_t> > split_values_of_tree;

	// to compute 'improvement over default' and such...
	num_t lower_cutoff(-std::numeric_limits<num_t>::infinity());
	num_t upper_cutoff(std::numeric_limits<num_t>::infinity());

  public:

	fANOVA_forest() : super() {}
	fANOVA_forest (forest_options<num_t, response_t, index_t> forest_opts): super(forest_opts) {};

  	/** \brief serialize function for saving forests with cerial*/
  	template<class Archive>
	void serialize(Archive & archive)
	{
		super::archive( archive);
	}

	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_type &rng){
		// fit the forest normaly
		super::fit(data, rng);

		// compute all the split values for all variables of each tree
		split_values_of_tree.reserve(super::the_trees.size());
		for (auto &t: super::the_trees)
			split_values_of_tree.emplace_back(t.all_split_values(types));
		
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
		upper_cutpoff= upper;

		prepare_trees_for_marginals();
	}

	std::pair<num_t, num_t> get_cutoffs(){ return(std::pair<num_t, num_t> (lower_cutoff, upper_cutoff);}


	void prepare_trees_form_marginal(){

	}


	/* \brief returns the marginal prediction when some variables are not specified (NANs)
	 *
	 * this function implements equation 1 of
	 * "An efficient Approach for Assessing Hyperparameter Importance"
	 * by Hutter et al.
	 */
	num_t marginal_prediction( std::<num_t> feature_vector){
		
	}



	/* \brief predict the mean and the variance of the mean prediction across a set of partial features
	 * 
	 * A very special function to predict the mean response of a a partial assignment for a given set.
	 * It takes the prediction of set-mean of every individual tree and combines to estimate the mean its
	 * total variance. The predictions of two trees are considered uncorrelated
	 * 
	 * \param features a (partial) configuration where unset values should be set to NaN
	 * \param set_features a 1d-array containing the (partial) assignments used for the averaging. Every NaN value will be replaced by the corresponding value from features. The array must hold set_size times the number of features entries! There is no consistency check!
	 * \param set_size number of feature vectors in set_features
	 * 
	 * \return std::tuple<num_t, num_t, num_t> mean and variance of empirical mean prediction of a feature vector averaged over. The last one is the estimated variance of a sample drawn from partial assignment.
	 */
    /*
	std::tuple<num_t, num_t, num_t> predict_mean_var_of_mean_response_on_set (num_t *features, num_t* set_features, index_t set_size){

			num_t fv[num_features];

			rfr::util::running_statistics<num_t> mean_stats, var_stats, sample_var_stats, sample_mean_stats;

			for (auto &t : the_trees){

					rfr::util::running_statistics<num_t> tree_mean_stats, tree_var_stats;

					for (auto i=0u; i < set_size; ++i){

							rfr::util::merge_two_vectors(features, &set_features[i*num_features], fv, num_features);

							num_t m , v; index_t n;
							std::tie(m, v, n) = t.predict_mean_var_N(fv);

							tree_mean_stats(m); tree_var_stats(v); sample_mean_stats(m); sample_var_stats(v);
					}

					mean_stats(tree_mean_stats.mean());
					var_stats(std::max<num_t>(0, tree_var_stats.mean()));
					
			}
			
			return(std::make_tuple(mean_stats.mean(), std::max<num_t>(0, mean_stats.variance()) + std::max<num_t>(0, var_stats.mean()/set_size), std::max<num_t>(0,sample_mean_stats.variance() + sample_var_stats.mean())));
	}
    */



	/* \brief yields the partition of the feature space induces by one tree
	 * 
	 * Every split in the tree divides the input space into two partitions.
	 * This means that every leaf of the tree corresponds to a 'rectangular
	 * domain'. This function finds all leaves and computes a representation
	 * of the partitioning.
	 * 
	 * Works for axis aligned splits only!
	 * 
	 * \param tree_index the index of the tree in the forest who's partitioning is requested
	 * \param pcs a representation of the parameter configuration space
	 * 
	 * \return std::vector<std::vector< std::vector<num_t> > > A vector of nested vectors representing intervals (numerical features) and possible values (categorical features) of each dimension.
	 */
	std::vector<std::vector< std::vector<num_t> > > partition_of_tree( index_t tree_index,
														std::vector<std::vector<num_t> > pcs){
		return(the_trees.at(tree_index).partition(pcs));
	}
	
	/* \brief returns the predictions of every tree marginalized over the NAN values in the feature_vector
	 * 
	 * TODO: more documentation over how the 'missing values' are handled
	 * 
	 * \param feature_vector non-specfied values (NaN) will be marginalized over according to the training data
	 */
	std::vector<num_t> marginalized_mean_predictions(const std::vector<num_t> &feature_vector) const {
		std::vector<num_t> rv;
		rv.reserve(the_trees.size());
		for (auto &t : the_trees)
			rv.emplace_back(t.marginalized_mean_prediction(feature_vector));
		return(rv);
	}
	
};


}}//namespace rfr::forests
#endif
