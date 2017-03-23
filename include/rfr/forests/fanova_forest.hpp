#ifndef RFR_FANOVA_FOREST_HPP
#define RFR_FANOVA_FOREST_HPP

#include "rfr/forests/regression_forest.hpp"
#include "rfr/trees/binary_fanova_tree.hpp"


namespace rfr{ namespace forests{

template <typename split_t, typename num_t = float, typename response_t = float, typename index_t = unsigned int,  typename rng_t=std::default_random_engine>
class fANOVA_forest: public	rfr::forests::regression_forest< rfr::trees::binary_fANOVA_tree<split_t, num_t, response_t, index_t, rng_t>, num_t, response_t, index_t, rng_t> {
  private:

	typedef rfr::forests::regression_forest<rfr::trees::binary_fANOVA_tree<split_t, num_t, response_t, index_t, rng_t> , num_t, response_t, index_t, rng_t> super;

	std::vector<std::vector<num_t> > pcs;
	
  protected:
	// to compute 'improvement over default' and such...
	num_t lower_cutoff;
	num_t upper_cutoff;

  public:

	fANOVA_forest() : 	super(),
						lower_cutoff (-std::numeric_limits<num_t>::infinity()),
						upper_cutoff (std::numeric_limits<num_t>::infinity()) {}
						
	fANOVA_forest (forest_options<num_t, response_t, index_t> forest_opts): super(forest_opts),
						lower_cutoff (-std::numeric_limits<num_t>::infinity()),
						upper_cutoff (std::numeric_limits<num_t>::infinity())
						{};

	virtual ~fANOVA_forest()	{};


  	/** \brief serialize function for saving forests with cerial*/
  	template<class Archive>
	void serialize(Archive & archive)
	{
		super::serialize( archive);
		archive ( lower_cutoff, upper_cutoff);
	}

	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_t &rng){
		// fit the forest normaly
		super::fit(data, rng);
		
		pcs.reserve(super::types.size());
		
		for (auto i=0u; i<super::types.size(); ++i){
			if (super::types[i] == 0)
				pcs.emplace_back(std::begin(super::bounds[i]), std::end(super::bounds[i]));
			else{
				pcs.emplace_back(super::types[i], 0);
				std::iota(pcs.back().begin(), pcs.back().end(), 0);
			}
		}
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
		precompute_marginals();
	}

	/* \brief to read out the used cutoffs */
	std::pair<num_t, num_t> get_cutoffs(){ return(std::pair<num_t, num_t> (lower_cutoff, upper_cutoff));}


	/* \brief just calls the precompute marginals function of every tree */
	void precompute_marginals(){
			
		for (auto &t: super::the_trees)
			t.precompute_marginals(lower_cutoff, upper_cutoff, pcs, super::types);
	}


	/* \brief returns the marginal prediction when some variables are not specified (NANs)
	 *
	 * this function implements equation 1 of
	 * "An efficient Approach for Assessing Hyperparameter Importance"
	 * by Hutter et al.
	 * 
	 * \returns mean of all trees' mean predictions
	 */

	num_t marginal_mean_prediction( const std::vector<num_t> & feature_vector){
		if (std::isnan(lower_cutoff))
			set_cutoffs(	-std::numeric_limits<num_t>::infinity(),
							 std::numeric_limits<num_t>::infinity());
		rfr::util::running_statistics<num_t> stat;
		
		for (auto &t: super::the_trees){
			auto m = t.marginalized_prediction_stat(feature_vector, pcs, super::types).mean();
			if (! std::isnan(m))
				stat.push(m);
		}
		
		return(stat.mean());
	}

	std::pair<num_t, num_t> marginal_mean_variance_prediction(const std::vector<num_t> & feature_vector){
		rfr::util::running_statistics<num_t> stat;
		
		for (auto &t: super::the_trees){
			num_t v = t.marginalized_prediction_stat(feature_vector, pcs, super::types).mean();
			if (!std::isnan(v))
				stat.push(v);
		}
			
		return(std::pair<num_t, num_t> (stat.mean(), stat.variance_sample()));
	}


	rfr::util::weighted_running_statistics<num_t> marginal_prediction_stat_of_tree( index_t tree_index, const std::vector<num_t> & feature_vector){
		if (std::isnan(lower_cutoff))
			set_cutoffs(	-std::numeric_limits<num_t>::infinity(),
							 std::numeric_limits<num_t>::infinity());

		auto &t = super::the_trees.at(tree_index);
		return(t.marginalized_prediction_stat(feature_vector, pcs, super::types));
	}

	std::vector<num_t> get_trees_total_variances (){
		std::vector<num_t> r; r.reserve(super::the_trees.size());
		for (auto &t: super::the_trees)
			r.push_back(t.get_total_variance());
		return(r);
	}



	/* \brief aggregates all used split values for all features in each tree
	 *
	 */
	std::vector<std::vector<std::vector<num_t> > > all_split_values(){
		std::vector<std::vector<std::vector<num_t> > > rv;
		rv.reserve(super::the_trees.size());
			
		for (auto &t: super::the_trees)
			rv.emplace_back(t.all_split_values(super::types));
		return(rv);
	}


};


}}//namespace rfr::forests
#endif
