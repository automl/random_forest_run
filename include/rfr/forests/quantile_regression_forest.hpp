#ifndef RFR_QUANTILE_REGRESSION_FOREST_HPP
#define RFR_QUANTILE_REGRESSION_FOREST_HPP

#include "rfr/forests/regression_forest.hpp"

namespace rfr{ namespace forests{


template <typename tree_t, typename num_t = float, typename response_t = float, typename index_t = unsigned int,  typename rng_t=std::default_random_engine>
class quantile_regression_forest: public rfr::forests::regression_forest<tree_t, num_t, response_t, index_t, rng_t> {
  private:
	typedef rfr::forests::regression_forest<tree_t, num_t, response_t, index_t, rng_t> super;

  public:

	quantile_regression_forest() : super()	{};
	quantile_regression_forest (forest_options<num_t, response_t, index_t> forest_opts): super(forest_opts) {};

	

	virtual ~quantile_regression_forest()	{};

	/* \brief implements the quantile regression forests of Meinshausen (2006)
	 *
	 * 	\param feature_vector you guessed it :)
	 *  \param quantiles a vector of all the quantiles to predict.
	 *
	 * 	\return std::vector<num_t> list of the corresponding response values
	 */

	std::vector<num_t> predict_quantiles (const std::vector<num_t> &feature_vector, std::vector<num_t> quantiles) const {

		std::sort(quantiles.begin(), quantiles.end());

		if (*quantiles.begin() < 0)
			throw std::runtime_error("quantiles cannot be <0.");

		if (quantiles.back() > 1)
			throw std::runtime_error("quantiles cannot be >1.");
		

		std::vector<std::pair<num_t, num_t> > value_weight_pairs;
		
		for (auto &t: super::the_trees){
			auto l = t.get_leaf(feature_vector);

			auto values = l.responses();
			auto weights = l.weights();

			auto stat = l.leaf_statistic();

			num_t factor = 1./(stat.sum_of_weights()*super::the_trees.size());
			for (auto i=0u; i<values.size(); ++i){
				value_weight_pairs.emplace_back(values[i], weights[i]*factor);
			}
		}
		std::sort(value_weight_pairs.begin(), value_weight_pairs.end(),
			[] (const std::pair<num_t, num_t> &a, const std::pair<num_t, num_t> &b) {return (a.first < b.first);}
		);

		std::vector<num_t> rv;
		rv.reserve(quantiles.size());

		num_t tw = 0;
		index_t index = 0;

		for (auto q: quantiles){
			while ( (index < value_weight_pairs.size()) && (tw < q) ){
				tw += value_weight_pairs[index].second;
				index++;
			}

			auto mew = std::min<unsigned int>(index,  value_weight_pairs.size()-1);
			
			rv.push_back((value_weight_pairs[mew]).first);
		}

		return(rv);
	}

};


}}//namespace rfr::forests
#endif
