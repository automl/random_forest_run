#ifndef RFR_REGRESSION_FOREST_HPP
#define RFR_REGRESSION_FOREST_HPP

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




template <typename tree_type, typename num_t = float, typename response_t = float, typename index_t = unsigned int,  typename rng_type=std::default_random_engine>
class regression_forest{
  protected:
	std::vector<tree_type> the_trees;
	index_t num_features;

	std::vector<std::vector<num_t> > bootstrap_sample_weights;
	
	num_t oob_error = NAN;
	
	// the forest needs to remember the data types on which it was trained
	std::vector<index_t> types;
	std::vector< std::array<num_t,2> > bounds;
	

  public:

	forest_options<num_t, response_t, index_t> options;


  	/** \brief serialize function for saving forests with cerial*/
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( options, the_trees, num_features, bootstrap_sample_weights, oob_error, types, bounds);
	}

	regression_forest(): options()	{}
	
	regression_forest(forest_options<num_t, response_t, index_t> opts): options(opts){}

	virtual ~regression_forest()	{};

	/**\brief growing the random forest for a given data set
	 * 
	 * \param data a filled data container
	 * \param rng the random number generator to be used
	 */
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_type &rng){

		if (options.num_trees <= 0)
			throw std::runtime_error("The number of trees has to be positive!");

		if ((!options.do_bootstrapping) && (data.num_data_points() < options.num_data_points_per_tree))
			throw std::runtime_error("You cannot use more data points per tree than actual data point present without bootstrapping!");


		types.resize(data.num_features());
		bounds.resize(data.num_features());
		for (auto i=0u; i<data.num_features(); ++i){
			types[i] = data.get_type_of_feature(i);
			auto p = data.get_bounds_of_feature(i);
			bounds[i][0] = p.first;
			bounds[i][1] = p.second;
		}

		the_trees.resize(options.num_trees);


		std::vector<index_t> data_indices( data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);

		num_features = data.num_features();
		
		// catch some stupid things that will make the forest crash when fitting
		if (options.num_data_points_per_tree == 0)
			throw std::runtime_error("The number of data points per tree is set to zero!");
		
		if (options.tree_opts.max_features == 0)
			throw std::runtime_error("The number of features used for a split is set to zero!");
		
		bootstrap_sample_weights.clear();

		for (auto &tree : the_trees){
            std::vector<num_t> bssf (data.num_data_points(), 0); // BootStrap Sample Frequencies
			// prepare the data(sub)set
			if (options.do_bootstrapping){
                std::uniform_int_distribution<index_t> dist (0,data.num_data_points()-1);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i){
					++bssf[dist(rng)]+=1;
				}
			}
			else{
				std::shuffle(data_indices.begin(), data_indices.end(), rng);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i)
                    ++bssf[data_indices[i]];
			}
			
			tree.fit(data, options.tree_opts, bssf, rng);
			
			// record sample counts for later use
			if (options.compute_oob_error)
				bootstrap_sample_weights.push_back(bssf);
		}
		
		oob_error = NAN;
		
		if (options.compute_oob_error){
			
			rfr::util::running_statistics<num_t> oob_error_stat;
			
			for (auto i=0u; i < data.num_data_points(); i++){

				rfr::util::running_statistics<num_t> prediction_stat;

				for (auto j=0u; j<the_trees.size(); j++){
					// only consider data points that were not part of that bootstrap sample
					if (bootstrap_sample_weights[j][i] == 0)
						prediction_stat.push(the_trees[j].predict( data.retrieve_data_point(i)));
				}
				
				// compute squared error of prediction
				oob_error_stat.push(std::pow(prediction_stat.mean() - data.response(i), 2));
			}
			oob_error = std::sqrt(oob_error_stat.mean());
		}
	}


	/* \brief combines the prediction of all trees in the forest
	 *
	 * Every random tree makes an individual prediction which are averaged for the forest's prediction.
	 *
	 * \param feature_vector a valid vector containing the features
	 * \return response_t the predicted value
	 */
    response_t predict( const std::vector<num_t> &feature_vector) const{

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats;
		for (auto &tree: the_trees)
			mean_stats.push(tree.predict(feature_vector));
		return(mean_stats.mean());
	}
    
    
   /* \brief makes a prediction for the mean and a variance estimation
    * 
    * Every tree returns the mean and the variance of the leaf the feature vector falls into.
    * These are combined to the forests mean prediction (mean of the means) and a variance estimate
    * (mean of the variance + variance of the means).
    * 
    * Use weighted_data = false if the weights assigned to each data point were frequencies, not importance weights.
    * Use this if you haven't assigned any weigths, too.
    * 
	* \param feature_vector a valid feature vector
	* \param weighted_data whether the data had importance weights
	* \return std::pair<response_t, num_t> mean and variance prediction
    */
    std::pair<num_t, num_t> predict_mean_var( const std::vector<num_t> &feature_vector, bool weighted_data = false){

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats, var_stats;
		for (auto &tree: the_trees){
			auto stat = tree.leaf_statistic(feature_vector);
			mean_stats.push(stat.mean()); 
			if (weighted_data) var_stats.push(stat.variance_unbiased_importance());
			else var_stats.push(stat.variance_unbiased_frequency());
		}
		
		return(std::pair<num_t, num_t> (mean_stats.mean(), std::max<num_t>(0, mean_stats.variance_sample() + var_stats.mean()) ));
	}


	/* \brief predict the mean and the variance deviation for a configuration marginalized over a given set of partial configurations
	 * 
	 * This function will be mostly used to predict the mean over a given set of instances, but could be used to marginalize over any discrete set of partial configurations.
	 * 
	 * \param features a (partial) configuration where unset values should be set to NaN
	 * \param set_features a array containing the (partial) assignments used for the averaging. Every NaN value will be replaced by the corresponding value from features.
	 * \param set_size number of feature vectors in set_features
	 * 
	 * \return std::pair<num_t, num_t> mean and variance prediction of a feature vector averaged over 
	 */
    /*
	std::pair<num_t, num_t> predict_mean_var_marginalized_over_set (num_t *features, num_t* set_features, index_t set_size){
		
		num_t fv[num_features];

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats, var_stats;
		for (auto i=0u; i < set_size; ++i){
			// construct the actual feature vector
			rfr::util::merge_two_vectors(features, &set_features[i*num_features], fv, num_features);

			num_t m , v;
			std::tie(m, v) = predict_mean_var(fv);

			mean_stats(m);
			var_stats(v);
		}
		return(std::pair<num_t, num_t> (mean_stats.mean(), std::max<num_t>(0, mean_stats.variance() + var_stats.mean()) ));
	}
    */

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

	/* \brief estimates the covariance of two feature vectors
	 * 
	 * 
	 * The covariance between to input vectors contains information about the
	 * feature space. For other models, like GPs, this is a natural quantity
	 * (e.g., property of the kernel). Here, we try to estimate it using the
	 * emprical covariance of the individual tree's predictions.
	 * 
	 * \param f1 a valid feature vector (no sanity checks are performed!)
	 * \param f2 a second feature vector (no sanity checks are performed!)
	 */
    
	num_t covariance (const std::vector<num_t> &f1, const std::vector<num_t> &f2){

		rfr::util::running_covariance<num_t> run_cov_of_means;

		for (auto &t: the_trees)
			run_cov_of_means.push(t.predict(f1),t.predict(f2));

		return(run_cov_of_means.covariance());
	}



	/* \brief computes the kernel of a 'Kernel Random Forest'
	 *
	 * Source: "Random forests and kernel methods" by Erwan Scornet
	 * 
	 * The covariance between to input vectors contains information about the
	 * feature space. For other models, like GPs, this is a natural quantity
	 * (e.g., property of the kernel). Here, we try to estimate it using the
	 * emprical covariance of the individual tree's predictions.
	 * 
	 * \param f1 a valid feature vector (no sanity checks are performed!)
	 * \param f2 a second feature vector (no sanity checks are performed!)
	 */
    
	num_t kernel (const std::vector<num_t> &f1, const std::vector<num_t> &f2){

		rfr::util::running_statistics<num_t> stat;

		for (auto &t: the_trees){
			auto l1 = t.find_leaf_index(f1);
			auto l2 = t.find_leaf_index(f2);

			stat.push(l1==l2);
		}
		return(stat.mean());
	}



    
	std::vector< std::vector<num_t> > all_leaf_values (const std::vector<num_t> &feature_vector) const {
		std::vector< std::vector<num_t> > rv;
		rv.reserve(the_trees.size());

		for (auto &t: the_trees){
			rv.push_back(t.leaf_entries(feature_vector));
		}
		return(rv);
	}


	
	/* \brief returns the predictions of every tree marginalized over the NAN values in the feature_vector
	 * 
	 * TODO: more documentation over how the 'missing values' are handled
	 * 
	 * \param feature_vector non-specfied values (NaN) will be marginalized over according to the training data
	 */
	//std::vector<num_t> marginalized_mean_predictions(const std::vector<num_t> &feature_vector) const {
	//	std::vector<num_t> rv;
	//	rv.reserve(the_trees.size());
	//	for (auto &t : the_trees)
	//		rv.emplace_back(t.marginalized_mean_prediction(feature_vector));
	//	return(rv);
	//}



	/* \brief updates the forest by adding the provided datapoint without a complete retraining
	 * 
	 * 
	 * As retraining can be quite expensive, this function can be used to quickly update the forest
	 * by finding the leafs the datapoints belong into and just inserting them. This is, of course,
	 * not the right way to do it for many data points, but it should be a good approximation for a few.
	 * 
	 * \param features a valid feature vector
	 * \param response the corresponding response value
	 * \param weight the associated weight
	 */
	void pseudo_update (std::vector<num_t> features, response_t response, num_t weight){
		for (auto &t: the_trees)
			t.pseudo_update(features, response, weight);
	}
	
	/* \brief undoing a pseudo update by removing a point
	 * 
	 * This function removes one point from the corresponding leaves into
	 * which the given feature vector falls
	 * 
	 * \param features a valid feature vector
	 * \param response the corresponding response value
	 * \param weight the associated weight
	 */
	void pseudo_downdate(std::vector<num_t> features, response_t response, num_t weight){
		for (auto &t: the_trees)
			t.pseudo_downdate(features, response, weight);
	}
	
	num_t out_of_bag_error(){return(oob_error);}

	/* \brief writes serialized representation into a binary file
	 * 
	 * \param filename name of the file to store the forest in. Make sure that the directory exists!
	 */
	void save_to_binary_file(const std::string filename){
		std::ofstream ofs(filename, std::ios::binary);
		binary_oarch_t oarch(ofs);
		serialize(oarch);
	}

	/* \brief deserialize from a binary file created by save_to_binary_file
	 *
	 * \param filename name of the file in which the forest is stored. 
	 */
	void load_from_binary_file(const std::string filename){
		std::ifstream ifs(filename, std::ios::binary);
		binary_iarch_t iarch(ifs);
		serialize(iarch);
	}

	/* serialize into a string; used for Python's pickle.dump
	 * 
	 * \return std::string a JSON serialization of the forest
	 */
	std::string ascii_string_representation(){
		std::stringstream oss;
		{
			ascii_oarch_t oarch(oss);
			serialize(oarch);
		}
		return(oss.str());
	}

	/* \brief deserialize from string; used for Python's pickle.load
	 * 
	 * \return std::string a JSON serialization of the forest
	 */
	void load_from_ascii_string( std::string const &str){
		std::stringstream iss;
		iss.str(str);
		ascii_iarch_t iarch(iss);
		serialize(iarch);
	}



	/* \brief stores a latex document for every individual tree
	 * 
	 * \param filename_template a string to specify the location and the naming scheme. Note the directory is not created, so make sure it exists.
	 * 
	 */
	void save_latex_representation(const std::string filename_template){
		for (auto i = 0u; i<the_trees.size(); i++){
			std::stringstream filename;
			filename << filename_template<<i<<".tex";
			the_trees[i].save_latex_representation(filename.str().c_str());
		}
	}

	void print_info(){
		for (auto t: the_trees){
			t.print_info();
		}
	}


	virtual unsigned int num_trees (){ return(the_trees.size());}
	
};


}}//namespace rfr::forests
#endif
