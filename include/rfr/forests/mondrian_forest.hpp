#ifndef RFR_MONDRIAN_FOREST_HPP
#define RFR_MONDRIAN_FOREST_HPP

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
class mondrian_forest{
  protected:
	std::vector<tree_t> the_trees;
	index_t num_features;

	std::vector<std::vector<num_t> > bootstrap_sample_weights;
	
	num_t oob_error = NAN;
	
	// the forest needs to remember the data types on which it was trained
	std::vector<index_t> types;

  public:
  	std::vector<tree_t> get_trees()const {return the_trees;}

	forest_options<num_t, response_t, index_t> options;
	index_t internal_index = 0;
	std::string name = "";
	//forest_options<num_t, response_t, index_t> options;


  	/** \brief serialize function for saving forests with cerial*/
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( options, the_trees, num_features, bootstrap_sample_weights, oob_error, types/*, bounds*/);
	}

	mondrian_forest(): options()	{}
	
	//mondrian_forest(mondrian_forest_options<num_t, response_t, index_t> opts): options(opts){}
	mondrian_forest(forest_options<num_t, response_t, index_t> opts): options(opts){}


	virtual ~mondrian_forest()	{};

	/**\brief growing the random forest for a given data set
	 * 
	 * \param data a filled data container
	 * \param rng the random number generator to be used
	 */
	virtual void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_t &rng){

		if (options.num_trees <= 0)
			throw std::runtime_error("The number of trees has to be positive!");

		if ((!options.do_bootstrapping) && (data.num_data_points() < options.num_data_points_per_tree))
			throw std::runtime_error("You cannot use more data points per tree than actual data point present without bootstrapping!");

		the_trees.resize(options.num_trees);


		std::vector<index_t> data_indices( data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);

		types.resize(data.num_features());
		//bounds.resize(data.num_features());

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
				index_t die = dist(rng);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i){
					die = dist(rng);
                    ++bssf[die];
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
		num_t predicted = 0;
		index_t amount_obb_test = 0;
		num_t pred,s_d, pred_mean;//
		bool bootstrapable = false;
		if (options.compute_oob_error){
			
			rfr::util::running_statistics<num_t> oob_error_stat;
			
			for (auto i=0u; i < data.num_data_points(); i++){

				rfr::util::running_statistics<num_t> prediction_stat;

				for (auto j=0u; j<the_trees.size(); j++){
					// only consider data points that were not part of that bootstrap sample
					if (bootstrap_sample_weights[j][i] == 0){
						amount_obb_test++;
						pred = the_trees[j].predict( data.retrieve_data_point(i), s_d, pred_mean,rng);
						prediction_stat.push(pred);
						bootstrapable = true;
					}
				}
				if(bootstrapable){
					// compute squared error of prediction
					oob_error_stat.push(std::pow(prediction_stat.mean() - data.response(i), 2));	
					oob_error = std::sqrt(oob_error_stat.mean());
				}
				bootstrapable = false;
				
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
    response_t predict( const std::vector<num_t> &feature_vector, response_t &standard_deviation, response_t &mean, rng_t &rng){

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> pred_stats, s_d_stats, pred_mean_stats;

		for (auto &tree: the_trees){
			pred_stats.push(tree.predict(feature_vector, standard_deviation, mean, rng));
			s_d_stats.push(standard_deviation);
			pred_mean_stats.push(mean);
		}
		standard_deviation = s_d_stats.mean();
		mean = pred_mean_stats.mean();
		return(pred_stats.mean());
	}

	response_t predict_deterministic(const std::vector<num_t> &feature_vector) const{

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats;
		for (auto &tree: the_trees)
			mean_stats.push(tree.predict_deterministic(feature_vector));
		return(mean_stats.mean());
	}
    
	response_t predict_median( const std::vector<num_t> &feature_vector, response_t &sd, response_t &mean, rng_t &rng) /*const*/{

		// collect the predictions of individual trees
		index_t top = the_trees.size();
		response_t pred;
		std::vector<response_t> preds, means, sds;
		for (index_t i = 0; i<the_trees.size(); i++){
			pred = the_trees[i].predict(feature_vector, sd, mean, rng);
			preds.emplace_back(pred);
			means.emplace_back(mean);
			sds.emplace_back(sd);
		}
		std::sort (preds.begin(), preds.end()); 
		std::sort (means.begin(), means.end()); 
		std::sort (sds.begin(), sds.end()); 
		if(the_trees.size()%2){
			index_t first = the_trees.size()/2, second = the_trees.size()/2 + 1;
			mean = (means[the_trees.size()/2] + means[the_trees.size()/2 + 1])/2 ;
			sd = (sds[the_trees.size()/2] + sds[the_trees.size()/2 + 1])/2 ;
			return(preds[the_trees.size()/2] + preds[the_trees.size()/2 + 1])/2;
		}
		else{
			mean = means[the_trees.size()/2 + 1];
			sd = sds[the_trees.size()/2 + 1];
			return(preds[the_trees.size()/2 + 1]);
		}
	}


	virtual void partial_fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_t &rng, index_t point){

		if (options.num_trees <= 0)
			throw std::runtime_error("The number of trees has to be positive!");

		if ((!options.do_bootstrapping) && (data.num_data_points() < options.num_data_points_per_tree))
			throw std::runtime_error("You cannot use more data points per tree than actual data point present without bootstrapping!");

		the_trees.resize(options.num_trees);


		std::vector<index_t> data_indices( data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);

		types.resize(data.num_features());
		//bounds.resize(data.num_features());

		num_features = data.num_features();
		
		// catch some stupid things that will make the forest crash when fitting
		if (options.num_data_points_per_tree == 0)
			throw std::runtime_error("The number of data points per tree is set to zero!");
		
		if (options.tree_opts.max_features == 0)
			throw std::runtime_error("The number of features used for a split is set to zero!");
		
		for (auto &tree : the_trees){
			tree.partial_fit(data, options.tree_opts, point, rng);
		}
		oob_error = NAN;
	}
    

	// check
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
		///re do
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
    
	//check
	std::vector< std::vector<num_t> > all_leaf_values (const std::vector<num_t> &feature_vector) const {
		std::vector< std::vector<num_t> > rv;
		rv.reserve(the_trees.size());

		for (auto &t: the_trees){
			rv.push_back(t.leaf_entries(feature_vector));
		}
		return(rv);
	}


	//check	
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

	//check
	/* \brief aggregates all used split values for all features in each tree
	 *
	 * TODO: move to fANOVA forest
	 */
	std::vector<std::vector<std::vector<num_t> > > all_split_values(const std::vector<index_t> &types){
		std::vector<std::vector<std::vector<num_t> > > rv;
		rv.reserve(the_trees.size());
			
		for (auto &t: the_trees)
			rv.emplace_back(t.all_split_values(types));
		return(rv);
	}

	//check
	/* \brief updates the forest by adding all provided datapoints without a complete retraining
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
	
	//check
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
	
};


}}//namespace rfr::forests
#endif
