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
#include <cereal/archives/portable_binary.hpp>
#include <iostream>
#include <sstream>



#include "rfr/trees/tree_options.hpp"
#include "rfr/forests/forest_options.hpp"
#include "rfr/util.hpp"

namespace rfr{ namespace forests{

typedef cereal::PortableBinaryInputArchive iarch_t;
typedef cereal::PortableBinaryOutputArchive oarch_t;



template <typename tree_type, typename num_t = float, typename response_t = float, typename index_t = unsigned int,  typename rng_type=std::default_random_engine>
class regression_forest{
  private:
	std::vector<tree_type> the_trees;
	index_t num_features;

	std::vector<std::vector<index_t> > dirty_leafs;
	std::vector<std::vector<num_t> > bootstrap_sample_weights;
	
	num_t oob_error;

  public:

	forest_options<num_t, response_t, index_t> options;


  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( options, the_trees, num_features, dirty_leafs, bootstrap_sample_weights, oob_error);
	}


	regression_forest(): options(), the_trees(){}


	regression_forest(forest_options<num_t, response_t, index_t> options): options(options){}


	/* \brief growing the random forest for a given data set
	 * 
	 * \param data a filled data container
	 * \param rng the random number generator to be used
	 */
	void fit(const rfr::data_containers::base<num_t, response_t, index_t> &data, rng_type &rng){

		if ((!options.do_bootstrapping) && (data.num_data_points() < options.num_data_points_per_tree))
			throw std::runtime_error("You cannot use more data points per tree than actual data point present without bootstrapping!");

		the_trees.resize(options.num_trees);


		std::vector<index_t> data_indices( data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);
		std::vector<index_t> data_indices_to_be_used( options.num_data_points_per_tree);

		num_features = data.num_features();
		
		// catch some stupid things that will make the forest crash when fitting
		if (options.num_data_points_per_tree == 0)
			throw std::runtime_error("The number of data points per tree is set to zero!");
		
		if (options.tree_opts.max_features == 0)
			throw std::runtime_error("The number of features used for a split is set to zero!");
		
		bootstrap_sample_weights.clear();

		for (auto &tree : the_trees){
            std::vector<num_t> bssw (data.num_data_points(), 0);
			// prepare the data(sub)set
			if (options.do_bootstrapping){
                std::uniform_int_distribution<index_t> dist (0,data.num_data_points()-1);
                auto die = std::bind(dist, rng);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i)
                    ++bssw[die()];
			}
			else{
				std::shuffle(data_indices.begin(), data_indices.end(), rng);
                for (auto i=0u; i < options.num_data_points_per_tree; ++i)
                    ++bssw[data_indices[i]];
			}
			tree.fit(data, options.tree_opts, bssw, rng);
			
			// record sample counts for later use
			if (options.compute_oob_error){
				bootstrap_sample_weights.emplace_back(std::vector<num_t> ( data.num_data_points(),0));
				for (auto &i: data_indices_to_be_used){
					++(bootstrap_sample_weights.back()[i]);
				}
			}
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
	 * \param feature_vector a valid (size and values) array containing features
	 *
	 * \return response_t the predicted value
	 */

    response_t predict( const std::vector<num_t> &feature_vector) const{

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats;
		for (auto &tree: the_trees){
			std::cout<<tree.predict(feature_vector)<<std::endl;
			mean_stats.push(tree.predict(feature_vector));
		}
		std::cout<<mean_stats.number_of_points()<<std::endl;
		return(mean_stats.mean());
	}
    
    
    /*
    std::pair<num_t, num_t> predict_mean_var( num_t * feature_vector){

		// collect the predictions of individual trees
		rfr::util::running_statistics<num_t> mean_stats, var_stats;
		for (auto &tree: the_trees){
			num_t m , v;	index_t n;

			std::tie(m, v, n) = tree.predict_mean_var_N(feature_vector);

			mean_stats(m); 
			var_stats(v);
		}
		
		return(std::pair<num_t, num_t> (mean_stats.mean(), std::max<num_t>(0, mean_stats.variance() + var_stats.mean()) ));
	}
	*/

	/* \brief combines the prediction of all trees in the forest
	 *
	 * Every random tree makes an individual prediction. From that, the mean and the standard
	 * deviation of those predictions is calculated. (See Frank's PhD thesis section 11.?)
	 *
	 * \param feature_vector a valid (size and values) array containing features
	 *
	 * \return std::pair<num_t, num_t> mean and sqrt(total variance = mean of variances + variance of means )
	 */
    
    /*
	std::pair<num_t, num_t> predict_mean_std( num_t * feature_vector){
		auto p = predict_mean_var(feature_vector);
		p.second = sqrt(p.second);
		return(p);
	}
    */

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
	 * The covariance between to input vectors contains information about the
	 * feature space. For other models, like GPs, this is a natural quantity
	 * (e.g., property of the kernel). Here, we try to estimate it using the
	 * total covariance of the individual tree's prediction and the assumption
	 * that cov(X,Y) = 0, if X and Y fall into different leafs, and cov(X,Y) = var(X) = var(Y)
	 * otherwise.
	 * 
	 * \param f1 a valid feature vector (no sanity checks are performed!)
	 * \param f2 a second feature vector (no sanity checks are performed!)
	 */
    
    /*
	num_t covariance (num_t* f1, num_t* f2){
		rfr::util::running_statistics<num_t> cov_stats;
		rfr::util::running_covariance<num_t> run_cov_of_means;
		
		for (auto &t: the_trees){
			auto l1 = t.find_leaf(f1);
			auto l2 = t.find_leaf(f2);
		
			num_t m1 , v1;	index_t n;
			std::tie(m1, v1, n) = t.predict_mean_var_N(f1);

			num_t m2 , v2;
			std::tie(m2, v2, n) = t.predict_mean_var_N(f2);
			
			// assumption here: cov = 0 if the leafs are different, and cov = var if both feature vectors fall into the same leaf
			if (l1 == l2) cov_stats(v2);
			else cov_stats(0);
			
			run_cov_of_means(m1,m2);
		}
		return(cov_stats.mean() + run_cov_of_means.covariance());
	}
	*/

    
	std::vector< std::vector<num_t> > all_leaf_values (const std::vector<num_t> &feature_vector){
		std::vector< std::vector<num_t> > rv;
		rv.reserve(the_trees.size());

		for (auto &t: the_trees){
			rv.push_back(t.leaf_entries(feature_vector));
		}
		return(rv);
	}

	forest_options<num_t, response_t, index_t> get_forest_options(){return(options);}

	std::vector<std::vector< std::vector<num_t> > > partition_of_tree( index_t tree_index,
														std::vector<std::vector<num_t> > pcs){
		return(the_trees[tree_index].partition(pcs));
	}
	
	/* \brief returns the predictions of every tree marginalized over the NAN values in the feature_vector
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

	std::vector<std::vector<std::vector<num_t> > > all_split_values(const std::vector<index_t> &types){
		std::vector<std::vector<std::vector<num_t> > > rv;
		rv.reserve(the_trees.size());
			
		for (auto &t: the_trees)
			rv.emplace_back(t.all_split_values(types));
		return(rv);
	}


	/* \brief updates the forest by adding all provided datapoints without a complete retraining
	 * 
	 * As retraining can be quite expensive, this function can be used to quickly update the forest
	 * by finding the leafs the datapoints belong into and just inserting them. This is, of course,
	 * not the right way to do it for many data points, but it should be a good approximation for a few.
	 * 
	 * \param data a data container instance that will be inserted into the tree
	 */
	void pseudo_update (const rfr::data_containers::base<num_t, response_t, index_t> &data){
		for (auto i=0u; i<data.num_data_points(); ++i){
			auto p = data.retrieve_data_point(i);
			dirty_leafs.emplace_back(std::vector<index_t> (the_trees.size(),0));
			auto it = (dirty_leafs.back()).begin();
			//for each tree
			for (auto &t: the_trees){
		
				index_t index = t.find_leaf(p);
		
				// add value
				t.the_nodes[index].push_response_value(data.response(i), data.weight(i));
		
				// note leaf as changed
				(*it) = index;
				it++;
			}
		}
	}
	
	/* \brief undoing a pseudo update by removing the last point added
	 * 
	 * This function removes one point from the corresponding leaves that
	 * were touched during a pseudo update.
	 * 
	 * \return bool whether the tree was altered
	 */
	bool pseudo_downdate(){
		if (dirty_leafs.empty())
			return(false);
		auto i = 0u;
		for (auto li: dirty_leafs.back())
			the_trees[i++].the_nodes[li].pop_repsonse_value();
		dirty_leafs.pop_back();
		return(true);
	}
	
	num_t out_of_bag_error(){return(oob_error);}

	// writes serialized representation into string (used for pickling in python)
	void save_to_binary_file(const std::string filename){
		std::ofstream ofs(filename, std::ios::binary);
		oarch_t oarch(ofs);
		serialize(oarch);
	}

	// deserialize from a representation provided by the string (used for unpickling in python)
	void load_from_binary_file(const std::string filename){
		std::ifstream ifs(filename, std::ios::binary);
		std::cout<<"opening file "<<filename<<std::endl;
		iarch_t iarch(ifs);
		serialize(iarch);
	}

	// serialize into a string; used for Python's pickle.dump
	std::string save_into_string(){
		std::stringstream oss;
		oarch_t oarch(oss);
		serialize(oarch);
		return(oss.str());
	}

	// deserialize from string; used for Python's pickle.load
	void load_from_string( std::string str){
		std::stringstream iss;
		iss.str(str);
		iarch_t iarch(iss);
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
