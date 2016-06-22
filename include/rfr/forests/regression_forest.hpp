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

typedef cereal::PortableBinaryInputArchive iarch_type;
typedef cereal::PortableBinaryOutputArchive oarch_type;



template <typename tree_type, typename rng_type, typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class regression_forest2{
  private:
	forest_options<num_type, response_type, index_type> forest_opts;
	std::vector<tree_type> the_trees;
	index_type num_features;
	//std::vector<std::vector<index_type> > dirty_leafs;
	//std::vector<std::vector<index_type> > bootstrap_sample_counts;
	
	num_type oob_error;

  public:

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive)
	{
		//archive( forest_opts, the_trees, num_features, dirty_leafs, bootstrap_sample_counts, oob_error);
		archive( forest_opts, the_trees, num_features);
	}


	regression_forest2(): forest_opts(), the_trees(){}

	//regression_forest2(forest_options<num_type, response_type, index_type> forest_opts): forest_opts(forest_opts), num_features(42){}
	regression_forest2(forest_options<num_type, response_type, index_type> forest_opts): forest_opts(forest_opts), the_trees(forest_opts.num_trees), num_features(42){}
};


template <typename tree_type, typename rng_type, typename num_type = float, typename response_type = float, typename index_type = unsigned int>
class regression_forest{
  private:
	forest_options<num_type, response_type, index_type> forest_opts;
	std::vector<tree_type> the_trees;
	index_type num_features;

	std::vector<std::vector<index_type> > dirty_leafs;
	std::vector<std::vector<index_type> > bootstrap_sample_counts;
	
	num_type oob_error;

  public:

  	/* serialize function for saving forests */
  	template<class Archive>
	void serialize(Archive & archive)
	{
		archive( forest_opts, the_trees, num_features, dirty_leafs, bootstrap_sample_counts, oob_error);
	}


	regression_forest(): forest_opts(), the_trees(){}


	regression_forest(forest_options<num_type, response_type, index_type> forest_opts): forest_opts(forest_opts), the_trees(forest_opts.num_trees){}


	/* \brief growing the random forest for a given data set
	 * 
	 * \param data a filled data container
	 * \param rng the random number generator to be used
	 */
	void fit(const rfr::data_containers::data_container_base<num_type, response_type, index_type> &data, rng_type &rng){

		if ((!forest_opts.do_bootstrapping) && (data.num_data_points() < forest_opts.num_data_points_per_tree))
			throw std::runtime_error("You cannot use more data points per tree than actual data point present without bootstrapping!");

		std::vector<index_type> data_indices( data.num_data_points());
		std::iota(data_indices.begin(), data_indices.end(), 0);
		std::vector<index_type> data_indices_to_be_used( forest_opts.num_data_points_per_tree);

		num_features = data.num_features();
		
		// catch some stupid things that will make the forest crash when fitting
		if (forest_opts.num_data_points_per_tree == 0)
			throw std::runtime_error("The number of data points per tree is set to zero!");
		
		if (forest_opts.tree_opts.max_features == 0)
			throw std::runtime_error("The number of features used for a split is set to zero!");
		
		bootstrap_sample_counts.clear();

		for (auto &tree : the_trees){
			// prepare the data(sub)set
			if (forest_opts.do_bootstrapping){
				std::uniform_int_distribution<index_type> dist (0,data.num_data_points()-1);
				auto dice = std::bind(dist, rng);
				std::generate_n(data_indices_to_be_used.begin(), data_indices_to_be_used.size(), dice);
			}
			else{
				std::shuffle(data_indices.begin(), data_indices.end(), rng);
				data_indices_to_be_used.assign(data_indices.begin(), data_indices.begin()+ forest_opts.num_data_points_per_tree);
			}
			tree.fit(data, forest_opts.tree_opts, data_indices_to_be_used, rng);
			
			// record sample counts for later use
			if (forest_opts.compute_oob_error){
				bootstrap_sample_counts.emplace_back(std::vector<index_type> ( data.num_data_points(),0));
				for (auto &i: data_indices_to_be_used){
					++(bootstrap_sample_counts.back()[i]);
				}
			}
		}
		
		oob_error = NAN;
		
		if (forest_opts.compute_oob_error){
			
			rfr::running_statistics<num_type> oob_error_stat;
			
			for (auto i=0u; i < data.num_data_points(); i++){

				rfr::running_statistics<num_type> prediction_stat;

				for (auto j=0u; j<the_trees.size(); j++){
					// only consider data points that were not part of that bootstrap sample
					if (bootstrap_sample_counts[j][i] == 0)
						prediction_stat (the_trees[j].predict( data.retrieve_data_point(i).data()));
				}
				
				// compute squared error of prediction
				oob_error_stat(std::pow(prediction_stat.mean() - data.response(i), 2));
			}
			oob_error = std::sqrt(oob_error_stat.mean());
		}
	}


	/* \brief combines the prediction of all trees in the forest
	 *
	 * Every random tree makes an individual prediction. From that, the mean and the standard
	 * deviation of those predictions is calculated. (See Frank's PhD thesis section 11.?)
	 *
	 * \param feature_vector a valid (size and values) array containing features
	 *
	 * \return std::pair<num_type, num_type> mean and total variance (= mean of variances + variance of means )
	 */
	std::pair<num_type, num_type> predict_mean_var( num_type * feature_vector){

		// collect the predictions of individual trees
		rfr::running_statistics<num_type> mean_stats, var_stats;
		for (auto &tree: the_trees){
			num_type m , v;	index_type n;

			std::tie(m, v, n) = tree.predict_mean_var_N(feature_vector);

			mean_stats(m); 
			var_stats(v);
		}
		
		return(std::pair<num_type, num_type> (mean_stats.mean(), std::max<num_type>(0, mean_stats.variance() + var_stats.mean()) ));
	}


	/* \brief combines the prediction of all trees in the forest
	 *
	 * Every random tree makes an individual prediction. From that, the mean and the standard
	 * deviation of those predictions is calculated. (See Frank's PhD thesis section 11.?)
	 *
	 * \param feature_vector a valid (size and values) array containing features
	 *
	 * \return std::pair<num_type, num_type> mean and sqrt(total variance = mean of variances + variance of means )
	 */
	std::pair<num_type, num_type> predict_mean_std( num_type * feature_vector){
		auto p = predict_mean_var(feature_vector);
		p.second = sqrt(p.second);
		return(p);
	}


	/* \brief predict the mean and the variance deviation for a configuration marginalized over a given set of partial configurations
	 * 
	 * This function will be mostly used to predict the mean over a given set of instances, but could be used to marginalize over any discrete set of partial configurations.
	 * 
	 * \param features a (partial) configuration where unset values should be set to NaN
	 * \param set_features a array containing the (partial) assignments used for the averaging. Every NaN value will be replaced by the corresponding value from features.
	 * \param set_size number of feature vectors in set_features
	 * 
	 * \return std::pair<num_type, num_type> mean and variance prediction of a feature vector averaged over 
	 */
	std::pair<num_type, num_type> predict_mean_var_marginalized_over_set (num_type *features, num_type* set_features, index_type set_size){
		
		num_type fv[num_features];

		// collect the predictions of individual trees
		rfr::running_statistics<num_type> mean_stats, var_stats;
		for (auto i=0u; i < set_size; ++i){
			// construct the actual feature vector
			rfr::merge_two_vectors(features, &set_features[i*num_features], fv, num_features);

			num_type m , v;
			std::tie(m, v) = predict_mean_var(fv);

			mean_stats(m);
			var_stats(v);
		}
		return(std::pair<num_type, num_type> (mean_stats.mean(), std::max<num_type>(0, mean_stats.variance() + var_stats.mean()) ));
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
	 * \return std::tuple<num_type, num_type, num_type> mean and variance of empirical mean prediction of a feature vector averaged over. The last one is the estimated variance of a sample drawn from partial assignment.
	 */
	std::tuple<num_type, num_type, num_type> predict_mean_var_of_mean_response_on_set (num_type *features, num_type* set_features, index_type set_size){

			num_type fv[num_features];

			rfr::running_statistics<num_type> mean_stats, var_stats, sample_var_stats, sample_mean_stats;

			for (auto &t : the_trees){

					rfr::running_statistics<num_type> tree_mean_stats, tree_var_stats;

					for (auto i=0u; i < set_size; ++i){

							rfr::merge_two_vectors(features, &set_features[i*num_features], fv, num_features);

							num_type m , v; index_type n;
							std::tie(m, v, n) = t.predict_mean_var_N(fv);

							tree_mean_stats(m); tree_var_stats(v); sample_mean_stats(m); sample_var_stats(v);
					}

					mean_stats(tree_mean_stats.mean());
					var_stats(std::max<num_type>(0, tree_var_stats.mean()));
					
			}
			
			return(std::make_tuple(mean_stats.mean(), std::max<num_type>(0, mean_stats.variance()) + std::max<num_type>(0, var_stats.mean()/set_size), std::max<num_type>(0,sample_mean_stats.variance() + sample_var_stats.mean())));
	}


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
	num_type covariance (num_type* f1, num_type* f2){
		rfr::running_statistics<num_type> cov_stats;
		rfr::running_covariance<num_type> run_cov_of_means;
		
		for (auto &t: the_trees){
			auto l1 = t.find_leaf(f1);
			auto l2 = t.find_leaf(f2);
		
			num_type m1 , v1;	index_type n;
			std::tie(m1, v1, n) = t.predict_mean_var_N(f1);

			num_type m2 , v2;
			std::tie(m2, v2, n) = t.predict_mean_var_N(f2);
			
			// assumption here: cov = 0 if the leafs are different, and cov = var if both feature vectors fall into the same leaf
			if (l1 == l2) cov_stats(v2);
			else cov_stats(0);
			
			run_cov_of_means(m1,m2);
		}
		return(cov_stats.mean() + run_cov_of_means.covariance());
	}


	std::vector< std::vector<num_type> > all_leaf_values (num_type * feature_vector){
		std::vector< std::vector<num_type> > rv;
		rv.reserve(the_trees.size());

		for (auto &t: the_trees){
			rv.push_back(t.leaf_entries(feature_vector));
		}
		return(rv);
	}


	forest_options<num_type, response_type, index_type> get_forest_options(){return(forest_opts);}

	std::vector<std::vector< std::vector<num_type> > > partition_of_tree( index_type tree_index,
														std::vector<std::vector<num_type> > pcs){
		return(the_trees[tree_index].partition(pcs));
	}
	
	/* \brief returns the predictions of every tree marginalized over the NAN values in the feature_vector
	 * 
	 * \param feature_vector non-specfied values (NaN) will be marginalized over according to the training data
	 */
	std::vector<num_type> marginalized_predictions(num_type *feature_vector){
		std::vector<num_type> rv;
		rv.reserve(the_trees.size());
		for (auto &t : the_trees)
			rv.emplace_back(t.marginalized_prediction(feature_vector));
		return(rv);
	}

	std::vector<std::vector<std::vector<num_type> > > all_split_values(index_type *types){
		std::vector<std::vector<std::vector<num_type> > > rv;
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
	void pseudo_update (const rfr::data_containers::data_container_base<num_type, response_type, index_type> &data){
		for (auto i=0u; i<data.num_data_points(); ++i){
			auto p = data.retrieve_data_point(i);
			dirty_leafs.emplace_back(std::vector<index_type> (the_trees.size(),0));
			auto it = (dirty_leafs.back()).begin();
			//for each tree
			for (auto &t: the_trees){
		
				index_type index = t.find_leaf(p.data());
		
				// add value
				t.the_nodes[index].push_response_value(data.response(i));
		
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
	
	num_type out_of_bag_error(){return(oob_error);}

	// writes serialized representation into string (used for pickling in python)
	void save_to_binary_file(const std::string filename){
		std::ofstream ofs(filename, std::ios::binary);
		oarch_type oarch(ofs);
		serialize(oarch);
	}

	// deserialize from a representation provided by the string (used for unpickling in python)
	void load_from_binary_file(const std::string filename){
		std::ifstream ifs(filename, std::ios::binary);
		std::cout<<"opening file "<<filename<<std::endl;
		iarch_type iarch(ifs);
		serialize(iarch);
	}

	// serialize into a string; used for Python's pickle.dump
	std::string save_into_string(){
		std::stringstream oss;
		oarch_type oarch(oss);
		serialize(oarch);
		return(oss.str());
	}

	// deserialize from string; used for Python's pickle.load
	void load_from_string( std::string str){
		std::stringstream iss;
		iss.str(str);
		iarch_type iarch(iss);
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
